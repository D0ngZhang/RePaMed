import os
import argparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.strategies import DDPStrategy
import numpy as np

import datasets_self.dataset
import pytorch_lightning as pl
from models import AutoencoderKL, UNetModel, LowFieldLatentDiffusion, StructuralEncoder, SemanticEncoder
from model_utilities import Encoder

Sockeye = False
Local = False
x315 = False
RTX5090 = True

RTX50902 = False

MultiGPUs = False

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("images/validation", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--fold", type=int, default=0, help="n out of 5 folds to train")


if Local:
    parser.add_argument("--dataset_path", type=str, default="E:\\RAE_data\\high_field_trainset", help="path of the dataset")
    parser.add_argument("--dataset_test_path", type=str, default="E:\\RAE_data\\high_field_testset", help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")

elif RTX5090:
    parser.add_argument("--dataset_path", type=str, default="/mnt/ssd/high_field_trainset",
                        help="path of the dataset")
    parser.add_argument("--dataset_test_path", type=str, default="/mnt/ssd/high_field_testset",
                        help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

elif Sockeye:
    parser.add_argument("--dataset_path", type=str, default="/arc/project/st-zjanew-1/donzhang/RAE/high_field_trainset",
                        help="path of the dataset")
    parser.add_argument("--dataset_test_path", type=str, default="/arc/project/st-zjanew-1/donzhang/RAE/high_field_testset",
                        help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
else:
    raise ValueError("no machine selected")

if MultiGPUs:
    parser.add_argument("--dist", type=bool, default=1, help="distribute or regular")
    parser.add_argument("--local_rank", default=-1, type=int)

parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")

parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--grad_accum_steps", type=int, default=1, help="interval between saving image samples")
opt = parser.parse_args()
print(opt)

torch.autograd.set_detect_anomaly(True)

def main():

    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    print("================")

    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    train_dataset = datasets_self.dataset.data_set(opt.dataset_path)
    val_dataset = datasets_self.dataset.data_set(opt.dataset_test_path)


    train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
            pin_memory=True,
            drop_last=True
        )
    val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
            pin_memory=True,
            drop_last=False
        )

    print(f"training data: {len(train_loader)} | testing data: {len(val_loader)}")

    first_stage_model = AutoencoderKL(ch=64, out_ch=2, ch_mult=(1, 2, 4), num_res_blocks=3,
                 attn_resolutions = [0], dropout=0.0, resamp_with_conv=True, in_channels=2,
                 resolution=448, z_channels=8, embed_dim=8,)
    ckpt_path = 'saved_models/vae_93.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    first_stage_model.load_state_dict(state_dict)

    model = UNetModel(image_size=[112, 112], in_channels=8 + 8, model_channels=128, out_channels=8, num_res_blocks=3,
                     attention_resolutions=[14, 7],
                     dropout=0.2,
                     channel_mult=(1, 2, 4, 8),
                     conv_resample=True,
                     dims=2,
                     num_classes=None,
                     use_checkpoint=False,
                     use_fp16=False,
                     num_heads=4,
                     num_head_channels=-1,
                     num_heads_upsample=-1,
                     use_scale_shift_norm=False,
                     resblock_updown=False,
                     use_new_attention_order=False,
                     use_spatial_transformer=True,
                     transformer_depth=3,
                     context_dim=768,
                     n_embed=None,
                     legacy=True)

    state_dict = torch.load("saved_models/expert_128.pth", map_location="cpu")
    encoder_state_dict = {
        k.replace("hf_img_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("hf_img_encoder.")
    }

    model_clip = Encoder(
        ch=64, ch_mult=(1, 2, 4),
        num_res_blocks=2, attn_resolutions=[],
        dropout=0.1, resamp_with_conv=True, z_channels=128,
        in_channels=2, resolution=224, depth=18,
    )

    model_clip.load_state_dict(encoder_state_dict)

    low_field_encoder = StructuralEncoder(in_channels = 2,
            out_channels = 8,
            hidden_channels = 16,
            num_downsamplings = 1,
            kernel_size = 3,
            use_batchnorm = True)

    semantic_encoder = SemanticEncoder(clip_encoder = model_clip)

    diffusion = LowFieldLatentDiffusion(first_stage_model=first_stage_model,
                 low_field_encoder=low_field_encoder, semantic_encoder=semantic_encoder,
                 model=model, image_size=448, channels=2, latent_channels=8, latent_size=112, timesteps=1000, learning_rate=opt.lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models/',
        filename='epoch-{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=1
    )

    if MultiGPUs:
        trainer = pl.Trainer(callbacks=checkpoint_callback, max_epochs=opt.n_epochs, accelerator="gpu", devices=4, gradient_clip_val=1.0,
                             strategy=DDPStrategy(find_unused_parameters=True), logger=True)
        trainer.fit(diffusion, train_loader, val_loader)
    else:
        trainer = pl.Trainer(callbacks=checkpoint_callback, max_epochs=opt.n_epochs, accelerator="gpu", devices=1, accumulate_grad_batches=4, gradient_clip_val=1.0, logger=True)
        trainer.fit(diffusion, train_loader, val_loader)

if __name__ == "__main__":
    main()
