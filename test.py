import os
import argparse
import torch
from torch.utils.data import DataLoader
from ldm.models.diffusion.ddim import DDIMSampler
import datasets_self.dataset_test as dataset
from models import (
    AutoencoderKL,
    UNetModel,
    LowFieldLatentDiffusion,
    StructuralEncoder,
    SemanticEncoder,
)
from model_utilities import Encoder  # backbone used inside SemanticEncoder
from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
import nibabel as nib
import numpy as np
import pandas as pd


def build_arch():
    # === match training architecture (no need to load separate VAE/CLIP weights) ===
    vae = AutoencoderKL(
        ch=64, out_ch=2, ch_mult=(1, 2, 4), num_res_blocks=3,
        attn_resolutions=[0], dropout=0.0, resamp_with_conv=True,
        in_channels=2, resolution=448, z_channels=8, embed_dim=8
    )

    unet = UNetModel(
        image_size=[112, 112],
        in_channels=8 + 8,  # latent 8 + LF cond 8
        model_channels=128,
        out_channels=8,
        num_res_blocks=3,
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
        legacy=True,
    )

    # semantic encoder uses your Encoder backbone internally
    clip_backbone = Encoder(
        ch=64, ch_mult=(1, 2, 4),
        num_res_blocks=2, attn_resolutions=[],
        dropout=0.1, resamp_with_conv=True, z_channels=128,
        in_channels=2, resolution=224, depth=18,
    )

    low_field_encoder = StructuralEncoder(
        in_channels=2, out_channels=8, hidden_channels=16,
        num_downsamplings=1, kernel_size=3, use_batchnorm=True
    )
    semantic_encoder = SemanticEncoder(clip_encoder=clip_backbone)

    model = LowFieldLatentDiffusion(
        first_stage_model=vae,
        low_field_encoder=low_field_encoder,
        semantic_encoder=semantic_encoder,
        model=unet,
        image_size=448,
        channels=2,
        latent_channels=8,
        latent_size=112,
        timesteps=1000,
        learning_rate=1e-4,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_test_path", type=str, default='C:\\Users\\zhang\\OneDrive - UBC\\project\\enhancement_DR_generation\\grad_cam_figure_new\\temp_org')
    ap.add_argument("--diffusion_ckpt",type=str, default='saved_models\\latent_diffusion_25.ckpt', help="e.g. saved_models/epoch-120.ckpt")
    ap.add_argument("--save_root", default='C:\\Users\\zhang\\OneDrive - UBC\\project\\enhancement_DR_generation\\grad_cam_figure_new\\temp_outputs', type=str)
    ap.add_argument("--batch_size", default=1, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--S", default=50, type=int, help="DDIM steps")
    ap.add_argument("--eta", default=0.0, type=float)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    saved_path = args.save_root
    os.makedirs(saved_path, exist_ok=True)

    # Data
    test_set = dataset.data_set(args.dataset_test_path, args.save_root)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    print("total samples:", len(test_set))

    # Build arch & load the single Lightning checkpoint
    diffusion = build_arch()
    ckpt = torch.load(args.diffusion_ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = diffusion.load_state_dict(sd, strict=False)
    if missing:   print("[Missing]", len(missing), "e.g.", missing[:5])
    if unexpected:print("[Unexpected]", len(unexpected), "e.g.", unexpected[:5])

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    diffusion.to(device).eval()
    print("pretrained model loaded")

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):

            hf_images = batch["hf_image"]
            lf_images = batch["lf_image"]
            sr_images = batch["sr_image"]

            print("hf_images:", hf_images.shape, "sr_images:", sr_images.shape)

            affines = batch["affine"]
            names = batch["name"]
            for b in range(hf_images.shape[0]):
                hf_image = hf_images[b].float()
                lf_image = lf_images[b].float()
                sr_image = sr_images[b].float()
                hf_image = torch.permute(hf_image, (1, 0, 2, 3))
                lf_image = torch.permute(lf_image, (1, 0, 2, 3))
                sr_image = torch.permute(sr_image, (1, 0, 2, 3))

                print("hf ", hf_image.shape, "lf ", lf_image.shape)

                name = names[b]
                affine = affines[b].numpy().astype('float64')
                hf_synth = sr_image.clone()
                for s in range(lf_image.shape[0]):
                    lf_image_slice = lf_image[s:s+1].to(device)

                    c_lf = diffusion.low_field_encoder(lf_image_slice)
                    c_context = diffusion.semantic_encoder(lf_image_slice)
                    cond = {}
                    cond["c_crossattn"] = c_context
                    cond["c_concat"] = [c_lf]

                    sampler = DDIMSampler(diffusion)
                    shape = (c_lf.shape[1], c_lf.shape[2], c_lf.shape[3])
                    samples, _ = sampler.sample(
                        S=args.S,
                        batch_size=1,
                        shape=shape,
                        conditioning=cond,
                        eta=args.eta,
                    )

                    hf_synth_slice = diffusion.decode_first_stage(samples)
                    hf_synth_slice = torch.nn.functional.interpolate(hf_synth_slice, size=hf_synth.shape[-2:])

                    hf_synth[s:s+1, ...] = hf_synth_slice


                hf_synth_t1 = hf_synth[:, 0:1, :, :].cpu()
                hf_synth_t2 = hf_synth[:, 1:2, :, :].cpu()

                sr_t1 = sr_image[:, 0:1, :, :].cpu()
                sr_t2 = sr_image[:, 1:2, :, :].cpu()

                hf_t1 = hf_image[:, 0:1, :, :]
                hf_t2 = hf_image[:, 1:2, :, :]

                brain_mask_t1 = ((hf_synth_t1 > 0) & (sr_t1 > 0)).float()
                brain_mask_t2 = ((hf_synth_t2 > 0) & (sr_t2 > 0)).float()

                hf_t1 = hf_t1 * brain_mask_t1
                hf_t2 = hf_t2 * brain_mask_t2

                hf_synth_t1 = hf_synth_t1 * brain_mask_t1
                hf_synth_t2 = hf_synth_t2 * brain_mask_t2


                hf_synth_t1 = torch.permute(hf_synth_t1, (1, 0, 2, 3))
                hf_synth_t2 = torch.permute(hf_synth_t2, (1, 0, 2, 3))

                sr_t1 = torch.permute(sr_t1, (1, 0, 2, 3))
                sr_t2 = torch.permute(sr_t2, (1, 0, 2, 3))


                hf_synth = torch.cat([hf_synth_t1, hf_synth_t2], dim=1)


                sr_image = torch.cat([sr_t1, sr_t2], dim=1)

                hf_synth_t1 = hf_synth_t1[0].numpy().astype(np.float32)
                hf_synth_t2 = hf_synth_t2[0].numpy().astype(np.float32)

                hf_t1 = hf_image[:, 0, :, :].numpy().astype(np.float32)
                hf_t2 = hf_image[:, 1, :, :].numpy().astype(np.float32)

                sr_t1 = sr_image[:, 0, :, :].numpy().astype(np.float32)
                sr_t2 = sr_image[:, 1, :, :].numpy().astype(np.float32)

                hf_synth_t1 = hf_synth_t1.transpose(1, 2, 0)
                hf_synth_t2 = hf_synth_t2.transpose(1, 2, 0)

                hf_t1 = hf_t1.transpose(1, 2, 0)
                hf_t2 = hf_t2.transpose(1, 2, 0)

                sr_t1 = sr_t1.transpose(1, 2, 0)
                sr_t2 = sr_t2.transpose(1, 2, 0)

                hf_synth_t1_nii = nib.Nifti1Image(hf_synth_t1, affine)
                hf_synth_t2_nii = nib.Nifti1Image(hf_synth_t2, affine)

                nib.save(hf_synth_t1_nii, os.path.join(saved_path, name.replace("HF_T1", "Syn_T1")))
                nib.save(hf_synth_t2_nii, os.path.join(saved_path, name.replace("HF_T1", "Syn_T2")))

if __name__ == "__main__":
    main()
