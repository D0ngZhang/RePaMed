import torch
import torch.nn as nn
from ldm.modules.ema import LitEma
import torch.distributed as dist

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from torchvision import models
from utils.loss import MSE_kspace_loss
import os
from torchvision.utils import save_image
from ldm.modules.diffusionmodules.openaimodel import *

from torchvision.transforms.functional import to_pil_image
import imageio
from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
from ldm.models.diffusion.ddpm import LatentDiffusion, DiffusionWrapper
from ldm.models.diffusion.ddim import DDIMSampler


class LowFieldLatentDiffusion(LatentDiffusion):
    def __init__(self,
                 first_stage_model: nn.Module,
                 low_field_encoder: nn.Module,
                 model: nn.Module,
                 semantic_encoder: nn.Module = None,
                 image_size: int = 448,
                 latent_size: int = 112,
                 channels: int = 2,
                 latent_channels: int = 8,
                 timesteps: int = 1000,
                 learning_rate: float = 1e-4,
                 parameterization="x0",
                 **kwargs):
        super().__init__(unet=model, first_stage_config=None, cond_stage_config=None,
                         timesteps=timesteps, image_size=image_size, channels=channels, **kwargs)
        """
        parameterization='x0'
        beta_schedule='cosine', 
        Args:
            first_stage_model (nn.Module): pretrained VAE to encode HF, frozen 
            low_field_encoder (nn.Module): the LF encoder to be trained
            unet (nn.Module): latent diffusion conditional UNet。
            text_encoder (nn.Module, optional): optional text encoder to extract text features
            image_size (int): the size of image
            channels (int): the channels of image
            timesteps (int): the number of timesteps
        """

        self.first_stage_model = first_stage_model
        self.low_field_encoder = low_field_encoder
        self.semantic_encoder = semantic_encoder

        self.first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        self.cond_stage_model = None
        self.conditioning_key = "hybrid"

        self.model = DiffusionWrapper(self.model, self.conditioning_key)
        self.model_ema = LitEma(self.model.diffusion_model)

        self.learning_rate = learning_rate
        self.latent_channels = latent_channels
        self.latent_size = latent_size

    def instantiate_first_stage(self, config):
        pass

    def instantiate_cond_stage(self, config):
        pass

    def get_input(self, batch, k=None, **kwargs):

        hf = batch["hf_image"].to(self.device)
        lf = batch["lf_image"].to(self.device)

        encoder_posterior = self.first_stage_model.encode(hf)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        c_lf = self.low_field_encoder(lf)

        if self.semantic_encoder is not None:
            c_context = self.semantic_encoder(lf)
        cond = {}
        if c_context is not None:
            cond["c_crossattn"] = c_context

        cond["c_concat"] = [c_lf]

        return z, cond


    def forward(self, x, c):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters()) + list(self.low_field_encoder.parameters())
        if self.semantic_encoder is not None:
            params += list(self.semantic_encoder.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)


        return opt

    def sample_ddim(self, batch, S=50, batch_size=8, eta=0.0):
        """
        使用 DDIM 采样方法生成图像。

        参数:
            batch: 包含 "hf_image" 和 "lf_image"（可选 "text"）的字典，用于计算条件信息。
            S: DDIM 采样步数（可少于训练时的总步数，通常可设置为 50～100）。
            batch_size: 采样时的批次大小。
            eta: 控制随机性（eta=0时采样确定）。

        返回:
            decoded_samples: 解码后的生成图像。
            intermediates: 采样过程中保存的中间 latent 表示（可用于调试或可视化）。
        """
        # 仅利用条件信息
        _, c = self.get_input(batch)
        shape = (self.latent_channels, self.latent_size, self.latent_size)

        # 初始化 DDIMSampler
        sampler = DDIMSampler(self)
        samples, intermediates = sampler.sample(
            S=S,
            batch_size=batch_size,
            shape=shape,
            conditioning=c,
            eta=eta
        )
        # 解码 latent 表示得到图像
        decoded_samples = self.decode_first_stage(samples)
        return decoded_samples, intermediates

    def calculate_metrics(self, pred, target):
        psnr = psnr_metric(pred, target)
        ssim = ssim_metric(pred, target)
        return psnr.item(), ssim.item()

    def save_diffusion_iterations(self, batch, save_dir, batch_idx, N=8, log_every_t=10, S=100, eta=0.0):
        """
        保存扩散过程图像（GIF）、最终结果图像（PNG），并计算 PSNR 和 SSIM。
        返回：psnr, ssim
        """
        is_main_process = True
        if dist.is_available() and dist.is_initialized():
            is_main_process = dist.get_rank() == 0

        if is_main_process:
            os.makedirs(save_dir, exist_ok=True)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        target_img = batch["hf_image"].to(self.device)
        input_img = batch["lf_image"].to(self.device)
        input_img = F.interpolate(input_img, size=target_img.shape[-2:], mode="bilinear")
        z, c = self.get_input(batch)

        decoded_samples, intermediates = self.sample_ddim(batch, S=S, batch_size=N, eta=eta)
        restored_img = self.decode_first_stage(z)

        x_inter = intermediates["x_inter"]  # (S+1, N, C, H, W)
        timesteps = list(range(0, len(x_inter)))
        for i, t in enumerate(timesteps):
            latent_t = x_inter[i]
            pred_img = self.decode_first_stage(latent_t)
            filename = os.path.join(save_dir, f"iter_{t:03d}.png")
            saved_img = torch.cat([input_img, restored_img, pred_img, target_img], dim=-1)
            save_image(saved_img, filename, nrow=1, normalize=True)

        psnr_pred2restore = psnr_metric(pred_img, restored_img).item()
        ssim_pred2restore = ssim_metric(pred_img, restored_img).item()

        psnr_restore2target = psnr_metric(restored_img, target_img).item()
        ssim_restore2target = ssim_metric(restored_img, target_img).item()

        psnr_pred2target = psnr_metric(pred_img, target_img).item()
        ssim_pred2target = ssim_metric(pred_img, target_img).item()

        return psnr_pred2restore, ssim_pred2restore, psnr_restore2target, ssim_restore2target, psnr_pred2target, ssim_pred2target

    def validation_step(self, batch, batch_idx):
        """
        每个验证 batch 都保存图像，并记录 PSNR / SSIM。
        """
        z, c = self.get_input(batch)
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        loss, loss_dict = self.p_losses(z, c, t)

        # 保存图像、过程，并计算 PSNR / SSIM
        save_dir = os.path.join("validation_diffusion_iters", f"epoch_{self.current_epoch}")
        psnr_p2r, ssim_p2r, psnr_r2t, ssim_r2t, psnr_p2t, ssim_p2t = self.save_diffusion_iterations(batch, save_dir, batch_idx, N=z.shape[0])

        metrics = {
            "val/psnr_p2r": psnr_p2r,
            "val/ssim_p2r": ssim_p2r,
            "val/psnr_r2t": psnr_r2t,
            "val/ssim_r2t": ssim_r2t,
            "val/psnr_p2t": psnr_p2t,
            "val/ssim_p2t": ssim_p2t,
        }

        loss_dict.update(metrics)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        z, c = self.get_input(batch)
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        # loss, loss_dict = self.p_losses(z, c, t)

        save_dir = os.path.join("test_diffusion_iters", f"epoch_{self.current_epoch}")
        psnr_p2r, ssim_p2r, psnr_r2t, ssim_r2t, psnr_p2t, ssim_p2t = self.save_diffusion_iterations(
            batch, save_dir, batch_idx, N=z.shape[0]
        )

        metrics = {
            "test/psnr_p2r": psnr_p2r,
            "test/ssim_p2r": ssim_p2r,
            "test/psnr_r2t": psnr_r2t,
            "test/ssim_r2t": ssim_r2t,
            "test/psnr_p2t": psnr_p2t,
            "test/ssim_p2t": ssim_p2t,
        }

        print(metrics)

        self.test_outputs.append(metrics)

        return metrics

    def on_test_start(self):
        # 每次 test 开始前，清空指标列表
        self.test_outputs = []

    def on_test_epoch_end(self):
        # 汇总 self.test_outputs 里的所有 batch 指标
        psnr_p2r = torch.tensor([x["test/psnr_p2r"] for x in self.test_outputs]).mean().item()
        ssim_p2r = torch.tensor([x["test/ssim_p2r"] for x in self.test_outputs]).mean().item()
        psnr_r2t = torch.tensor([x["test/psnr_r2t"] for x in self.test_outputs]).mean().item()
        ssim_r2t = torch.tensor([x["test/ssim_r2t"] for x in self.test_outputs]).mean().item()
        psnr_p2t = torch.tensor([x["test/psnr_p2t"] for x in self.test_outputs]).mean().item()
        ssim_p2t = torch.tensor([x["test/ssim_p2t"] for x in self.test_outputs]).mean().item()

        print("\n================= Final Test Results =================")
        print(f"PSNR P2R: {psnr_p2r:.2f} | SSIM P2R: {ssim_p2r:.4f}")
        print(f"PSNR R2T: {psnr_r2t:.2f} | SSIM R2T: {ssim_r2t:.4f}")
        print(f"PSNR P2T: {psnr_p2t:.2f} | SSIM P2T: {ssim_p2t:.4f}")
        print("======================================================\n")


class StructuralEncoder(nn.Module):
    """
    将 2 通道输入 (224×224) 压缩到 28×28 空间分辨率，并输出 128 通道。
    """

    def __init__(
            self,
            in_channels: int = 2,  # 输入 MRI 通道数
            out_channels: int = 16,  # 最终输出通道数
            hidden_channels: int = 64,
            num_downsamplings: int = 3,
            kernel_size: int = 3,
            use_batchnorm: bool = True
    ):
        """
        :param in_channels: 输入通道数 (低场MRI可能是2通道)
        :param out_channels: 输出通道数 (希望得到 128)
        :param hidden_channels: 下采样过程中的基准通道数
        :param num_downsamplings: 下采样次数，每次 stride=2，使分辨率减半
                                  224 -> 112 -> 56 -> 28 (一共3次)
        :param kernel_size: 卷积核大小 (默认3)
        :param use_batchnorm: 是否插入 BatchNorm2d
        """
        super().__init__()

        layers = []
        prev_ch = in_channels

        for i in range(num_downsamplings):
            # 第一次卷积， stride=2 做下采样
            layers.append(
                nn.Conv2d(prev_ch, 4**i * hidden_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(4**i * hidden_channels))
            layers.append(nn.ReLU(inplace=True))

            # 再加一层卷积 (stride=1) 提升特征表达
            layers.append(
                nn.Conv2d(4**i * hidden_channels, 4**i * hidden_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(4**i * hidden_channels))
            layers.append(nn.ReLU(inplace=True))

            prev_ch = 4**i * hidden_channels

        # 最后用 1×1 卷积把 hidden_channels -> out_channels
        layers.append(
            nn.Conv2d(prev_ch, out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.structural_encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        structural_features = self.structural_encoder(x)

        return structural_features

class TokenProjector(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.proj(x)

class SemanticEncoder(nn.Module):


    def __init__(self, clip_encoder, in_channels=[64, 128, 256], out_dim=768, token_H=28, token_W=28):
        super().__init__()

        self.clip_encoder = clip_encoder
        self.clip_encoder.eval()
        self.out_dim = out_dim
        self.token_H = token_H
        self.token_W = token_W
        self.proj_layers = nn.ModuleList([
            TokenProjector(in_ch, out_dim)
            for in_ch in in_channels
        ])
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )


    def forward(self, inputs):  # features = [f1, f2, f3]

        features = self.clip_encoder(inputs)

        token_list = []
        avg_tokens = []

        for i, feat in enumerate(features):
            x = self.proj_layers[i](feat)  # [B, D, H, W]
            x = F.adaptive_avg_pool2d(x, (self.token_H, self.token_W))  # → [B, D, 28, 28]
            token_list.append(x.view(x.size(0), self.out_dim, -1).transpose(1, 2))  # → [B, 784, D]
            avg_tokens.append(x.mean(dim=(2, 3)))  # [B, D]

        w = torch.stack(avg_tokens, dim=1).mean(-1)  # [B, 3]
        w = torch.softmax(self.mlp(w), dim=1)  # [B, 3]

        out = sum(w[:, i].unsqueeze(-1).unsqueeze(-1) * token_list[i] for i in range(3))  # [B, 784, D]
        return out


class AutoencoderKL(nn.Module):
    def __init__(self, ch=64, out_ch=2, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions = [28], dropout=0.1, resamp_with_conv=True, in_channels=2,
                 resolution=224, z_channels=32, embed_dim=32, ckpt_path=None, ignore_keys=[], image_key="image",
                 colorize_nlabels=None):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(
            ch=ch, out_ch=z_channels, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions,
            dropout=dropout, resamp_with_conv=resamp_with_conv,
            in_channels=in_channels, resolution=resolution, z_channels=z_channels
        )

        self.decoder = Decoder(
            ch=ch, out_ch=out_ch, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions,
            dropout=dropout, resamp_with_conv=resamp_with_conv,
            in_channels=z_channels, resolution=resolution, z_channels=z_channels
        )
        self.quant_conv = nn.Conv2d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 3), int(in_width / 2 ** 3)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class AutoencoderKLLoss(nn.Module):
    def __init__(self, kl_weight=0.000001, kspace_weight=0.00001, pixel_weight=1.0):

        super().__init__()
        self.pixel_weight = pixel_weight
        self.kl_weight = kl_weight
        self.kspace_weight = kspace_weight
        self.gan_weight = 0.01
        self.percept_weight = 0.05
        self.criterion_GAN = nn.MSELoss()

    def forward(self, inputs, reconstructions, posterior, discriminator, percept):


        valid = np.ones([inputs.size(0), 1, int(inputs.size(2) / 2 ** 3), int(inputs.size(3) / 2 ** 3)])
        valid = torch.from_numpy(valid).float().to(inputs.device)

        pred_real = discriminator(torch.cat((inputs, inputs), 1)).detach()
        pred_fake = discriminator(torch.cat((inputs, reconstructions), 1))

        # Adversarial loss (relativistic average GAN)
        GAN_loss = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        pixel_loss = F.l1_loss(inputs, reconstructions)
        kspace_loss = MSE_kspace_loss(reconstructions, inputs)
        kl_loss = posterior.kl().mean()

        perceptual_loss = percept(reconstructions, inputs)

        # 4. 组合最终损失
        total_loss = (
            self.pixel_weight * pixel_loss +
            self.kspace_weight * kspace_loss +
            self.kl_weight * kl_loss + self.gan_weight * GAN_loss + self.percept_weight * perceptual_loss
        )

        return total_loss, {
            "pixel_loss": pixel_loss.detach().item(),
            "kspace_loss": kspace_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
            "perceptual_loss": perceptual_loss.detach().item(),
            "GAN_loss": GAN_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
        }


class VGGPerceptualLoss(nn.Module):
    def __init__(self, Local=True, feature_layers=('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'), weights=None):
        super(VGGPerceptualLoss, self).__init__()
        if Local:
            vgg = models.vgg19(pretrained=True).features
            torch.save(vgg.state_dict(), "vgg19.pth")
        else:
            vgg = models.vgg19(pretrained=False).features
            vgg.load_state_dict(torch.load("vgg19.pth", map_location="cpu"))

        self.feature_layers = feature_layers

        self.slice1 = nn.Sequential(*[vgg[x] for x in range(3)])  # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(3, 8)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(8, 17)])  # relu3_3
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(17, 26)])  # relu4_3

        for param in self.parameters():
            param.requires_grad = False

        if weights is None:
            self.weights = {layer: 1.0 for layer in feature_layers}
        else:
            self.weights = weights

    def forward(self, x, y):

        x_features = {}
        y_features = {}

        h_x = self.normalize(x)
        h_y = self.normalize(y)
        h_x = self.slice1(h_x)
        h_y = self.slice1(h_y)
        x_features['relu1_2'] = h_x
        y_features['relu1_2'] = h_y
        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        x_features['relu2_2'] = h_x
        y_features['relu2_2'] = h_y
        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        x_features['relu3_3'] = h_x
        y_features['relu3_3'] = h_y
        h_x = self.slice4(h_x)
        h_y = self.slice4(h_y)
        x_features['relu4_3'] = h_x
        y_features['relu4_3'] = h_y

        loss = 0.0
        for layer in self.feature_layers:
            loss += self.weights[layer] * F.l1_loss(x_features[layer], y_features[layer])
        return loss

    def normalize(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            x = torch.cat([x, x[:, :1]], dim=1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std
