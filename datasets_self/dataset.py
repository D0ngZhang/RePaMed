import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torchio as tio
import nibabel as nib
from scipy.ndimage import gaussian_filter1d
import re
import pandas as pd


mean = np.array([0.5,])
std = np.array([0.5,])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(1):
        tensors[:, c, ...].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


def _make_image_namelist(dir):
    img_path = []
    namelist = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            print(fname)
            if fname.endswith('t1_stripped.nii.gz'):
                item_path = os.path.join(root, fname)
                if os.path.exists(item_path.replace('t1_stripped', 't2_stripped')) and os.path.exists(item_path.replace('t1_stripped', 't1_fine')):
                    namelist.append(fname)
                    img_path.append(item_path)

    return namelist, img_path


def fix_nifti_direction_cosines(file_path, output_path):
    img = nib.load(file_path)
    header = img.header.copy()
    affine = img.affine.copy()

    # 修复方向余弦矩阵
    R = affine[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_fixed = np.dot(U, Vt)
    affine[:3, :3] = R_fixed

    fixed_img = nib.Nifti1Image(img.get_fdata(), affine, header)
    nib.save(fixed_img, output_path)
    return output_path

class data_set(dataset_torch):
    def __init__(self, root):
        self.root = root

        self.img_names, self.img_paths = _make_image_namelist(os.path.join(self.root))

        self.epi = 0
        self.img_num = len(self.img_names)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self):
        return len(self.img_names)


    def load_img(self, path):
        try:
            img = tio.ScalarImage(path)
        except RuntimeError as e:
            print(f"Error reading {path}: {e}")
            fixed_path = fix_nifti_direction_cosines(path, path)
            try:
                img = tio.ScalarImage(fixed_path)
            except RuntimeError as e:
                print(f"Failed to read fixed image {fixed_path}: {e}")
                return None
        return img

    def __getitem__(self, index):
        path_img_c1_t1 = self.img_paths[index]
        path_img_c1_t2 = path_img_c1_t1.replace('t1', 't2')
        case_name = self.img_names[index]

        img_c1_t1 = self.load_img(path_img_c1_t1)
        img_c1_t2 = self.load_img(path_img_c1_t2)

        mask_c1 = self.load_img(path_img_c1_t1.replace('t1_stripped', 't1_fine'))

        mask_c1.data = combine_to_4_classes(mask_c1.data)

        img_c1_t1.data = (img_c1_t1.data + 1) / 2
        img_c1_t2.data = (img_c1_t2.data + 1) / 2

        img_c1_t1_lf, _ = low_field_simulation_t1wi(img_c1_t1, mask_c1)
        img_c1_t2_lf, _ = low_field_simulation_t2wi(img_c1_t2, mask_c1)

        img_c1_t1_hf_data = img_c1_t1.data
        img_c1_t2_hf_data = img_c1_t2.data

        img_c1_t1_lf_data = img_c1_t1_lf.data
        img_c1_t2_lf_data = img_c1_t2_lf.data

        img_c1_t1_hf_data = torch.permute(img_c1_t1_hf_data, (0, 3, 1, 2))
        img_c1_t2_hf_data = torch.permute(img_c1_t2_hf_data, (0, 3, 1, 2))


        img_c1_t1_lf_data = torch.permute(img_c1_t1_lf_data, (0, 3, 1, 2))
        img_c1_t2_lf_data = torch.permute(img_c1_t2_lf_data, (0, 3, 1, 2))

        img_c1_t1_lf_data = torch.clamp(img_c1_t1_lf_data, 0, 1)
        img_c1_t2_lf_data = torch.clamp(img_c1_t2_lf_data, 0, 1)

        num = random.randint(1, 17)

        img_c1_hf_data = torch.cat((img_c1_t1_hf_data[:, num, ...], img_c1_t2_hf_data[:, num, ...]), dim=0)
        img_c1_lf_data = torch.cat((img_c1_t1_lf_data[:, num, ...], img_c1_t2_lf_data[:, num, ...]), dim=0)

        imgs = {"hf_image": img_c1_hf_data, "lf_image": img_c1_lf_data, "affine": img_c1_t1.affine}
        return imgs


def combine_to_4_classes(freesurfer_mask):

    new_mask = np.zeros_like(freesurfer_mask, dtype=np.uint8)

    wm_labels = [1,2,3,4,5,6,7,8]
    for val in wm_labels:
        new_mask[freesurfer_mask == val] = 1

    gm_labels = [9,10,11,12,13,14,15,16,17,18,19,20]
    for val in gm_labels:
        new_mask[freesurfer_mask == val] = 2

    csf_labels = [23,24,25]
    for val in csf_labels:
        new_mask[freesurfer_mask == val] = 3

    others_labels = [21, 22]
    for val in others_labels:
        new_mask[freesurfer_mask == val] = 4

    new_mask[~np.isin(freesurfer_mask, wm_labels + gm_labels + csf_labels + others_labels)] = 4

    return new_mask


def add_artifact_to_image(img: tio.ScalarImage, ghosting_prob=0.3, spike_prob=0.3, bias_field_prob=0.1, seed=None):
    """
    添加常见真实低场MRI中的非结构性artifact：
    - Ghosting (轻度)
    - Spike（高频射频干扰）
    - Bias field（非中心对称型，可选）
    """
    if seed is not None:
        random.seed(seed)

    img.data = img.data * 255.0

    transforms = []

    if random.random() < ghosting_prob:
        transforms.append(
            tio.RandomGhosting(
                num_ghosts=(1, 3),  # 1~3 个ghost
                axes=(0, 1, 2),
                intensity=(0.01, 0.05),
                p=1
            )
        )

    if random.random() < spike_prob:
        transforms.append(
            tio.RandomSpike(
                num_spikes=1,
                intensity=(0.03, 0.07),
                p=1
            )
        )

    if random.random() < bias_field_prob:
        transforms.append(
            tio.RandomBiasField(
                coefficients=0.2,
                order=2,
                p=1
            )
        )

    composite = tio.Compose(transforms)
    img = composite(img)

    img.data = torch.clamp(img.data / 255.0, 0, 1)

    return img


def low_field_simulation_t1wi(
    img,
    mask,
    blur_std=None,
    bias_coefficient=None,
    bias_degree=None,
    sampled_snrs=None,
):
    repeat = True

    if blur_std is None:
        repeat = False
        blur_std = random.uniform(0.5, 1.2)
        bias_coefficient = random.uniform(0.08, 0.25)
        bias_degree = random.choice([2, 3, 4])
        snr_means = [9, 11, 5, 7]
        snr_cov = np.array([[8, 4, 2, 3],
                            [4, 10, 2, 3],
                            [2, 2, 6, 1],
                            [3, 3, 1, 8]])
        sampled_snrs = np.random.multivariate_normal(snr_means, snr_cov)
        sampled_snrs = np.clip(sampled_snrs, 2.5, 25.0)
    else:
        blur_std = np.clip(blur_std, a_min=0.5, a_max=1.2)
        bias_coefficient = np.clip(bias_coefficient, a_min=0.08, a_max=0.25)
        bias_degree = int(np.clip(bias_degree, a_min=0.5, a_max=1.0) * 4)
        sampled_snrs = np.clip(sampled_snrs, a_min=0.1, a_max=1.0) * 25

    target_size = (224, 224, 18)
    resize_transform = tio.Resize(target_shape=target_size, image_interpolation='linear')
    img = resize_transform(img)
    resize_transform = tio.Resize(target_shape=target_size, image_interpolation='nearest')
    mask = resize_transform(mask)

    # 2. 初始化仿真图像
    img_data = img.data.clone()  # 保持 img 的原始数据不被修改
    img_affine = img.affine
    mask = mask.data

    # 3. 遍历每个组织，按 SNR 添加噪声
    for region_label, snr in enumerate(sampled_snrs, start=1):
        region_mask = mask == region_label

        if not region_mask.any():
            continue

        # 计算噪声强度
        region_signal_mean = img_data[region_mask].mean()
        if region_signal_mean <= 0 or snr <= 1e-6:
            noise_std = 0.0
        else:
            noise_std = region_signal_mean / snr

        # 添加噪声
        noise = np.random.normal(0, noise_std, size=region_mask.sum()).astype(np.float32)
        img_data[region_mask] += noise

    # 4. 添加方向性模糊 (针对 z 轴 = d 维度)
    blur_sigma_z = blur_std  # 在 z 轴方向的模糊标准差
    img_data = img_data.numpy()  # 转为 numpy 数组以使用 scipy.ndimage
    img_data = gaussian_filter1d(img_data, sigma=blur_sigma_z, axis=3)  # 对 z 轴模糊
    img_data = torch.tensor(img_data)  # 转回 tensor 格式

    # 5. 添加全局偏场
    bias_transform = tio.RandomBiasField(coefficients=bias_coefficient, order=bias_degree)
    img_data = bias_transform(tio.ScalarImage(tensor=img_data, affine=img_affine)).data

    # 6. 参数记录与裁剪
    params = np.zeros(7)
    bias_degree = bias_degree / 4
    sampled_snrs = sampled_snrs / 35

    params[0] = blur_std
    params[1] = bias_coefficient
    params[2] = bias_degree
    params[3:] = sampled_snrs

    img_data = torch.clip(img_data, min=0, max=1)  # 裁剪到 0~1 范围

    if repeat:
        return tio.ScalarImage(tensor=img_data, affine=img_affine)
    else:
        return tio.ScalarImage(tensor=img_data, affine=img_affine), params


def low_field_simulation_t2wi(
    img,
    mask,
    blur_std=None,
    bias_coefficient=None,
    bias_degree=None,
    sampled_snrs=None
):
    repeat = True

    if blur_std is None:

        repeat = False

        blur_std = random.uniform(0.6, 1.3)
        bias_coefficient = random.uniform(0.1, 0.28)
        bias_degree = random.choice([2, 3, 4])
        snr_means = [10, 12, 5, 8]  # GM, WM, CSF, Other 的 SNR 均值
        snr_cov = np.array([
            [7, 4, 2, 3],
            [4, 9, 2, 3],
            [2, 2, 6, 1],
            [3, 3, 1, 7],
        ])
        sampled_snrs = np.random.multivariate_normal(snr_means, snr_cov)
        sampled_snrs = np.clip(sampled_snrs, 2.5, 30.0)
    else:
        blur_std = np.clip(blur_std, a_min=0.6, a_max=1.3)
        bias_coefficient = np.clip(bias_coefficient, a_min=0.1, a_max=0.28)
        bias_degree = int(bias_degree * 4)
        sampled_snrs = np.clip(sampled_snrs, a_min=0.08333, a_max=1.0) * 30

    target_size = (224, 224, 18)
    resize_transform = tio.Resize(target_shape=target_size, image_interpolation='linear')
    img = resize_transform(img)
    resize_transform = tio.Resize(target_shape=target_size, image_interpolation='nearest')
    mask = resize_transform(mask)

    img_data = img.data.clone()  # 保持 img 的原始数据不被修改
    img_affine = img.affine
    mask = mask.data

    # 3. 遍历每个组织，按 SNR 添加噪声
    for region_label, snr in enumerate(sampled_snrs, start=1):
        region_mask = mask == region_label

        if not region_mask.any():
            continue

        # 计算噪声强度
        region_signal_mean = img_data[region_mask].mean()
        if region_signal_mean <= 0 or snr <= 1e-6:
            noise_std = 0.0
        else:
            noise_std = region_signal_mean / snr

        # 添加噪声
        noise = np.random.normal(0, noise_std, size=region_mask.sum()).astype(np.float32)
        img_data[region_mask] += noise

    # 4. 添加方向性模糊 (针对 z 轴 = d 维度)
    blur_sigma_z = blur_std  # 在 z 轴方向的模糊标准差
    img_data = img_data.numpy()  # 转为 numpy 数组以使用 scipy.ndimage
    img_data = gaussian_filter1d(img_data, sigma=blur_sigma_z, axis=3)  # 对 z 轴模糊
    img_data = torch.tensor(img_data)  # 转回 tensor 格式

    # 5. 添加全局偏场
    bias_transform = tio.RandomBiasField(coefficients=bias_coefficient, order=bias_degree)
    img_data = bias_transform(tio.ScalarImage(tensor=img_data, affine=img_affine)).data

    # 6. 参数记录与裁剪
    params = np.zeros(7)
    bias_degree = bias_degree / 4
    sampled_snrs = sampled_snrs / 30

    params[0] = blur_std
    params[1] = bias_coefficient
    params[2] = bias_degree
    params[3:] = sampled_snrs

    img_data = torch.clip(img_data, min=0, max=1)  # 裁剪到 0~1 范围

    if repeat:
        return tio.ScalarImage(tensor=img_data, affine=img_affine)
    else:
        return tio.ScalarImage(tensor=img_data, affine=img_affine), params

