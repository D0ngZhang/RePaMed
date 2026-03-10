import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torchio as tio
import nibabel as nib
from scipy.ndimage import gaussian_filter1d


mean = np.array([0.5,])
std = np.array([0.5,])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(1):
        tensors[:, c, ...].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


def _make_image_namelist(dir, val_namelist=None):
    img_path = []
    namelist = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('t1_stripped.nii.gz'):
                if val_namelist is not None and fname in val_namelist:
                    continue
                item_path = os.path.join(root, fname)
                if os.path.exists(item_path.replace('t1', 't2')) and os.path.exists(item_path.replace('t1_stripped', 't1_fine')):
                    namelist.append(fname)
                    img_path.append(item_path)

            if len(namelist) >= 60000:
                break

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
    def __init__(self, root, mode='train'):
        self.root = root
        self.root_validation = self.root.replace('high_field_trainset', 'high_field_testset')
        assert mode in ['train', 'val']
        self.mode = mode
        val_names, _ = _make_image_namelist(self.root_validation)
        self.val_namelist = set(val_names)  # 转成集合，加速查找

        if self.mode == 'train':
            self.img_names, self.img_paths = _make_image_namelist(self.root, val_namelist=self.val_namelist)
        else:
            self.img_names, self.img_paths = _make_image_namelist(self.root_validation)

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
            # 尝试修复 NIfTI 文件并重新读取
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

        # target_size = (224, 224, 18)
        # resize_transform = tio.Resize(target_shape=target_size, image_interpolation='linear')
        # img_c1_t1 = resize_transform(img_c1_t1)
        # img_c1_t2 = resize_transform(img_c1_t2)

        # target_size = (384, 384, 16)
        # center_crop_or_pad = tio.CropOrPad(target_shape=target_size)
        # img_c1_t1 = center_crop_or_pad(img_c1_t1)
        # img_c1_t2 = center_crop_or_pad(img_c1_t2)

        img_c1_t1.data = (img_c1_t1.data + 1)/2
        img_c1_t2.data = (img_c1_t2.data + 1)/2

        img_c1_t1_hf_data = img_c1_t1.data
        img_c1_t2_hf_data = img_c1_t2.data

        img_c1_t1_hf_data = torch.permute(img_c1_t1_hf_data, (0, 3, 1, 2))
        img_c1_t2_hf_data = torch.permute(img_c1_t2_hf_data, (0, 3, 1, 2))

        imgs = {"hf-c1-t1": img_c1_t1_hf_data, "hf-c1-t2": img_c1_t2_hf_data, "affine": img_c1_t1.affine}
        return imgs


def combine_to_4_classes(freesurfer_mask):
    """
    将 Freesurfer 的 25 个标签合并成 4 类:
        1 => White Matter (WM)
        2 => Gray Matter (GM)
        3 => CSF
        4 => Others

    参数:
    -------
    freesurfer_mask : np.ndarray
        形状 (D, H, W) 或 (H, W) 的整型掩膜，每个体素取值 1~25。

    返回:
    -------
    new_mask : np.ndarray
        与 freesurfer_mask 同形状，取值为 {1,2,3,4}，分别表示 4 大类。
    """

    new_mask = np.zeros_like(freesurfer_mask, dtype=np.uint8)

    # 1) 白质 WM: 标签 1~8
    wm_labels = [1,2,3,4,5,6,7,8]
    for val in wm_labels:
        new_mask[freesurfer_mask == val] = 1

    # 2) 灰质 GM: 这里示例把 9~20 都当成灰质
    gm_labels = [9,10,11,12,13,14,15,16,17,18,19,20]
    for val in gm_labels:
        new_mask[freesurfer_mask == val] = 2

    # 3) 脑脊液 CSF: 23,24,25 (侧/三/四脑室)
    csf_labels = [23,24,25]
    for val in csf_labels:
        new_mask[freesurfer_mask == val] = 3

    # 4) 其他 (Others): 这里示例把 21(pons), 22(cerebellum) 放入此类
    #    或者任何其它不在上面列表里的值，也可以统一归到 4
    #    例如:
    others_labels = [21, 22]
    for val in others_labels:
        new_mask[freesurfer_mask == val] = 4

    # 若你想把“所有未指定标签”都默认为 4，可再加一行:
    # new_mask[~np.isin(freesurfer_mask, wm_labels + gm_labels + csf_labels + others_labels)] = 4

    return new_mask


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
        blur_std = random.uniform(0.2, 0.5)
        bias_coefficient = random.uniform(0.1, 0.2)
        bias_degree = random.choice([2, 3, 4])
        snr_means = [40, 50, 20, 30]  # T1WI: GM, WM, CSF, Other 的 SNR 均值
        snr_cov = np.array([
            [15, 10, 5, 8],
            [10, 20, 6, 7],
            [5, 6, 10, 3],
            [8, 7, 3, 12],
        ])
        sampled_snrs = np.random.multivariate_normal(snr_means, snr_cov)
    else:
        blur_std = np.clip(blur_std, a_min=0.2, a_max=0.5)
        bias_coefficient = np.clip(bias_coefficient, a_min=0.05, a_max=0.2)
        bias_degree = int(np.clip(bias_degree, a_min=0.5, a_max=1.0) * 4)
        sampled_snrs = np.clip(sampled_snrs, a_min=0.0625, a_max=1.0) * 80

    target_size = (192, 192, 16)
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
    sampled_snrs = sampled_snrs / 80

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

        blur_std = random.uniform(0.3, 0.7)
        bias_coefficient = random.uniform(0.08, 0.15)
        bias_degree = random.choice([2, 3, 4])
        snr_means = [30, 40, 20, 25]  # GM, WM, CSF, Other 的 SNR 均值
        snr_cov = np.array([
            [12, 8, 4, 6],
            [8, 15, 5, 6],
            [4, 5, 8, 3],
            [6, 6, 3, 10],
        ])
        sampled_snrs = np.random.multivariate_normal(snr_means, snr_cov)
    else:
        blur_std = np.clip(blur_std, a_min=0.3, a_max=0.6)
        bias_coefficient = np.clip(bias_coefficient, a_min=0.05, a_max=0.2)
        bias_degree = int(bias_degree * 4)
        sampled_snrs = np.clip(sampled_snrs, a_min=0.0625, a_max=1.0) * 80

    target_size = (192, 192, 16)
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
    sampled_snrs = sampled_snrs / 80

    params[0] = blur_std
    params[1] = bias_coefficient
    params[2] = bias_degree
    params[3:] = sampled_snrs

    img_data = torch.clip(img_data, min=0, max=1)  # 裁剪到 0~1 范围

    if repeat:
        return tio.ScalarImage(tensor=img_data, affine=img_affine)
    else:
        return tio.ScalarImage(tensor=img_data, affine=img_affine), params

