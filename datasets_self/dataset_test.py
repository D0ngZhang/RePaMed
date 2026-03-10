import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torchio as tio
import nibabel as nib


mean = np.array([0.5,])
std = np.array([0.5,])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(1):
        tensors[:, c, ...].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


def _make_image_namelist(dir, dir2):
    img_path = []
    namelist = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('HF_T1.nii.gz'):
                item_path = os.path.join(root, fname)
                if os.path.exists(item_path.replace('HF_T1', 'HF_T2')) and os.path.exists(item_path.replace('HF_T1', 'LF_T1')) and os.path.exists(item_path.replace('HF_T1', 'LF_T2')):
                    if not os.path.exists(os.path.join(dir2, fname.replace('HF_T1', 'Syn_T1'))) or not os.path.exists(os.path.join(dir2, fname.replace('HF_T1', 'Syn_T2'))):
                        print("path t1", os.path.join(dir2, fname.replace('HF_T1', 'Syn_T1')), "  path t2", os.path.join(dir2, fname.replace('HF_T1', 'Syn_T2')))
                        namelist.append(fname)
                        img_path.append(item_path)
    return namelist, img_path


def fix_nifti_direction_cosines(file_path, output_path):
    img = nib.load(file_path)
    header = img.header.copy()
    affine = img.affine.copy()

    R = affine[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_fixed = np.dot(U, Vt)
    affine[:3, :3] = R_fixed

    fixed_img = nib.Nifti1Image(img.get_fdata(), affine, header)
    nib.save(fixed_img, output_path)
    return output_path

class data_set(dataset_torch):
    def __init__(self, root, root2):
        self.root = root
        self.root2 = root2
        self.img_names, self.img_paths = _make_image_namelist(self.root, self.root2)

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
        path_img_t1 = self.img_paths[index]

        name = self.img_names[index]

        path_img_t2 = path_img_t1.replace('T1', 'T2')

        img_t1 = self.load_img(path_img_t1)
        img_t2 = self.load_img(path_img_t2)

        img_t1_lf = self.load_img(path_img_t1.replace('HF_T1', 'LF_T1'))
        img_t2_lf = self.load_img(path_img_t2.replace('HF_T2', 'LF_T2'))

        affine = img_t1.affine


        img_t1.data = torch.permute(img_t1.data, (0, 3, 1, 2))
        img_t2.data = torch.permute(img_t2.data, (0, 3, 1, 2))
        img_t1_lf.data = torch.permute(img_t1_lf.data, (0, 3, 1, 2))
        img_t2_lf.data = torch.permute(img_t2_lf.data, (0, 3, 1, 2))

        new_shape = (18, 448, 448)
        resize = tio.Resize(new_shape, image_interpolation='linear')

        img_t1 = resize(img_t1)
        img_t2 = resize(img_t2)
        img_t1_sr = resize(img_t1_lf)
        img_t2_sr = resize(img_t2_lf)

        img_t1_hf_data = img_t1.data
        img_t2_hf_data = img_t2.data
        img_t1_lf_data = img_t1_lf.data
        img_t2_lf_data = img_t2_lf.data

        img_t1_sr_data = img_t1_sr.data
        img_t2_sr_data = img_t2_sr.data

        img_t1_lf_data = (img_t1_lf_data - torch.min(img_t1_lf_data)) / (torch.max(img_t1_lf_data) - torch.min(img_t1_lf_data))
        img_t2_lf_data = (img_t2_lf_data - torch.min(img_t2_lf_data)) / (
                    torch.max(img_t2_lf_data) - torch.min(img_t2_lf_data))
        img_t1_hf_data = (img_t1_hf_data - torch.min(img_t1_hf_data)) / (
                    torch.max(img_t1_hf_data) - torch.min(img_t1_hf_data))
        img_t2_hf_data = (img_t2_hf_data - torch.min(img_t2_hf_data)) / (
                    torch.max(img_t2_hf_data) - torch.min(img_t2_hf_data))

        img_t1_sr_data = (img_t1_sr_data - torch.min(img_t1_sr_data)) / (
                torch.max(img_t1_sr_data) - torch.min(img_t1_sr_data))
        img_t2_sr_data = (img_t2_sr_data - torch.min(img_t2_sr_data)) / (
                torch.max(img_t2_sr_data) - torch.min(img_t2_sr_data))

        img_hf_data = torch.cat((img_t1_hf_data, img_t2_hf_data), dim=0)
        img_lf_data = torch.cat((img_t1_lf_data, img_t2_lf_data), dim=0)
        img_sr_data = torch.cat((img_t1_sr_data, img_t2_sr_data), dim=0)

        imgs = {"hf_image": img_hf_data, "sr_image": img_sr_data, "lf_image": img_lf_data, "affine": affine, "name": name}
        return imgs
