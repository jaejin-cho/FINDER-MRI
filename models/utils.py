# -------------------------------------------------------------------
# utils.py: Utility functions: visualization, I/O, numpy FFT helpers,
# mask generation utilities, evaluation metrics, and b-vector loader.
# -------------------------------------------------------------------

import os
import io
import numpy as np
import h5py
import torch
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

def mosaic(img, num_row, num_col, fig_num, clim, title='',
           use_transpose=False, use_flipud=False):
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    if img.ndim < 3:
        img_res = img
        plt.imshow(img_res)
        plt.gray()
        plt.clim(clim)
    else:
        if img.shape[2] != (num_row * num_col):
            print('sizes do not match')
        else:
            if use_transpose:
                for slc in range(img.shape[2]):
                    img[:, :, slc] = np.transpose(img[:, :, slc])
            if use_flipud:
                img = np.flipud(img)

            img_res = np.zeros((img.shape[0] * num_row, img.shape[1] * num_col))
            idx = 0
            for r in range(num_row):
                for c in range(num_col):
                    img_res[r * img.shape[0]:(r + 1) * img.shape[0],
                            c * img.shape[1]:(c + 1) * img.shape[1]] = img[:, :, idx]
                    idx += 1
        plt.imshow(img_res)
        plt.gray()
        plt.clim(clim)

    plt.suptitle(title, color='white', fontsize=48)

# 미리 만들어 뒀는데 삭제하는 것이 나은가?
# def make_coords_for_slice(nx, ny, nz, z_index):
#     X, Y = np.mgrid[0:nx, 0:ny]
#     x = X.reshape(-1).astype(np.float32)
#     y = Y.reshape(-1).astype(np.float32)
#     z = np.full_like(x, fill_value=z_index, dtype=np.float32)

#     if nx > 1:
#         x /= (nx - 1)
#     if ny > 1:
#         y /= (ny - 1)
#     if nz > 1:
#         z /= (nz - 1)

#     coords = np.stack([x, y, z], axis=1)
#     return coords

# 학습시 사용하는 데이터 분할 함수와 기타 기능들 inference에서는 사용하지 않음.
# def norm(tensor, axes=(0, 1, 2), keepdims=True):
#     for axis in axes:
#         tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)
#     if not keepdims:
#         return tensor.squeeze()
#     return tensor

# def find_center_ind(kspace, axes=(1, 2, 3)):
#     center_locs = norm(kspace, axes=axes).squeeze()
#     return np.argsort(center_locs)[-1:]

# def index_flatten2nd(ind, shape):
#     array = np.zeros(np.prod(shape))
#     array[ind] = 1
#     ind_nd = np.nonzero(np.reshape(array, shape))
#     return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

# def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):
#     nrow, ncol = input_data.shape[0], input_data.shape[1]
#     center_kx = int(find_center_ind(input_data, axes=(1, 2)))
#     center_ky = int(find_center_ind(input_data, axes=(0, 2)))

#     temp_mask = np.copy(input_mask)
#     temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
#               center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

#     pr = np.ndarray.flatten(temp_mask)
#     ind = np.random.choice(
#         np.arange(nrow * ncol),
#         size=int(np.count_nonzero(pr) * rho),
#         replace=False,
#         p=pr / np.sum(pr)
#     )
#     [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

#     loss_mask = np.zeros_like(input_mask)
#     loss_mask[ind_x, ind_y] = 1
#     trn_mask = input_mask - loss_mask
#     return trn_mask, loss_mask

# Remove background using coil sensitivity map
def rm_bg(img, csm):
    rcsm = torch.sum(torch.abs(csm), dim=1, keepdim=True)
    cmask = (rcsm > 0).float()
    rec = cmask * img
    return rec

# # compute timing
# def compute_timing(cfg, dir_mask: str, esp: float):
#     """
#     Derive time_o / time_r from the first org mask and build the DFT matrix.
#     Mutates and returns cfg.

#     Args:
#         cfg:      SimpleNamespace config object (must have cfg.ny)
#         dir_mask: directory containing mask_org_*.npy files
#         esp:      echo spacing [s]

#     Returns:
#         cfg with esp, time_o, time_r, fftmtx fields populated
#     """
#     from scipy.linalg import dft

#     mask_path = os.path.join(
#         dir_mask, f"mask_org_z{0:03d}_q{0:03d}_p{0:03d}.npy")
#     mask_org  = np.load(mask_path, mmap_mode='r')

#     indy_o  = np.max(np.int16(mask_org[0,]), axis=0)
#     indy_o  = indy_o * np.cumsum(indy_o)
#     time_o  = indy_o * esp

#     indy_r  = np.max(np.int16(mask_org[0,]), axis=0)
#     indy_r  = np.flip(indy_r * np.cumsum(indy_r))
#     time_r  = indy_r * esp

#     fftmtx  = dft(cfg.ny, scale='sqrtn')
#     fftmtx  = np.fft.fftshift(fftmtx, axes=(0, 1))

#     cfg.esp    = esp
#     cfg.time_o = time_o
#     cfg.time_r = time_r
#     cfg.fftmtx = fftmtx
#     return cfg

# results
# def calculate_nrmse(recon_img, gt_img):
#     """Calculate NRMSE (0 ~ 100%)"""
#     rmse = np.linalg.norm(np.abs(recon_img) - np.abs(gt_img))
#     nrmse = rmse / np.linalg.norm(np.abs(gt_img)) * 100
#     return nrmse

# def calculate_nrmse_sos(recon_img, gt_img):
#     """Calculate NRMSE after sum-of-squares"""
#     rmse = np.linalg.norm(recon_img - gt_img)
#     nrmse = rmse / np.linalg.norm(gt_img) * 100
#     return nrmse

# def calculate_ssim(img1, img2, data_range=None):
#     """Calculate SSIM between two 2D images"""
#     if data_range is None:
#         data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
#     return ssim(img1, img2, data_range=data_range)