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

# Remove background using coil sensitivity map
def rm_bg(img, csm):
    rcsm = torch.sum(torch.abs(csm), dim=1, keepdim=True)
    cmask = (rcsm > 0).float()
    rec = cmask * img
    return rec