import os
import io
import numpy as np
import h5py
import torch
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Remove background using coil sensitivity map
def rm_bg(img, csm):
    rcsm = torch.sum(torch.abs(csm), dim=1, keepdim=True)
    cmask = (rcsm > 0).float()
    rec = cmask * img
    return rec