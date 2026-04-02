# ---------------------------------
# models.py: FINDER Building Blocks
# ---------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data_consistency import data_consistency

def fft2c(img, unitary_opt=True):
    # Centered 2-D FFT with optional unitary normalisation.
    norm_mode = "ortho" if unitary_opt else "backward"
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.fftshift(img, dim=(-2, -1)),
            norm=norm_mode),
        dim=(-2, -1))


def ifft2c(k, unitary_opt=True):
    # Centered 2-D IFFT with optional unitary normalisation.
    norm_mode = "ortho" if unitary_opt else "backward"
    return torch.fft.ifftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(k, dim=(-2, -1)),
            norm=norm_mode),
        dim=(-2, -1))

# activation func, ReLU
def activation_func(activation, is_inplace=False):
    return nn.ModuleDict([
        ['ReLU', nn.ReLU(inplace=is_inplace)],
        ['None', nn.Identity()]
    ])[activation]


def BatchNorm(is_batch_norm, features):
    return nn.BatchNorm2d(features) if is_batch_norm else nn.Identity()

# conv
def conv_layer(filter_size, padding=1, is_batch_norm=False,
               activation_type='ReLU'):
    kernel_size, in_c, out_c = filter_size
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c,
                  kernel_size=kernel_size, padding=padding, bias=False),
        BatchNorm(is_batch_norm, in_c),
        activation_func(activation_type)
    )

# ResNet
def ResNetBlock(filter_size):
    return nn.Sequential(
        conv_layer(filter_size, activation_type='ReLU'),
        conv_layer(filter_size, activation_type='None')
    )


class ResNetBlocksModule(nn.Module):
    def __init__(self, device, filter_size, num_blocks):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList(
            [ResNetBlock(filter_size=filter_size) for _ in range(num_blocks)])

    def forward(self, x):
        scale_factor = torch.tensor([0.1], dtype=torch.float32).to(self.device)
        for layer in self.layers:
            x = x + layer(x) * scale_factor
        return x

# ResBlock, Residual Block with InstanceNorm for CNN Encoder
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act(out + residual)
        return out

# CNN Encoder
class UniversalCNNEncoder(nn.Module):
    def __init__(self, in_ch=2, base_ch=32, out_ch_list=None):
        super().__init__()
        if out_ch_list is None:
            out_ch_list = [16, 32, 32]

        self.stages      = nn.ModuleList()
        self.projections = nn.ModuleList()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_ch = base_ch
        for target_out_ch in out_ch_list:
            next_ch = curr_ch * 2
            self.stages.append(ResBlock(curr_ch, next_ch, stride=2))
            self.projections.append(nn.Conv2d(next_ch, target_out_ch, kernel_size=1))
            curr_ch = next_ch

    def forward(self, x):
        features = []
        out = self.stem(x)
        for stage, proj in zip(self.stages, self.projections):
            out        = stage(out)
            compressed = proj(out)
            features.append(compressed)
        return features

class FourierEmbedder(nn.Module):
    def __init__(self, input_dim, mapping_size=64, scale=10.0):
        super().__init__()
        self.input_dim    = input_dim
        self.mapping_size = mapping_size
        self.scale        = scale
        self.B     = nn.Parameter(
            torch.randn(input_dim, mapping_size) * scale, requires_grad=False)
        self.out_dim = mapping_size * 2

    def forward(self, x):
        x_proj = (2. * torch.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SpatioTemporalDiffEmbedder(nn.Module):
    def __init__(self,
                 spatial_dim=3,
                 spatial_map_size=64,
                 spatial_scale=10.0,
                 use_diff_emb=True,
                 diff_vec_dim=3,
                 diff_map_size=16,
                 diff_scale=1.0):
        super().__init__()
        self.use_diff_emb = use_diff_emb
        self.spatial_emb = FourierEmbedder(
            input_dim=spatial_dim,
            mapping_size=spatial_map_size,
            scale=spatial_scale)
        # diffusion Fourier Embedder
        if self.use_diff_emb:
            self.diff_emb = FourierEmbedder(
                input_dim=diff_vec_dim,
                mapping_size=diff_map_size,
                scale=diff_scale)
            self.out_dim = self.spatial_emb.out_dim + self.diff_emb.out_dim
        else:
            self.out_dim = self.spatial_emb.out_dim

    def forward(self, coords, diff_vec=None):
        """
        Args:
            coords:   (B, N_pixels, 3)
            diff_vec: (B, 3)
        Returns:
            embedding: (B, N_pixels, Total_Dim)
        """
        batch_size, num_pixels, _ = coords.shape
        s_feat = self.spatial_emb(coords)

        if self.use_diff_emb:
            d_feat_vec = self.diff_emb(diff_vec)
            d_feat     = d_feat_vec.unsqueeze(1).expand(-1, num_pixels, -1)
            return torch.cat([s_feat, d_feat], dim=-1)
        else:
            return s_feat

class INR_Generator(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=1,
                 depth=5, output_scale=200.0):
        super().__init__()
        self.output_scale = output_scale

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        self.backbone   = nn.Sequential(*layers)
        self.field_head = nn.Linear(hidden_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.field_head.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.field_head.bias, 0.0)

    def forward(self, x):
        feat  = self.backbone(x)
        field = self.field_head(feat)
        return field * self.output_scale



def SSDU_kspace_transform(nw_output, sens_maps, mask, field_maps, config):
    """
    Image-domain network output -> K-space via forward operator.

    Args:
        nw_output:  (B, 4, nx, ny)  real/imag split  [AP_re, AP_im, PA_re, PA_im, ...]
        sens_maps:  (B, nc, nx, ny)
        mask:       (B, nx, ny, n_views)
        field_maps: (B, nx, ny)
        config:     holds time_o, time_r, fftmtx

    Returns:
        kspace_batch: (B, nc, nx, ny, n_views)
    """
    all_recons = []
    for ii in range(nw_output.shape[0]):
        Encoder          = data_consistency(sens_maps[ii], mask[ii], field_maps[ii], config)
        img_complex      = torch.complex(nw_output[ii][0:2], nw_output[ii][2:4])
        nw_output_kspace = Encoder.E_Op(img_complex)
        all_recons.append(nw_output_kspace.unsqueeze(0))
    return torch.cat(all_recons, dim=0)

def Atb_transform(kdata, csm, mask, field, config):
    """
    Adjoint operator A^H * b.

    Args:
        kdata: (B, nc, nx, ny, n_views)
        csm:   (B, nc, nx, ny)
        mask:  (B, nx, ny, n_views)
        field: (B, nx, ny)
        config: holds time_o, time_r, fftmtx

    Returns:
        img_out: (B, n_views, nx, ny)
    """
    all_recons = []
    for ii in range(kdata.shape[0]):
        Encoder   = data_consistency(csm[ii], mask[ii], field[ii], config)
        recon_img = Encoder.Eh_Op(kdata[ii])
        all_recons.append(recon_img.unsqueeze(0))
    return torch.cat(all_recons, dim=0)
