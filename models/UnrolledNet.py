import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    fft2c, ifft2c,
    SpatioTemporalDiffEmbedder,
    INR_Generator,
    UniversalCNNEncoder,
)
from .ResNet import ResNet
from .data_consistency import dc_block
from .modules import SSDU_kspace_transform, Atb_transform

# loss는 남겨두는 것이 나은가 inference에서는 사용안하니까
class LCCLoss(nn.Module):
    """Local Cross Correlation Loss for (B, X, Y) inputs."""
    def __init__(self, win_size=9, eps=1e-5):
        super().__init__()
        self.win_size = win_size
        self.eps      = eps

    def forward(self, img_ap, img_pa):
        if img_ap.dim() == 3:
            img_ap = img_ap.unsqueeze(1)
        if img_pa.dim() == 3:
            img_pa = img_pa.unsqueeze(1)

        channels = img_ap.shape[1]
        weight   = torch.ones(
            (channels, 1, self.win_size, self.win_size),
            device=img_ap.device, dtype=img_ap.dtype)
        weight  /= self.win_size * self.win_size
        padding  = self.win_size // 2

        mu_ap    = F.conv2d(img_ap, weight, padding=padding, groups=channels)
        mu_pa    = F.conv2d(img_pa, weight, padding=padding, groups=channels)
        mu_ap_sq = F.conv2d(img_ap * img_ap, weight, padding=padding, groups=channels)
        mu_pa_sq = F.conv2d(img_pa * img_pa, weight, padding=padding, groups=channels)
        mu_ap_pa = F.conv2d(img_ap * img_pa, weight, padding=padding, groups=channels)

        var_ap    = mu_ap_sq - mu_ap ** 2
        var_pa    = mu_pa_sq - mu_pa ** 2
        cov_ap_pa = mu_ap_pa - mu_ap * mu_pa

        cc = cov_ap_pa / (torch.sqrt(var_ap * var_pa) + self.eps)
        return 1 - torch.mean(cc)


class CustomLoss(nn.Module):
    def __init__(self, eps=1e-6, scalar=1 / 2):
        super().__init__()
        self.eps    = eps
        self.scalar = scalar
        self.lcc_criterion1 = LCCLoss(win_size=31)
        self.lcc_criterion2 = LCCLoss(win_size=9)

    def get_smoothness_loss(self, field_map):
        if field_map is None:
            return 0.0
        dx = torch.abs(field_map[..., :, 1:] - field_map[..., :, :-1])
        dy = torch.abs(field_map[..., 1:, :] - field_map[..., :-1, :])
        return torch.mean(dx) + torch.mean(dy)

    def get_gradient_l1_loss(self, img1, img2):
        dx1 = torch.abs(img1[..., :, 1:] - img1[..., :, :-1])
        dx2 = torch.abs(img2[..., :, 1:] - img2[..., :, :-1])
        dy1 = torch.abs(img1[..., 1:, :] - img1[..., :-1, :])
        dy2 = torch.abs(img2[..., 1:, :] - img2[..., :-1, :])
        return torch.mean(torch.abs(dx1 - dx2)) + torch.mean(torch.abs(dy1 - dy2))

    def forward(self, yhat, y, im_out, field_delta=None,
                lw=(50, 100, 500, 25, 1)):
        """
        lw order: [k_weight, grad_weight, l2_weight, lcc_weight, smooth_weight]
        """
        # K-space consistency
        term1  = (torch.norm(torch.abs(yhat - y)) + self.eps) / \
                 (torch.norm(torch.abs(y))         + self.eps)
        term2  = (torch.norm(torch.abs(yhat - y), p=1) + self.eps) / \
                 (torch.norm(torch.abs(y), p=1)         + self.eps)
        loss_k = self.scalar * (term1 + term2)

        img_mag = torch.abs(torch.complex(im_out[:, 0:2], im_out[:, 2:4]))
        ap_img  = img_mag[:, 0]
        pa_img  = img_mag[:, 1]

        loss_grad   = self.get_gradient_l1_loss(ap_img, pa_img)
        loss_l2     = F.mse_loss(ap_img, pa_img)
        loss_cc     = (0.5 * self.lcc_criterion1(ap_img, pa_img) +
                       0.5 * self.lcc_criterion2(ap_img, pa_img))
        loss_smooth = self.get_smoothness_loss(field_delta)

        total_loss = (loss_k      * lw[0] +
                      loss_grad   * lw[1] +
                      loss_l2     * lw[2] +
                      loss_cc     * lw[3] +
                      loss_smooth * lw[4])

        print(f"K: {loss_k:.4f} | Grad: {loss_grad:.4f} | "
              f"L2: {loss_l2:.4f} | CC: {loss_cc:.4f} | Sm: {loss_smooth:.4f}")
        return total_loss


# UnrolledNet
_LAMBDA_START = 0.05
class UnrolledNet(nn.Module):
    def __init__(self, config, device=torch.device('cuda:0')):
        super().__init__()
        self.config = config
        self.device = device

        # Unrolled regularisers
        self.regularizer_i = ResNet(
            self.device, in_ch=8,
            num_of_resblocks=config.nb_res_blocks, out_ch=8)
        self.regularizer_k = ResNet(
            self.device, in_ch=8,
            num_of_resblocks=config.nb_res_blocks, out_ch=8)
        self.lam1 = nn.Parameter(torch.tensor([_LAMBDA_START]))
        self.lam2 = nn.Parameter(torch.tensor([_LAMBDA_START]))

        # Embedding
        self.embedding_net = SpatioTemporalDiffEmbedder(
            spatial_dim=config.coord_dim,
            spatial_map_size=64,
            spatial_scale=10,
            use_diff_emb=True,
            diff_vec_dim=3,
            diff_map_size=16,
            diff_scale=4,
        )
        self.emb_dim = self.embedding_net.out_dim

        # Image encoder
        img_out_list       = [32, 32, 64]
        self.img_encoder   = UniversalCNNEncoder(
            in_ch=2, base_ch=32, out_ch_list=img_out_list).to(device)
        self.img_feat_dim  = sum(img_out_list)

        # Field encoder
        field_out_list     = [16, 16, 32]
        self.field_encoder = UniversalCNNEncoder(
            in_ch=1, base_ch=16, out_ch_list=field_out_list).to(device)
        self.field_feat_dim = sum(field_out_list)

        # INR field generator
        total_in_features = self.emb_dim + self.img_feat_dim + self.field_feat_dim
        self.field_net = INR_Generator(
            in_dim=total_in_features,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            depth=config.depth,
            output_scale=100,
        )

    # Freeze / unfreeze helpers
    def _set_grad(self, modules, requires_grad=True):
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            if isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = requires_grad
            elif isinstance(module, nn.Parameter):
                module.requires_grad = requires_grad

    def set_train_phase(self, phase):
        recon_group = [self.regularizer_i, self.regularizer_k,
                       self.lam1, self.lam2]
        field_group = [self.embedding_net, self.img_encoder,
                       self.field_encoder, self.field_net]

        if phase == 'field_only':
            self._set_grad(field_group, True)
            self._set_grad(recon_group, False)
            print("Set Phase: [Field Update] (Recon Network Frozen)")
        elif phase == 'recon_only':
            self._set_grad(field_group, False)
            self._set_grad(recon_group, True)
            print("Set Phase: [Recon Update] (Field Network Frozen)")
        elif phase == 'joint':
            self._set_grad(field_group, True)
            self._set_grad(recon_group, True)
            print("Set Phase: [Joint Training] (All Unfrozen)")
        else:
            raise ValueError("phase must be 'field_only', 'recon_only', or 'joint'")

    # Forward
    def forward(self, kdata, trn_mask, loss_mask, sens_maps,
                field_init, x_init, coords, bvecs):
        B, nc, nx, ny, _ = kdata.shape

        # Embeddings
        c_emb  = self.embedding_net(
            coords.view(B, -1, 3), diff_vec=bvecs.view(B, 3))
        i_feats = self.img_encoder(torch.abs(x_init))
        f_feats = self.field_encoder(field_init.unsqueeze(1))

        # Grid sampling
        coords_xy = coords.view(B, -1, 3)[..., :2]
        grid      = coords_xy.view(B, 1, -1, 2)

        sampled_features = []
        for f in i_feats:
            s = F.grid_sample(f, grid, align_corners=True, mode='bilinear')
            sampled_features.append(s.squeeze(2).permute(0, 2, 1))
        for f in f_feats:
            s = F.grid_sample(f, grid, align_corners=True, mode='bilinear')
            sampled_features.append(s.squeeze(2).permute(0, 2, 1))

        # INR field prediction
        inr_input   = torch.cat([c_emb] + sampled_features, dim=-1)
        field_delta = self.field_net(inr_input)
        field_inr   = field_delta.squeeze(-1).view(B, nx, ny)

        # Adjoint initialisation
        atb   = Atb_transform(kdata, sens_maps, trn_mask, field_inr, self.config)
        x_atb = torch.cat([atb.real, atb.imag], dim=1)
        x     = x_atb

        # Unrolled iterations
        for _ in range(self.config.nb_unroll_blocks):

            # Image domain
            x_vc       = torch.cat([x[:, 0:2], -x[:, 2:4]], dim=1)
            reg_in     = torch.cat([x, x_vc], dim=1)
            i_reg_4ch  = self.regularizer_i(reg_in.float())
            res_orig   = i_reg_4ch[:, 0:4]
            res_vc     = i_reg_4ch[:, 4:8]
            res_vc_r   = torch.cat([res_vc[:, 0:2], -res_vc[:, 2:4]], dim=1)
            i_reg      = x + (res_orig + res_vc_r) / 2

            # K-space domain
            x_complex     = torch.complex(x[:, 0:2], x[:, 2:4])
            k             = fft2c(x_complex)
            k_ri          = torch.cat([k.real, k.imag], dim=1)
            k_conj        = torch.cat([k_ri[:, 0:2], -k_ri[:, 2:4]], dim=1)
            k_vc          = torch.flip(k_conj, dims=[2, 3])
            reg_k_in      = torch.cat([k_ri, k_vc], dim=1)
            k_reg_4ch     = self.regularizer_k(reg_k_in.float())
            res_k_orig    = k_reg_4ch[:, 0:4]
            res_k_vc      = k_reg_4ch[:, 4:8]
            res_k_vc_flip = torch.flip(res_k_vc, dims=[2, 3])
            res_k_vc_r    = torch.cat([res_k_vc_flip[:, 0:2],
                                       -res_k_vc_flip[:, 2:4]], dim=1)
            k_reg         = k_ri + (res_k_orig + res_k_vc_r) / 2
            re_k          = ifft2c(torch.complex(k_reg[:, 0:2], k_reg[:, 2:4]))
            re_k          = torch.cat([re_k.real, re_k.imag], dim=1)

            # Data consistency
            rhs = x_atb + self.lam1 * i_reg + self.lam2 * re_k
            x   = dc_block(rhs, sens_maps, trn_mask,
                            self.lam1 + self.lam2, self.config, field_inr)

        nw_kspace_output = SSDU_kspace_transform(
            x, sens_maps, loss_mask, field_inr, self.config)

        return x, nw_kspace_output, field_inr
