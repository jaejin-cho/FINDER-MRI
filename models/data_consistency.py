# ------------------------------------------------------
# dc_block.py: Physics-Based Data Consistency (DC) Layer
# ------------------------------------------------------

import torch

class data_consistency:
    def __init__(self, csm, mask, field_map, config):
        self.csm    = csm
        self.mask   = mask
        self.config = config
        self.field  = field_map

        self.device        = csm.device
        self.dtype_complex = csm.dtype
        self.dtype_float   = self.field.dtype

        self.time_o_tensor = torch.from_numpy(config.time_o).to(
            device=self.device, dtype=self.dtype_float)
        self.time_r_tensor = torch.from_numpy(config.time_r).to(
            device=self.device, dtype=self.dtype_float)
        self.fftmtx_tensor = torch.from_numpy(config.fftmtx).to(
            device=self.device, dtype=self.dtype_complex)
        self.readout_times = [self.time_o_tensor, self.time_r_tensor]

    # Normal Operator (E^H * E + mu * I) * x
    def EhE_Op(self, img, mu):
        masked_kspace     = self.E_Op(img)
        image_space_comb  = self.Eh_Op(masked_kspace)
        return image_space_comb + mu * img

    # Forward Operator:  Image (n_views, nx, ny) -> K-space (nc, nx, ny, n_views)
    def E_Op(self, img):
        n_views, nx, ny = img.shape
        nc = self.csm.shape[0]

        hybrid_data = torch.zeros(
            (nc, nx, ny, n_views), device=self.device, dtype=self.dtype_complex)

        for ix in range(nx):
            b0_row  = self.field[ix, :].unsqueeze(0)
            csm_row = self.csm[:, ix, :]

            for v in range(n_views):
                t_vec     = self.readout_times[v].view(ny, 1)
                fm_phs    = 2 * torch.pi * torch.matmul(t_vec, b0_row)
                phase_map = torch.exp(1j * fm_phs)
                E_matrix  = self.fftmtx_tensor * phase_map

                coil_img_row = img[v, ix, :].unsqueeze(0) * csm_row
                hybrid_row   = torch.matmul(E_matrix, coil_img_row.permute(1, 0))
                hybrid_data[:, ix, :, v] = hybrid_row.permute(1, 0)

        # FFT along readout (x -> kx)
        kdata = torch.fft.fftshift(
            torch.fft.fft(
                torch.fft.ifftshift(hybrid_data, dim=1),
                dim=1, norm='ortho'),
            dim=1)

        return kdata * self.mask
    
    # Adjoint Operator:  K-space (nc, nx, ny, n_views) -> Image (n_views, nx, ny)
    def Eh_Op(self, kdata):
        nc, nx, ny, n_views = kdata.shape

        # IFFT along readout (kx -> x)
        himg = torch.fft.fftshift(
            torch.fft.ifft(
                torch.fft.ifftshift(kdata * self.mask, dim=1),
                dim=1, norm='ortho'),
            dim=1)

        img_out = torch.zeros(
            (n_views, nx, ny), device=self.device, dtype=self.dtype_complex)

        for ix in range(nx):
            b0_row  = self.field[ix, :].unsqueeze(0)
            csm_row = self.csm[:, ix, :]

            for v in range(n_views):
                t_vec      = self.readout_times[v].view(ny, 1)
                fm_phs     = 2 * torch.pi * torch.matmul(t_vec, b0_row)
                phase_map  = torch.exp(1j * fm_phs)
                E_matrix   = self.fftmtx_tensor * phase_map
                E_adjoint  = torch.conj(E_matrix).transpose(-1, -2)

                signal_row         = himg[:, ix, :, v]
                signal_row_permuted = signal_row.permute(1, 0)
                coil_imgs          = torch.matmul(E_adjoint, signal_row_permuted)

                csm_permuted    = csm_row.permute(1, 0)
                combined_pixel  = torch.sum(coil_imgs * torch.conj(csm_permuted), dim=1)
                img_out[v, ix, :] = combined_pixel

        return img_out


def zdot_reduce_sum(input_x, input_y):
    dims = tuple(range(len(input_x.shape)))
    return (torch.conj(input_x) * input_y).sum(dims).real


def conjgrad(rhs, sens_maps, mask, mu, field, config):
    Encoder = data_consistency(sens_maps, mask, field, config)
    rhs     = torch.complex(rhs[0:2, ...], rhs[2:4, ...])
    mu      = mu.type(torch.complex64)

    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rsnot   = zdot_reduce_sum(r, r)
    rsold, rsnew = rsnot, rsnot

    for _ in range(config.CG_Iter):
        Ap    = Encoder.EhE_Op(p, mu)
        pAp   = zdot_reduce_sum(p, Ap)
        alpha = rsold / pAp
        x     = x + alpha * p
        r     = r - alpha * Ap
        rsnew = zdot_reduce_sum(r, r)
        beta  = rsnew / rsold
        rsold = rsnew
        p     = beta * p + r

    return torch.cat([x.real, x.imag], dim=0)


def dc_block(rhs, sens_maps, mask, mu, config, field):
    cg_recons = []
    for ii in range(rhs.shape[0]):
        cg_recon = conjgrad(rhs[ii], sens_maps[ii], mask[ii], mu, field[ii], config)
        cg_recons.append(cg_recon.unsqueeze(0))
    return torch.cat(cg_recons, 0)
