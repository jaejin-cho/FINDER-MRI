# -----------------------------------------------------
# config.py: FINDER Hyperparameter & Scan Configuration
# -----------------------------------------------------

import numpy as np
from types import SimpleNamespace


def get_config(
    # Scan Geometry: MRI acquisition dimensions
    nx: int = 220,
    ny: int = 220,
    nz: int = 1,
    nc: int = 32,
    nd: int = 4,

    bvec: np.ndarray = None,

    esp: float = 9.4e-4,

    # INR Field Estimator: coordinate-based MLP architecture
    coord_dim = 3,
    out_dim = 24,

    hidden_dim = 512,
    depth = 5,

    # Unrolled Reconstruction Network
    CG_Iter = 10,
    init_features = 46,
    nb_res_blocks = 16,
    nb_unroll_blocks = 8,

    # K-space is partitioned into training, loss, and validation masks
    num_reps = 10,
    rho_val = 0.2,
    rho_train = 0.4,

    # training schedule and hyperparameters
    batch_size = 1,
    lr = 5e-4,
    num_epochs = 1,
    stop_training = 50,
) -> SimpleNamespace:

    cfg = SimpleNamespace(
        nx=nx, ny=ny, nz=nz, nc=nc, nd=nd,
        bvec=bvec,
        coord_dim=coord_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        CG_Iter=CG_Iter,
        init_features=init_features,
        nb_res_blocks=nb_res_blocks,
        nb_unroll_blocks=nb_unroll_blocks,
        num_reps=num_reps,
        rho_val=rho_val,
        rho_train=rho_train,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        stop_training=stop_training,
        esp=esp,
        time_o=None,
        time_r=None,
        fftmtx=None,
    )

    return cfg