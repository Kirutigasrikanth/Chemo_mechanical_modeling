import numpy as np
from src.rkpm.shape_functions import gaussian_kernel, gaussian_kernel_grad

def build_laplacian(particles, h):
    """
    Construct a discrete Laplacian matrix using RKPM kernels (1D version).
    """
    N = len(particles)
    L = np.zeros((N, N))
    dx = particles[1] - particles[0]

    for i, xi in enumerate(particles):
        for j, xj in enumerate(particles):
            if i != j:
                w = gaussian_kernel(xi, xj, h)
                dphi = gaussian_kernel_grad(xi, xj, h)
                L[i, j] = w * dphi / dx  # crude approx
        L[i, i] = -np.sum(L[i, :])  # row-sum zero
    return L

def diffusion_rhs(c, D, L):
    """
    Right-hand side of diffusion equation: dc/dt = D * Laplacian * c
    """
    return D * (L @ c)

