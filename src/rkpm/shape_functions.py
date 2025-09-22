import numpy as np

def gaussian_kernel(x, xI, h):
    """
    Gaussian kernel in 2D.
    x, xI : np.array([x,y])
    h     : support radius
    """
    r = np.linalg.norm(x - xI) / h
    return np.exp(-r**2)

def gaussian_kernel_grad(x, xI, h):
    """
    Gradient of Gaussian kernel in 2D.
    Returns: np.array([dφ/dx, dφ/dy])
    """
    diff = (x - xI) / h
    return -2 / h * diff * np.exp(-np.dot(diff, diff))

def rkpm_shape_and_grad(x, particles, h, order=1):
    """
    Compute RKPM shape functions and their gradients at point x (2D).
    
    Args:
        x : np.array([x,y]) evaluation point
        particles : list/array of np.array([xi, yi]) particle positions
        h : support radius
        order : 0 (partition of unity) or 1 (linear consistency)
    Returns:
        N : array of shape function values [N1, N2, ...]
        dN : array of gradients [ [dN1/dx, dN1/dy], ..., [dNn/dx, dNn/dy] ]
    """
    # kernel weights and gradients
    W = np.array([gaussian_kernel(x, xi, h) for xi in particles])
    dW = np.array([gaussian_kernel_grad(x, xi, h) for xi in particles])
    if np.sum(W) == 0:
        return np.zeros(len(particles)), np.zeros((len(particles),2))

    if order == 0:
        N = W / np.sum(W)
        # Gradient via quotient rule
        dN = []
        for i in range(len(particles)):
            term1 = dW[i] / np.sum(W)
            term2 = W[i] * np.sum(dW, axis=0) / (np.sum(W)**2)
            dN.append(term1 - term2)
        return N, np.array(dN)

    elif order == 1:
        # Linear consistency → build moment matrix
        A = np.zeros((3, 3))
        for wi, xi in zip(W, particles):
            dx, dy = xi - x
            phi = np.array([1, dx, dy])
            A += wi * np.outer(phi, phi)
        A_inv = np.linalg.inv(A)

        # Compute shape functions + gradients
        N = np.zeros(len(particles))
        dN = np.zeros((len(particles), 2))

        coeff = A_inv @ np.array([1, 0, 0])  # reproduces constant + linear

        for i, (wi, xi, dwi) in enumerate(zip(W, particles, dW)):
            dx, dy = xi - x
            phi = np.array([1, dx, dy])
            N[i] = wi * (phi @ coeff)

            # Gradient: ∇N = ∇w * (phi·coeff) - w * (coeff[1:])   (approx)
            dphi_dx = np.array([0, -1, 0])
            dphi_dy = np.array([0, 0, -1])
            # derivative contributions
            dN[i, :] = dwi * (phi @ coeff) - wi * np.array([coeff[1], coeff[2]])
        return N, dN

    else:
        raise ValueError("Only order=0 or order=1 supported.")

