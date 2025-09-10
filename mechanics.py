
import numpy as np

def build_bonds(P, nbrs):
    rows_i = []
    rows_j = []
    R0 = []
    Eij = []
    for i, ids in enumerate(nbrs):
        for j in ids:
            if j > i:
                d = P[j] - P[i]
                r = np.linalg.norm(d)
                if r == 0:
                    continue
                e = d / r
                rows_i.append(i); rows_j.append(j)
                R0.append(r); Eij.append(e)
    return np.array(rows_i, int), np.array(rows_j, int), np.array(R0, float), np.array(Eij, float)

def spring_forces(P, u, bonds, eps_field, k_spring=1.0, k_damp=0.1, v=None, dt=1e-3, crit_stretch=0.01, broken=None, swelling_axis=None):
    if v is None:
        v = np.zeros_like(u)
    i, j, R0, Eij = bonds
    Np = P.shape[0]
    Nb = R0.shape[0]
    if broken is None:
        broken = np.zeros(Nb, dtype=bool)

    Pi = P[i] + u[i]
    Pj = P[j] + u[j]
    dij = Pj - Pi
    rij = np.linalg.norm(dij, axis=1) + 1e-12
    eij = dij / rij[:,None]

    eps_i = eps_field[i]
    eps_j = eps_field[j]
    eps_bond = 0.5*(eps_i + eps_j)
    if swelling_axis is not None:
        ax = np.array(swelling_axis, dtype=float)
        proj = np.abs((eij @ ax))
        eps_bond = eps_bond * proj
    Lrest = R0 * (1.0 + eps_bond)

    stretch = (rij - Lrest) / (R0 + 1e-12)
    newly_broken = (stretch > crit_stretch) & (~broken)
    broken = broken | newly_broken

    intact = ~broken
    fmag = np.zeros(Nb)
    fmag[intact] = k_spring * (rij[intact] - Lrest[intact])
    fij = fmag[:,None] * eij

    F = np.zeros_like(u)
    np.add.at(F, i, -fij)
    np.add.at(F, j, +fij)

    F -= k_damp * v

    E_spring = 0.5*np.sum(k_spring * (rij[intact] - Lrest[intact])**2)
    E_break = np.sum(newly_broken)

    return F, broken, {"E_spring": E_spring, "n_broken": int(np.sum(broken))}

def integrate_quasistatic(P, bonds, eps_fn, steps=2000, dt=5e-4, k_spring=100.0, k_damp=0.5, crit_stretch=0.002,
                          clamp_mask=None, swelling_axis=None, callback=None):
    N = P.shape[0]
    u = np.zeros((N,2))
    v = np.zeros_like(u)
    broken = None

    for step in range(steps):
        eps_field = eps_fn(step)
        F, broken, stats = spring_forces(P, u, bonds, eps_field,
                                         k_spring=k_spring, k_damp=k_damp,
                                         v=v, dt=dt, crit_stretch=crit_stretch,
                                         broken=broken, swelling_axis=swelling_axis)

        if clamp_mask is not None:
            F[clamp_mask] = 0.0
            v[clamp_mask] = 0.0
            u[clamp_mask] = 0.0

        v += dt * F
        u += dt * v

        if callback is not None and (step % max(1, steps//10) == 0 or step==steps-1):
            callback(step, P, u, broken, stats)

    return u, broken
