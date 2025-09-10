
import numpy as np

def grid_particles(nx=30, ny=20, Lx=1.0, Ly=0.6, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    if jitter > 0:
        P += rng.normal(scale=jitter*min(Lx/(nx-1), Ly/(ny-1)), size=P.shape)
    return P

def neighbor_list(P, horizon):
    N = P.shape[0]
    nbrs = [[] for _ in range(N)]
    for i in range(N):
        d = P - P[i]
        dist = np.hypot(d[:,0], d[:,1])
        ids = np.where((dist>0) & (dist<=horizon))[0]
        nbrs[i] = ids.tolist()
    return nbrs
