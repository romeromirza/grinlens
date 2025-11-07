import numpy as np
from scipy.interpolate import interp1d
from stl import mesh 

def predict_m_at_n(n_known, nm, n_new, *, kind='linear', extrapolate=True):
    """
    Linear interpolation 

    Parameters 
    -------------
    n_known: array_like shape (N,)
    nm:      shape (N, M)  -> rows indexed by n, columns are M features
    n_new:   scalar or array of new n values

    returns: shape (M,) if n_new is scalar, else (len(n_new), M)
    """
    if nm.shape[0] != n_known.shape[0]:
        raise ValueError("nm.shape[0] must equal n_known.shape[0]")

    # sort by n to satisfy interp1d
    order = np.argsort(n_known)
    n_sorted = n_known[order]
    nm_sorted = nm[order]

    fill_value = 'extrapolate' if extrapolate else (nm_sorted[0], nm_sorted[-1])
    f = interp1d(
        n_sorted, nm_sorted,
        kind=kind, axis=0,
        bounds_error=False, fill_value=fill_value
    )
    return f(n_new)

def get_lun_eps(r, R=1.0):
    """
    Luneburg lens dielectric distribution. 
    2 at the lens center -> 1 at the surface.
    
    Parameters
    -------------
    r: (float)
        Radius from center of spherical lens (in mm)
    R: (float)
        Radius of lens (in mm)

    Returns
    -------------
    eps: (float)
        Dielectric constant (relative permitivity)
    """
    return 2.0 - (r/R)**2.0

def get_vf(eps_emt, eps_mat):
    """
    Compute volume fraction (density) to obtain given dielectric constant.
    Used Maxwell-Garnett theory approximation for composite materials.

    Parameters
    -------------
    eps_mat: (float)
        Dielectric constant of the printed material.
    eps_emt: (float)
        Desired dielectric constant.
    """

    # return (eps_mat + 2.0) / ((3*(eps_mat - 1))/(eps_emt - 1) + eps_mat - 1) # PREV
    vf = ((eps_mat+2.0)*(eps_emt-1.0))/((eps_mat-1.0)*(eps_emt+2.0))
    vf[vf<0] = 0
    vf[vf>1] = 1.0
    return vf

def sphere_mesh(radius=10.0, n_lat=64, n_lon=64):
    phi = np.linspace(0, np.pi, n_lat)
    theta = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)

    # build triangles on the grid
    tris = []
    for i in range(n_lat-1):
        for j in range(n_lon):
            jp = (j+1) % n_lon
            v00 = np.array([x[i, j],   y[i, j],   z[i, j]])
            v01 = np.array([x[i, jp],  y[i, jp],  z[i, jp]])
            v10 = np.array([x[i+1,j],  y[i+1,j],  z[i+1,j]])
            v11 = np.array([x[i+1,jp], y[i+1,jp], z[i+1,jp]])
            tris.append([v00, v10, v01])
            tris.append([v01, v10, v11])
    data = np.zeros(len(tris), dtype=mesh.Mesh.dtype)
    m = mesh.Mesh(data, remove_empty_areas=False)
    for k, t in enumerate(tris):
        m.vectors[k] = t
    
    return m