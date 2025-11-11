
# streamlit_grin_app.py
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sys
import trimesh


sys.path.append("./")
# Colormaps
try:
    import cmasher as cmr
    CMAP_EPS = cmr.amethyst
    CMAP_T   = cmr.ember
    CMAP_LAT = cmr.neutral
except Exception:
    CMAP_EPS = 'viridis'
    CMAP_T   = 'inferno'
    CMAP_LAT = 'Greys'

# --- Import project modules ---
try:
    import grin
    import grin_utils as gutils
except Exception as e:
    st.error("Could not import grin.py / grin_utils.py. Place them next to this app.")
    st.stop()

# --- Ensure resource CSVs exist where grin.py expects ---
def ensure_resources():
    here = "./"
    os.makedirs(os.path.join(here, "beams"), exist_ok=True)
    os.makedirs(os.path.join(here, "d2t"), exist_ok=True)

    fallback_roots = [here, "/mnt/data"]
    files = {
        ("beams", "diamond_beams.csv"),
        ("beams", "octet_beams.csv"),
        ("beams", "fluorite_fverts.csv"),
        ("beams", "fluorite_caverts.csv"),
        ("d2t", "diamond_d2t.csv"),
        ("d2t", "gyroid_d2t.csv"),
        ("d2t", "octet_d2t.csv"),
        ("d2t", "fluorite_d2t.csv"),
    }
    for sub, fname in files:
        tgt = os.path.join(here, sub, fname)
        if os.path.exists(tgt):
            continue
        for root in fallback_roots:
            cand = os.path.join(root, fname)
            if os.path.exists(cand):
                try:
                    import shutil
                    shutil.copyfile(cand, tgt)
                except Exception:
                    pass
                break

ensure_resources()

st.set_page_config(page_title="GRIN Lens Lattice Designer", layout="wide")
st.title("GRIN Lens Lattice Designer")
st.caption("Build Luneburg spheres and cylinders, use a custom permittivity function, or import a permittivity point cloud. Scroll down to export thickness grid and STL mesh.")
st.caption("Scroll down to export thickness grid and STL mesh.")


# --- Add logo on top right ---
logo_path = "./inkbitlogo.png"  # replace with your actual file path or URL
# Streamlit columns for layout
col1, col2 = st.columns([8, 2])
with col2:
    st.image(logo_path, use_container_width=True)

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Inputs")
    lattice = st.selectbox("Lattice", ["gyroid", "diamond", "octet", "fluorite"], index=0)
    eps_mat = st.number_input("Material εᵣ (2.4 for Inkbit COT)", min_value=1.01, max_value=10.0, value=2.40, step=0.01, format="%.2f")
    cell_size = st.number_input("Unit cell size [mm]", min_value=1.0, max_value=8.0, value=5.0, step=0.1, format="%.1f")
    min_th = st.number_input("Minimum beam thickness [mm]", min_value=0.0, max_value=5.0, value=0.1, step=0.05, format="%.2f")
    edge_mode = st.selectbox("Edge mode (Clip: lens will be truncated at min. thickness).", ["clip", "extend"], index=0)

    st.divider()
    st.subheader("Output grid shape ")
    nx = st.number_input("Nx", min_value=3, max_value=401, value=101, step=2)
    ny = st.number_input("Ny", min_value=3, max_value=401, value=101, step=2)
    nz = st.number_input("Nz", min_value=3, max_value=401, value=101, step=2)
    shape = st.selectbox("Profile", ["luneburg_sphere", "luneburg_cylinder", "custom_function", "custom_grid"], index=0)

    st.subheader("Physical size [mm]")
    X_user = Y_user = Z_user = None
    uploaded = None
    if shape == "luneburg_sphere":
        st.caption("Luneburg sphere: X = Y = Z = 2R")
    elif shape == "luneburg_cylinder":
        st.caption("Luneburg cylinder: X = Y = 2R. Set Z below.")
        Z_user = st.number_input("Z", min_value=0.0, max_value=2000.0, value=60.0, step=1.0, key="Z_only")
    else:  # custom_function or custom_grid
        colx, coly, colz = st.columns(3)
        with colx:
            X_user = st.number_input("X", min_value=0.0, max_value=2000.0, value=60.0, step=1.0)
        with coly:
            Y_user = st.number_input("Y", min_value=0.0, max_value=2000.0, value=60.0, step=1.0)
        with colz:
            Z_user = st.number_input("Z", min_value=0.0, max_value=2000.0, value=60.0, step=1.0)

    st.subheader("Parameters")
    R = st.number_input("Radius R [mm] (for sphere/cylinder)", min_value=0.0, max_value=2000.0, value=60.0, step=1.0)

    custom_expr = None
    if shape == "custom_function":
        st.markdown("Enter ε(x,y,z). Variables are in mm, on a 0..X, 0..Y, 0..Z grid.")
        default_expr = "2 - ( ( ((x-30)**2 + (y-30)**2 + (z-30)**2)**0.5 )/30)**3"
        custom_expr = st.text_area("ε(x,y,z) =", value=default_expr, height=120, help="Use numpy ops: sin, cos, exp, sqrt, etc.")

    if shape == "custom_grid":
        uploaded = st.file_uploader("Upload ε-grid file (4 columns: x y z ε)", type=["txt", "dat", "csv"])
        st.caption("Grid must contain exactly Nx×Ny×Nz rows. Coordinates must form a rectilinear grid.")
        units = st.selectbox("Units", ["mm", "cm", "m"], index=0)

    build = st.button("Make lens", type="primary")

# --- Build lens and persist ---
internal_grid = 64

def _nearest_index(val, arr):
    i = int(np.searchsorted(arr, val))
    if i <= 0: return 0
    if i >= len(arr): return len(arr)-1
    return i if abs(val - arr[i]) < abs(val - arr[i-1]) else i-1

if build:
    try:
        ln = grin.lens(lattice=lattice, eps_mat=float(eps_mat), cell_size=float(cell_size), grid_size=internal_grid,
                        min_thickness=float(min_th), edge_mode=edge_mode)

        if shape == "luneburg_sphere":
            X = Y = Z = float(2.0 * R)
            ln.make(shape=shape, out_shape=(int(nx), int(ny), int(nz)), R=float(R), X=X, Y=Y, Z=Z)

        elif shape == "luneburg_cylinder":
            X = Y = float(2.0 * R)
            Z = float(Z_user if Z_user is not None else 2.0 * R)
            ln.make(shape=shape, out_shape=(int(nx), int(ny), int(nz)), R=float(R), X=X, Y=Y, Z=Z)

        elif shape == "custom_function":
            X = float(X_user); Y = float(Y_user); Z = float(Z_user)
            safe_ns = {
                "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
                "pi": np.pi, "e": np.e, "X": X, "Y": Y, "Z": Z, "max": max, "min": min, "clip": np.clip
            }
            def eps_func(x, y, z):
                return eval(custom_expr, {"__builtins__": {}}, dict(safe_ns, x=x, y=y, z=z))
            ln.make(shape="custom_grid", out_shape=(int(nx), int(ny), int(nz)), X=X, Y=Y, Z=Z, custom_eps_func=eps_func)

        else:  # custom_grid
            if uploaded is None:
                st.error("Upload a 4-column file first."); st.stop()
            
            try:
                data = _np.loadtxt(uploaded)
            except Exception as e:
                st.error(f"Failed to read file: {e}"); st.stop()
            if data.ndim != 2 or data.shape[1] < 4:
                st.error("File must have at least 4 columns: x y z epsilon."); st.stop()
            x, y, z, values = data[:,0], data[:,1], data[:,2], data[:,3]
            if units == 'cm':
                x*=10
                y*=10
                z*=10
            elif units == 'm':
                x*=1000
                y*=1000
                z*=1000
            x_unique = _np.unique(x)
            y_unique = _np.unique(y)
            z_unique = _np.unique(z)
            nx_f, ny_f, nz_f = len(x_unique), len(y_unique), len(z_unique)
            if (nx_f, ny_f, nz_f) != (int(nx), int(ny), int(nz)):
                st.error(f"Grid size mismatch. File grid = ({nx_f},{ny_f},{nz_f}) but Nx,Ny,Nz = ({int(nx)},{int(ny)},{int(nz)})."); st.stop()
            eps_3d = _np.ones((nx_f, ny_f, nz_f), dtype=float)
            for xi, yi, zi, val in zip(x, y, z, values):
                i = _np.where(x_unique == xi)[0][0]
                j = _np.where(y_unique == yi)[0][0]
                k = _np.where(z_unique == zi)[0][0]
                eps_3d[i, j, k] = val
            eps_3d[eps_3d != eps_3d] = 1.0
            # physical size defaults from file extents if user left zeros
            X =  x_unique.max() - x_unique.min()
            Y = y_unique.max() - y_unique.min()
            Z = z_unique.max() - z_unique.min()
            if X <= 0 or Y <= 0 or Z <= 0:
                st.error("Physical dimensions X,Y,Z must be > 0."); st.stop()

            ln.make(shape="custom_grid", out_shape=(int(nx), int(ny), int(nz)), R=float(R), X=X, Y=Y, Z=Z, custom_eps_grid=eps_3d)

        st.session_state["grin_result"] = {
            "eps_grid": ln.eps_grid,
            "t_grid": ln.thickness_grid,
            "lattice_vol": ln.lattice,
            "steps": (ln._lens__stepx, ln._lens__stepy, ln._lens__stepz),
            "params": {"X": float(X), "Y": float(Y), "Z": float(Z), "R": float(R)},
        }
        st.session_state["writer"] = ln
        st.success("Lens built")
    except Exception as e:
        st.error(f"Build failed: {e}")

# --- Visualize if result present ---
if "grin_result" in st.session_state:
    res = st.session_state["grin_result"]
    eps_grid = res["eps_grid"]
    t_grid   = res["t_grid"]
    lattice  = res["lattice_vol"]
    stepx, stepy, stepz = res["steps"]
    X = res["params"]["X"]; Y = res["params"]["Y"]; Z = res["params"]["Z"]; R = res["params"]["R"]

    depth_eps = int(eps_grid.shape[2])
    depth_t   = int(t_grid.shape[2])
    lat_depth = int(lattice.shape[2])
    if depth_eps != depth_t:
        st.error("ε and thickness depths must match"); st.stop()

    total_depth_mm = Z if Z > 0 else max(2*R, 1e-9)
    if not isinstance(stepz, (int, float)) or stepz <= 0:
        stepz = total_depth_mm / max(depth_eps, 1)

    z_idx = st.slider("Slice in z-direction", 0, depth_eps - 1, depth_eps // 2, step=1, key="z_idx_shared")

    z_norm_center = (z_idx + 0.5) / max(depth_eps, 1)
    z_pos_mm_center = z_norm_center * total_depth_mm
    st.caption(f"z = {z_pos_mm_center:.3f} mm")

    z_norm_edge = z_idx / max(depth_eps - 1, 1)
    idx_lat = int(np.clip(round(z_norm_edge * (lat_depth - 1)), 0, lat_depth - 1))

    eps_slice = eps_grid[:, :, z_idx]
    t_slice   = t_grid[:, :, z_idx]
    lat_slice = lattice[:, :, idx_lat]

    eps_min = float(np.nanmin(eps_grid)); eps_max = float(np.nanmax(eps_grid))
    t_min   = float(np.nanmin(t_grid));   t_max   = float(np.nanmax(t_grid))
    lat_min = float(np.nanmin(lattice));  lat_max = float(np.nanmax(lattice))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Permittivity ε**")
        fig = plt.figure()
        im = plt.imshow(eps_slice.T, origin="lower", cmap=CMAP_EPS, aspect="equal", vmin=eps_min, vmax=eps_max)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks([]); plt.yticks([])
        st.pyplot(fig, clear_figure=True, width='stretch')

    with c2:
        st.markdown("**Thickness [mm]**")
        fig = plt.figure()
        im = plt.imshow(t_slice.T, origin="lower", cmap=CMAP_T, aspect="equal", vmin=t_min, vmax=t_max)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks([]); plt.yticks([])
        st.pyplot(fig, clear_figure=True, width='stretch')

    with c3:
        st.markdown("**Lattice**")
        fig = plt.figure()
        im = plt.imshow(lat_slice.T, origin="lower", cmap=CMAP_LAT, aspect="equal", vmin=lat_min, vmax=lat_max)
        plt.xticks([]); plt.yticks([])
        st.pyplot(fig, clear_figure=True, width='stretch')

    st.divider()

    st.subheader("Export thickness grid")
    out_name = st.text_input("File name", value="thickness_grid.txt")
    if st.button("Make thickness file"):
        try:
            ln = st.session_state.get("writer", None)
            if ln is None:
                st.error("No lens object available for writing. Rebuild first.")
            else:
                tmp_path = out_name if os.path.isabs(out_name) else os.path.join(os.getcwd(), out_name)
                ln.write_thickness(tmp_path)
                with open(tmp_path, "rb") as f:
                    data = f.read()
                st.download_button("Download thickness grid", data=data, file_name=os.path.basename(out_name), mime="text/plain")
                st.info("File generated. Use the button above to download.")
        except Exception as e:
            st.error(f"Write failed: {e}")

    st.subheader("Export mesh STL (sphere and cylinder only)")
    mesh_name = st.text_input("Output mesh name", value="mesh")
    if st.button("Make mesh STL"):
        try:
            ln = st.session_state.get("writer", None)
            if ln is None:
                st.error("No lens object available for writing. Rebuild first.")
            else:
                tmp_path = mesh_name if os.path.isabs(mesh_name) else os.path.join(os.getcwd(), mesh_name)
                ln.make_mesh()
                mesh_stl = ln.mesh
                mesh_stl.export(tmp_path+'.stl')
                with open(tmp_path+'.stl', "rb") as f:
                    data = f.read()
                st.download_button("Download mesh", data=data, file_name=os.path.basename(mesh_name)+'.stl')
                st.info("File generated. Use the button above to download.")
        except Exception as e:
            st.error(f"Write failed: {e}")
else:
    st.info("Set parameters on the left, then press **Make lens**.")
