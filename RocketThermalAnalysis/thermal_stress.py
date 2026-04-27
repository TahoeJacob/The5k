"""
thermal_stress.py
Plane-strain thermo-elastic solver on the 2-D wall conduction unit cell.

Formulation
-----------
Generalized plane strain in the axial (z) direction:
  ε_zz is a single global DOF,
  ∫ σ_zz dA = F_z_ext = 0  (free axial expansion of the slice).

In-plane BCs
  x = 0, x = W:  u_x = 0 (symmetry — channel CL and land CL)
  y = 0:         traction = +P_gas·ŷ (hot gas pushes wall outward)
  channel faces: traction = −P_cool·n̂_solid (coolant pressurises void)
  y = H (top):   free surface (no closeout, per user choice)

Discretisation
  Q4 bilinear elements on the existing rectangular (tensor-product) grid.
  2×2 Gauss quadrature.
  Material properties E(T̄), α(T̄) evaluated per element at the average
  of the 4 corner temperatures from the thermal solve.

Outputs
  sigma_vm_field (ny-1, nx-1)   element-centroid von Mises stress [Pa]
  sigma_zz_field (ny-1, nx-1)   element-centroid axial stress     [Pa]
  is_solid_elem  (ny-1, nx-1)   element mask
  eps_zz                        uniform axial strain [-]
  plus scalar summaries: σ_vm_peak, σ_vm_hw, σ_vm_corner, σ_zz_hw
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from materials import Material


# ---------------------------------------------------------------------------
# Q4 shape functions and integration
# ---------------------------------------------------------------------------
_GAUSS_2x2 = [
    (-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), 1.0),
    ( 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), 1.0),
    ( 1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0), 1.0),
    (-1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0), 1.0),
]


def _q4_B(xi, eta, x_c, y_c):
    """
    Strain-displacement matrix B (3×8) for a Q4 element at parametric
    coordinates (xi, eta).  Corner order: (0,0) BL, (1,0) BR, (1,1) TR, (0,1) TL.

    For an axis-aligned rectangle the Jacobian is diagonal and constant,
    but we handle the general case for generality.
    """
    # Shape function derivatives w.r.t. (xi, eta)
    dN_dxi  = 0.25 * np.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi ), -(1 + xi ),  (1 + xi ),  (1 - xi )])

    # Jacobian J (2×2)
    J = np.array([
        [np.dot(dN_dxi,  x_c), np.dot(dN_dxi,  y_c)],
        [np.dot(dN_deta, x_c), np.dot(dN_deta, y_c)],
    ])
    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)

    # dN/dx, dN/dy
    dN_dxy = Jinv @ np.vstack([dN_dxi, dN_deta])  # shape (2, 4)
    dN_dx = dN_dxy[0]
    dN_dy = dN_dxy[1]

    # B matrix (3×8) — engineering strains [ε_xx, ε_yy, γ_xy]
    B = np.zeros((3, 8))
    for a in range(4):
        B[0, 2 * a    ] = dN_dx[a]
        B[1, 2 * a + 1] = dN_dy[a]
        B[2, 2 * a    ] = dN_dy[a]
        B[2, 2 * a + 1] = dN_dx[a]
    return B, detJ


def _D_plane_strain(E, nu):
    """In-plane 3×3 elastic matrix for plane strain, engineering shear."""
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu,        0.0],
        [nu,       1.0 - nu,  0.0],
        [0.0,      0.0,       0.5 * (1.0 - 2.0 * nu)],
    ])


def _element_routines(x_corners, y_corners, T_corners,
                       material: Material, T_ref):
    """
    Compute element contributions for generalized plane strain.

    Returns
    -------
    k_uu  (8,8)  : in-plane stiffness
    k_uz  (8,)   : coupling to ε_zz DOF
    k_zz  scalar : ε_zz self-stiffness
    f_u   (8,)   : in-plane thermal load
    f_z   scalar : ε_zz thermal load
    A     scalar : element area
    """
    # Element-average temperature for property evaluation
    T_avg = float(np.mean(T_corners))
    E = material.E(T_avg)
    nu = material.nu
    alpha = material.alpha(T_avg)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))
    beta = E * alpha / (1.0 - 2.0 * nu)         # = (3λ+2μ)·α

    D_ip = _D_plane_strain(E, nu)

    k_uu = np.zeros((8, 8))
    k_uz = np.zeros(8)
    f_u  = np.zeros(8)
    A = 0.0

    # Loop Gauss points
    for (xi, eta, w) in _GAUSS_2x2:
        B, detJ = _q4_B(xi, eta, x_corners, y_corners)
        dA = detJ * w
        A += dA

        # Interpolate ΔT at this Gauss point from corner values
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ])
        T_gp = float(np.dot(N, T_corners))
        dT = T_gp - T_ref

        # K_uu = ∫ Bᵀ D_ip B dA
        k_uu += (B.T @ D_ip @ B) * dA

        # K_uz = ∫ Bᵀ [λ, λ, 0]ᵀ dA      (column 4 of D_gen, rows 1–3)
        k_uz += B.T @ np.array([lam, lam, 0.0]) * dA

        # f_u  = ∫ Bᵀ [β·ΔT, β·ΔT, 0]ᵀ dA
        f_u += B.T @ np.array([beta * dT, beta * dT, 0.0]) * dA

    # K_zz and f_z use element-constant properties; integrate ΔT analytically
    # For Q4 with 2×2 Gauss on a rectangle, the mean of T_gp over the 4 GPs
    # equals the mean of T_corners.  We reuse T_avg for f_z.
    k_zz = (lam + 2.0 * mu) * A
    f_z  = beta * (T_avg - T_ref) * A

    return k_uu, k_uz, k_zz, f_u, f_z, A


# ---------------------------------------------------------------------------
# Element enumeration & mechanical DOF numbering
# ---------------------------------------------------------------------------
def _enumerate_solid_elements(node_map, nx, ny):
    """
    Return list of (i, j) cells whose 4 corners are all solid nodes.
    Element (i, j) has corners (i,j), (i+1,j), (i+1,j+1), (i,j+1).
    """
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if ((i,     j    ) in node_map and
                (i + 1, j    ) in node_map and
                (i + 1, j + 1) in node_map and
                (i,     j + 1) in node_map):
                elems.append((i, j))
    return elems


def _build_mech_dof_map(elems, node_map, nx):
    """
    Assign mechanical DOFs to nodes that appear in at least one solid
    element, with two global extras:

      • master_ux  — shared u_x on the land centerline (x = x_nodes[-1])
                     so the cell may expand circumferentially as a rigid
                     translation.  Left free, giving ∫σ_xx dy = 0 on that
                     edge (physical: free hoop on the periodic slice).
      • eps_zz     — generalized plane-strain axial DOF.

    Returns
    -------
    mech_map : {(i,j): mech_index}        node→mech index mapping
    dof_of   : {(mech_idx, comp): dof}    global DOF index per (node, component)
    N_dof    : int                        total number of global DOFs
    master_ux_dof : int                   global index of the shared u_x DOF
    eps_zz_dof    : int                   global index of the ε_zz DOF
    """
    used = set()
    for (i, j) in elems:
        used.update([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
    ordered = sorted(used)
    mech_map = {key: i for i, key in enumerate(ordered)}

    dof_of = {}
    next_dof = 0
    master_ux_dof = next_dof; next_dof += 1
    for midx, key in enumerate(ordered):
        i, j = key
        if i == nx - 1:
            dof_of[(midx, 0)] = master_ux_dof
        else:
            dof_of[(midx, 0)] = next_dof; next_dof += 1
        dof_of[(midx, 1)] = next_dof; next_dof += 1
    eps_zz_dof = next_dof; next_dof += 1

    return mech_map, dof_of, next_dof, master_ux_dof, eps_zz_dof


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _assemble_system(elems, mech_map, dof_of, N_dof, zz_dof,
                     node_map, x_nodes, y_nodes,
                     T_field, material, T_ref):
    """
    Assemble K and f for the generalized plane-strain problem with a
    shared master_ux DOF on the land centerline (right edge).
    """
    rows, cols, vals = [], [], []
    f = np.zeros(N_dof)

    for (i, j) in elems:
        corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        x_c = np.array([x_nodes[ci] for (ci, _) in corners])
        y_c = np.array([y_nodes[cj] for (_, cj) in corners])
        T_c = np.array([T_field[node_map[c]] for c in corners])

        k_uu, k_uz, k_zz, f_u, f_z, A = _element_routines(
            x_c, y_c, T_c, material, T_ref)

        # Element global DOF indices (x, y, x, y, ...) — right-edge u_x's
        # automatically collapse to master_ux via dof_of.
        dof = []
        for c in corners:
            m = mech_map[c]
            dof.append(dof_of[(m, 0)])
            dof.append(dof_of[(m, 1)])

        # Scatter k_uu
        for a in range(8):
            for b in range(8):
                rows.append(dof[a]); cols.append(dof[b])
                vals.append(k_uu[a, b])

        # Scatter k_uz and k_zu (symmetric)
        for a in range(8):
            rows.append(dof[a]); cols.append(zz_dof)
            vals.append(k_uz[a])
            rows.append(zz_dof); cols.append(dof[a])
            vals.append(k_uz[a])

        rows.append(zz_dof); cols.append(zz_dof); vals.append(k_zz)

        for a in range(8):
            f[dof[a]] += f_u[a]
        f[zz_dof] += f_z

    K = sp.csr_matrix((vals, (rows, cols)), shape=(N_dof, N_dof))
    return K, f


# ---------------------------------------------------------------------------
# Pressure loads
# ---------------------------------------------------------------------------
def _apply_pressure_loads(f, mech_map, dof_of, node_map,
                          x_nodes, y_nodes, is_void,
                          chan_w_half, chan_t, chan_h, P_gas, P_cool):
    """
    Apply edge pressure tractions to consistent nodal forces.

    Faces loaded:
      • Hot wall y=0           : t = +P_gas · ŷ  (pushes wall radially outward)
      • Channel base y=chan_t  : t = −P_cool · ŷ (x < chan_w_half)
      • Channel side x=w/2     : t = +P_cool · x̂ (chan_t < y < chan_t+chan_h)
      • Channel ceiling        : t = +P_cool · ŷ (x < chan_w_half,
                                                  y = chan_t + chan_h)
    Top (y=H) and land-CL (x=W) are free surfaces (per user spec).

    Linear Q4 with Gauss-2 along an edge gives the standard lumped-consistent
    force: each end node gets P·L/2.
    """
    nx = len(x_nodes)
    ny = len(y_nodes)

    def _add(node_key, fx, fy):
        if node_key not in mech_map:
            return
        m = mech_map[node_key]
        f[dof_of[(m, 0)]] += fx
        f[dof_of[(m, 1)]] += fy

    # --- Hot wall y=0 ---
    for i in range(nx - 1):
        L = x_nodes[i + 1] - x_nodes[i]
        half = 0.5 * P_gas * L
        _add((i,     0), 0.0, +half)
        _add((i + 1, 0), 0.0, +half)

    # --- Channel base y = chan_t, x in [0, chan_w_half] ---
    j_base = int(np.searchsorted(y_nodes, chan_t - 1e-12))
    if j_base < ny and abs(y_nodes[j_base] - chan_t) < 1e-9:
        for i in range(nx - 1):
            if x_nodes[i + 1] > chan_w_half + 1e-12:
                break
            L = x_nodes[i + 1] - x_nodes[i]
            half = 0.5 * P_cool * L
            _add((i,     j_base), 0.0, -half)
            _add((i + 1, j_base), 0.0, -half)

    # --- Channel side x = chan_w_half, y in [chan_t, chan_t+chan_h] ---
    i_side = int(np.searchsorted(x_nodes, chan_w_half - 1e-12))
    if i_side < nx and abs(x_nodes[i_side] - chan_w_half) < 1e-9:
        y_top_void = chan_t + chan_h
        for j in range(ny - 1):
            if y_nodes[j + 1] <= chan_t + 1e-12:
                continue
            if y_nodes[j] >= y_top_void - 1e-12:
                break
            L = y_nodes[j + 1] - y_nodes[j]
            half = 0.5 * P_cool * L
            _add((i_side, j    ), +half, 0.0)
            _add((i_side, j + 1), +half, 0.0)

    # --- Channel ceiling y = chan_t + chan_h, x in [0, chan_w_half] ---
    y_ceil = chan_t + chan_h
    j_ceil = int(np.searchsorted(y_nodes, y_ceil - 1e-12))
    if j_ceil < ny and abs(y_nodes[j_ceil] - y_ceil) < 1e-9:
        for i in range(nx - 1):
            if x_nodes[i + 1] > chan_w_half + 1e-12:
                break
            L = x_nodes[i + 1] - x_nodes[i]
            half = 0.5 * P_cool * L
            _add((i,     j_ceil), 0.0, +half)
            _add((i + 1, j_ceil), 0.0, +half)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
def _apply_symmetry_bcs(K, f, mech_map, dof_of, x_nodes):
    """
    Fix:
      • u_x = 0 on x=0 (channel CL — true symmetry plane)
      • u_y = 0 at one node to kill rigid-body y-translation

    The right-edge u_x is already shared (master_ux) and left FREE, so the
    cell may expand circumferentially and ∫σ_xx dy = 0 is enforced weakly.
    Generalized plane strain fixes rigid-body z via ε_zz.
    """
    nx = len(x_nodes)
    bc_dofs = set()
    for (i, j), m in mech_map.items():
        if i == 0:
            bc_dofs.add(dof_of[(m, 0)])      # u_x on channel CL

    # Pin u_y at channel CL / hot wall (0, 0) if present; otherwise
    # lowest-j node at i=0.
    pin_key = None
    for j in range(0, 10 ** 6):
        if (0, j) in mech_map:
            pin_key = (0, j)
            break
    if pin_key is not None:
        bc_dofs.add(dof_of[(mech_map[pin_key], 1)])

    K = K.tolil()
    for d in bc_dofs:
        K.rows[d] = [d]
        K.data[d] = [1.0]
        f[d] = 0.0
    K = K.tocsc()
    for d in bc_dofs:
        col = K.getcol(d).toarray().ravel()
        col[d] = 0.0
        for r in np.nonzero(col)[0]:
            K[r, d] = 0.0
        K[d, d] = 1.0
    return K.tocsr(), f, bc_dofs


# ---------------------------------------------------------------------------
# Stress recovery
# ---------------------------------------------------------------------------
def _recover_stress(u_g, eps_zz, elems, mech_map, dof_of, node_map,
                    x_nodes, y_nodes, T_field, material: Material, T_ref):
    """
    Compute element-centroid stresses and return:
      sigma_xx, sigma_yy, sigma_xy, sigma_zz, sigma_vm  (all dict keyed by (i,j))
    """
    sx = {}; sy = {}; sxy = {}; sz = {}; svm = {}
    for (i, j) in elems:
        corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        x_c = np.array([x_nodes[ci] for (ci, _) in corners])
        y_c = np.array([y_nodes[cj] for (_, cj) in corners])
        T_c = np.array([T_field[node_map[c]] for c in corners])
        T_avg = float(np.mean(T_c))
        E = material.E(T_avg)
        nu = material.nu
        alpha = material.alpha(T_avg)
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu  = E / (2.0 * (1.0 + nu))
        beta = E * alpha / (1.0 - 2.0 * nu)

        # Nodal displacements for this element (8,)
        u_e = np.zeros(8)
        for a, c in enumerate(corners):
            m = mech_map[c]
            u_e[2 * a    ] = u_g[dof_of[(m, 0)]]
            u_e[2 * a + 1] = u_g[dof_of[(m, 1)]]

        # Strain at centroid (xi=eta=0)
        B, _ = _q4_B(0.0, 0.0, x_c, y_c)
        eps_ip = B @ u_e        # [ε_xx, ε_yy, γ_xy]
        e_xx, e_yy, g_xy = eps_ip

        dT = T_avg - T_ref

        # σ_xx = (λ+2μ)e_xx + λ·e_yy + λ·ε_zz − β·ΔT
        s_xx = (lam + 2.0 * mu) * e_xx + lam * e_yy + lam * eps_zz - beta * dT
        s_yy = lam * e_xx + (lam + 2.0 * mu) * e_yy + lam * eps_zz - beta * dT
        s_xy = mu * g_xy
        s_zz = lam * e_xx + lam * e_yy + (lam + 2.0 * mu) * eps_zz - beta * dT

        # Von Mises (3D)
        vm = np.sqrt(0.5 * ((s_xx - s_yy) ** 2 +
                            (s_yy - s_zz) ** 2 +
                            (s_zz - s_xx) ** 2 +
                            6.0 * s_xy ** 2))

        sx[(i, j)]  = s_xx
        sy[(i, j)]  = s_yy
        sxy[(i, j)] = s_xy
        sz[(i, j)]  = s_zz
        svm[(i, j)] = vm

    return sx, sy, sxy, sz, svm


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def solve_stress_2d(T_field, node_map, x_nodes, y_nodes, is_void,
                    chan_w_half, chan_t, chan_h,
                    P_gas, P_cool, material: Material, T_ref):
    """
    Solve the generalized plane-strain thermo-elastic problem on the
    unit cell.  Returns a dict with fields and scalar summaries.
    """
    nx = len(x_nodes)
    ny = len(y_nodes)

    elems = _enumerate_solid_elements(node_map, nx, ny)
    if not elems:
        return None

    mech_map, dof_of, N_dof, master_ux_dof, zz_dof = _build_mech_dof_map(
        elems, node_map, nx)
    K, f = _assemble_system(elems, mech_map, dof_of, N_dof, zz_dof,
                            node_map, x_nodes, y_nodes,
                            T_field, material, T_ref)

    _apply_pressure_loads(f, mech_map, dof_of, node_map, x_nodes, y_nodes,
                          is_void, chan_w_half, chan_t, chan_h, P_gas, P_cool)

    K, f, _ = _apply_symmetry_bcs(K, f, mech_map, dof_of, x_nodes)

    # Solve
    u_g = spla.spsolve(K.tocsr(), f)
    eps_zz = float(u_g[zz_dof])

    sx, sy, sxy, sz, svm = _recover_stress(
        u_g, eps_zz, elems, mech_map, dof_of, node_map,
        x_nodes, y_nodes, T_field, material, T_ref)

    # --- Build element-centroid 2D fields for the viewer (NaN for void) ---
    sigma_vm_field = np.full((ny - 1, nx - 1), np.nan)
    sigma_zz_field = np.full((ny - 1, nx - 1), np.nan)
    is_solid_elem  = np.zeros((ny - 1, nx - 1), dtype=bool)
    for (i, j), v in svm.items():
        sigma_vm_field[j, i] = v
        sigma_zz_field[j, i] = sz[(i, j)]
        is_solid_elem[j, i] = True

    # --- Scalar summaries ---
    sigma_vm_peak = float(np.nanmax(sigma_vm_field))

    # Hot wall average (j=0 row, across all solid elements)
    hw_vm = [svm[(i, 0)] for i in range(nx - 1) if (i, 0) in svm]
    sigma_vm_hw = float(np.mean(hw_vm)) if hw_vm else 0.0
    hw_zz = [sz[(i, 0)] for i in range(nx - 1) if (i, 0) in sz]
    sigma_zz_hw = float(np.mean(hw_zz)) if hw_zz else 0.0

    # Channel-base / land-root corner: i just outside chan_w_half, j = j_base
    j_base = int(np.searchsorted(y_nodes, chan_t - 1e-12))
    i_side = int(np.searchsorted(x_nodes, chan_w_half - 1e-12))
    corner_key = (i_side, j_base) if (i_side, j_base) in svm else None
    if corner_key is None and (i_side - 1, j_base) in svm:
        corner_key = (i_side - 1, j_base)
    sigma_vm_corner = float(svm[corner_key]) if corner_key else sigma_vm_peak

    return {
        "sigma_vm_field": sigma_vm_field,
        "sigma_zz_field": sigma_zz_field,
        "is_solid_elem":  is_solid_elem,
        "eps_zz":         eps_zz,
        "sigma_vm_peak":  sigma_vm_peak,
        "sigma_vm_hw":    sigma_vm_hw,
        "sigma_vm_corner": sigma_vm_corner,
        "sigma_zz_hw":    sigma_zz_hw,
        "mech_map":       mech_map,
        "dof_of":         dof_of,
        "u_global":       u_g,
    }
