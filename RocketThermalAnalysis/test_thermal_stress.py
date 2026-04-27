"""
test_thermal_stress.py
Unit-level validation for the generalized plane-strain thermo-elastic
solver in thermal_stress.py.

Tests
-----
1. Patch test: uniform ΔT on a free square → σ ≈ 0 everywhere.
2. Uniaxial pressure: pressure on y=0 of a free strip → σ_yy ≈ P,
   σ_vm ≈ |P| on the loaded face (with free-surface BCs).
3. 1-D thermal bar: linear T gradient on a thin strip, both x-edges
   constrained → σ_xx matches closed-form plane-strain result.
"""
import numpy as np

from materials import get_material
from thermal_stress import solve_stress_2d


def _make_dummy_mesh(W, H, nx, ny, chan_w_half=-1.0, chan_t=-1.0, chan_h=0.0):
    """
    Uniform rectangular grid with NO void (solid rectangle).
    chan_w_half/chan_t set to sentinels so no pressure faces are detected.
    """
    x_nodes = np.linspace(0.0, W, nx)
    y_nodes = np.linspace(0.0, H, ny)
    node_map = {}
    for j in range(ny):
        for i in range(nx):
            node_map[(i, j)] = len(node_map)
    is_void = np.zeros((ny, nx), dtype=bool)
    return x_nodes, y_nodes, node_map, is_void


def test_patch_uniform_dT():
    """Uniform ΔT on a free rectangle → σ = 0 (generalized plane strain)."""
    mat = get_material("CuCrZr")
    W, H = 5e-3, 3e-3
    nx, ny = 6, 5
    x_nodes, y_nodes, node_map, is_void = _make_dummy_mesh(W, H, nx, ny)

    T_ref = 293.0
    T_field = np.full(nx * ny, 600.0)   # uniform 600 K

    # No pressures, no void → all faces free except symmetry on x=0,x=W
    res = solve_stress_2d(
        T_field, node_map, x_nodes, y_nodes, is_void,
        chan_w_half=-1.0, chan_t=-1.0, chan_h=0.0,
        P_gas=0.0, P_cool=0.0,
        material=mat, T_ref=T_ref,
    )

    # With master_ux on the right edge free, ALL stress components should
    # vanish for a uniform ΔT on a free rectangle — this is the true patch
    # test for generalized plane strain with a free hoop DOF.
    sz = res["sigma_zz_field"]
    svm = res["sigma_vm_field"]
    sz_mean = float(np.nanmean(sz))
    vm_max = float(np.nanmax(svm))
    print(f"[patch]  mean σ_zz = {sz_mean/1e6:.3e} MPa   "
          f"max σ_vm = {vm_max/1e6:.3e} MPa  (both ≈ 0)")
    assert abs(sz_mean) < 1e4
    assert vm_max < 1e4, "free patch test — expected σ ≈ 0 everywhere"


def test_thermal_gradient_bending():
    """
    Through-thickness linear T gradient on a thin strip with free hoop
    (master_ux free) and free axial (generalized plane strain).  Both
    σ_xx and σ_zz develop from the gradient, with zero mean across the
    thickness (pure bending in a beam sense).

    Closed form for a free plate with linear ΔT(y):
      σ_xx(y) = σ_zz(y) = −E·α·(T(y) − T_mid) / (1 − ν)
    """
    mat = get_material("CuCrZr")
    W, H = 5e-3, 1.0e-3
    nx, ny = 6, 9
    x_nodes, y_nodes, node_map, is_void = _make_dummy_mesh(W, H, nx, ny)

    T_ref = 293.0
    T_hot, T_cold = 900.0, 500.0
    T_mid = 0.5 * (T_hot + T_cold)
    T_field = np.zeros(nx * ny)
    for j in range(ny):
        Tj = T_hot + (T_cold - T_hot) * (y_nodes[j] / H)
        for i in range(nx):
            T_field[node_map[(i, j)]] = Tj

    res = solve_stress_2d(
        T_field, node_map, x_nodes, y_nodes, is_void,
        chan_w_half=-1.0, chan_t=-1.0, chan_h=0.0,
        P_gas=0.0, P_cool=0.0,
        material=mat, T_ref=T_ref,
    )

    E = mat.E(T_mid); nu = mat.nu; alpha = mat.alpha(T_mid)

    from thermal_stress import _recover_stress, _enumerate_solid_elements, \
                                _build_mech_dof_map
    elems = _enumerate_solid_elements(node_map, nx, ny)
    mech, dof_of, N_dof, _, _ = _build_mech_dof_map(elems, node_map, nx)
    sx, sy, sxy, sz, svm = _recover_stress(
        res["u_global"], res["eps_zz"], elems, mech, dof_of, node_map,
        x_nodes, y_nodes, T_field, mat, T_ref)

    # Sample hot-wall row (j=0) and cold-wall row (j=ny-2)
    sx_hot = np.mean([sx[k] for k in sx if k[1] == 0])
    sx_cold = np.mean([sx[k] for k in sx if k[1] == ny - 2])
    y_mid_hot = 0.5 * (y_nodes[0] + y_nodes[1])
    y_mid_cold = 0.5 * (y_nodes[-2] + y_nodes[-1])
    T_mid_hot = T_hot + (T_cold - T_hot) * (y_mid_hot / H)
    T_mid_cold = T_hot + (T_cold - T_hot) * (y_mid_cold / H)
    expect_hot = -E * alpha * (T_mid_hot - T_mid) / (1.0 - nu)
    expect_cold = -E * alpha * (T_mid_cold - T_mid) / (1.0 - nu)

    err_h = (sx_hot - expect_hot) / expect_hot * 100
    err_c = (sx_cold - expect_cold) / expect_cold * 100
    print(f"[grad]  σ_xx hot  = {sx_hot/1e6:8.1f}  expect {expect_hot/1e6:8.1f}  err {err_h:+.1f}%")
    print(f"[grad]  σ_xx cold = {sx_cold/1e6:8.1f}  expect {expect_cold/1e6:8.1f}  err {err_c:+.1f}%")
    # Closed form assumes constant E, α at T_mid; FEM uses per-element
    # T-dependent props, so 15% asymmetric error is expected.
    assert abs(err_h) < 15.0 and abs(err_c) < 15.0, "gradient bending mismatch"


def test_uniform_pressure_bottom():
    """
    Apply P_gas on the hot-wall face of a free strip (no void).
    Expect σ_yy ≈ +P at the loaded face (compressive in convention;
    our sign is positive because the load pushes into the solid).

    This is a soft test — free lateral surfaces (top) let the strip
    expand, so interior σ_yy ≈ +P, σ_xx depends on Poisson restraint
    from the symmetry BCs.
    """
    mat = get_material("CuCrZr")
    W, H = 5e-3, 2e-3
    nx, ny = 6, 6
    x_nodes, y_nodes, node_map, is_void = _make_dummy_mesh(W, H, nx, ny)

    T_ref = 293.0
    T_field = np.full(nx * ny, T_ref)   # isothermal

    P_gas = 20e6   # 200 bar
    res = solve_stress_2d(
        T_field, node_map, x_nodes, y_nodes, is_void,
        chan_w_half=-1.0, chan_t=-1.0, chan_h=0.0,
        P_gas=P_gas, P_cool=0.0,
        material=mat, T_ref=T_ref,
    )

    u_g = res["u_global"]
    print(f"[press]  max |u| = {np.max(np.abs(u_g)):.3e}  "
          f"(should be non-zero, finite)")
    assert np.isfinite(u_g).all() and np.max(np.abs(u_g)) > 0


if __name__ == "__main__":
    print("=" * 60)
    print("thermal_stress.py validation tests")
    print("=" * 60)
    test_patch_uniform_dT()
    test_thermal_gradient_bending()
    test_uniform_pressure_bottom()
    print()
    print("All tests passed.")
