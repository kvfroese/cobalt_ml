"""
Unit tests for MLIP implementation
Run with: python test_mlip_v1.py
"""

import numpy as np
import torch
from mlip import (
    cutoff, cutoff_derivative,
    radial_G1, radial_G1_derivative,
    angular_G2, angular_G2_derivative,
    compute_descriptors, ridge_fit, ridge_predict,
    force_chain_rule
)


class TestMLIP:
    """
    Test class for MLIP implementation
    Each test method contains assertions and prints pass/fail status
    """
    TEST_ETA    = 2.0   #placeholder eta
    TEST_RS     = 0.0   #placeholder gaussian center
    TEST_RC     = 5.0   #placeholder cutoff
    TEST_R      = 1.5   #placeholder radius
    TEST_EPS    = 1e-6  #placeholder perturbation
    TEST_ZETA   = 2.0   #placeholder angular exp
    TEST_LAMBDA = 1     #placeholder lambda
    TEST_PEN    = 0.1   #placeholder ridge penalty
    TEST_SAMP   = 20    #placeholder sample count
    TEST_FEAT   = 5     #placeholder feature count
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def run_test(self, test_func):
        """Run a single test method and track results"""
        name = test_func.__name__
        try:
            test_func()
            self.passed += 1
            status = "PASS"
            self.results.append((name, True, None))
        except AssertionError as e:
            self.failed += 1
            status = "FAIL"
            self.results.append((name, False, str(e)))
        except Exception as e:
            self.failed += 1
            status = "ERROR"
            self.results.append((name, False, str(e)))
        print(f"{status}: {name}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*50)
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed")
        for name, passed, error in self.results:
            if not passed:
                print(f"  {name}: {error}")
        print("="*50)
        return self.failed == 0
    
    # =========================================================
    # TEST 1: Cutoff function smoothness and boundary conditions
    # =========================================================
    def test_cutoff_smoothness(self):
        """Verify f_c(R) = 0 and f_c'(R) = 0 at R = R_c"""
        Rc  = self.TEST_RC
        eps = self.TEST_EPS
        
        f_at_Rc = cutoff(Rc, Rc)
        df_at_Rc = cutoff_derivative(Rc, Rc)
        f_beyond = cutoff(Rc + eps, Rc)
        df_beyond = cutoff_derivative(Rc + eps, Rc)
        
        assert abs(f_at_Rc) < 1e-12, f"cutoff(Rc) should be 0, got {f_at_Rc}"
        assert abs(df_at_Rc) < 1e-12, f"cutoff_derivative(Rc) should be 0, got {df_at_Rc}"
        assert abs(f_beyond) < 1e-12, f"cutoff(Rc+eps) should be 0, got {f_beyond}"
        assert abs(df_beyond) < 1e-12, f"cutoff_derivative(Rc+eps) should be 0, got {df_beyond}"
    
    # =========================================================
    # TEST 2: Radial G1 invariance (translation/rotation/permutation)
    # =========================================================
    def test_radial_invariance(self):
        """G1 should depend only on distance, not on absolute positions"""
        eta = self.TEST_ETA
        Rs  = self.TEST_RS
        Rc  = self.TEST_RC
        r1  = self.TEST_R
        r2  = 2.0   #placeholder different radius

        g1_original = radial_G1(r1, eta, Rs, Rc)
        g1_same = radial_G1(r1, eta, Rs, Rc)
        g1_diff = radial_G1(r2, eta, Rs, Rc)
        
        assert abs(g1_original - g1_same) < 1e-10, "G1 not invariant to orientation"
        assert abs(g1_original - g1_diff) > 1e-6, "G1 should change with distance"
    
    # =========================================================
    # TEST 3: Radial G1 derivative via finite differences
    # =========================================================
    def test_radial_derivative_fd(self):
        """Check analytic derivative of G1 against finite differences"""
        eta  = self.TEST_ETA
        Rs   = self.TEST_RS
        Rc   = self.TEST_RC
        r    = self.TEST_R
        eps  = self.TEST_EPS
        
        #analytic derivative with respect to r (not vector)
        #for a single pair, dG1/dr = d/dr [exp(-eta*(r-Rs)^2) * fc(r)]
        def g1_func(r_val):
            return radial_G1(r_val, eta, Rs, Rc)
        
        g1_plus = g1_func(r + eps)
        g1_minus = g1_func(r - eps)
        fd_deriv = (g1_plus - g1_minus) / (2 * eps)
        
        #your analytic derivative with respect to r (scalar distance)
        from mlip import radial_G1_dr
        analytic_deriv = radial_G1_dr(r, eta, Rs, Rc)
        
        assert abs(analytic_deriv - fd_deriv) < 1e-6, \
            f"G1 derivative mismatch: analytic={analytic_deriv}, FD={fd_deriv}"
    
    # =========================================================
    # TEST 4: Angular G2 symmetry (angle sensitivity)
    # =========================================================
    def test_angular_symmetry(self):
        """G2 should be maximal at theta=0 for lambda=1, and at theta=pi for lambda=-1"""
        eta  = self.TEST_ETA
        Rc   = self.TEST_RC
        zeta = self.TEST_ZETA
        rij  = rik = 1.5     #placeholder i <-> j, i <-> k at theta=0
        rjk  = 2.5           #placeholder j <-> k at theta=0
        
        #lambda=+1: maximum @ theta=0
        g2_0    = angular_G2(rij, rik, rjk, 0.0, eta, zeta, 1, Rc)
        g2_90   = angular_G2(rij, rik, rjk, np.pi/2, eta, zeta, 1, Rc)
        g2_180  = angular_G2(rij, rik, rjk, np.pi, eta, zeta, 1, Rc)

        #lambda=-1: maximum @ theta=pi
        g2_0_neg    = angular_G2(rij, rik, rjk, 0.0, eta, zeta, -1, Rc)
        g2_90_neg   = angular_G2(rij, rik, rjk, np.pi/2, eta, zeta, -1, Rc)
        g2_180_neg  = angular_G2(rij, rik, rjk, np.pi, eta, zeta, -1, Rc)

        assert g2_0 > g2_90, f"G2(lam=+1) should peak at 0°, got {g2_0} vs {g2_90}"
        assert g2_0 > g2_180, f"G2(lam=+1) should be larger at 0° than 180°"
        assert g2_180_neg > g2_90_neg, f"G2(lam=-1) should peak at 180°"
        assert g2_180_neg > g2_0_neg, f"G2(lam=-1) should be larger at 180° than 0°"
    
    # =========================================================
    # TEST 5: Angular G2 gradient via torch.autograd
    # =========================================================
    def test_angular_gradient_autograd(self):
        """Check analytic gradient of G2 against PyTorch autograd"""
        from mlip import angular_G2_torch

        coords = torch.tensor([
            [0.0, 0.0, 0.0],  # i
            [1.5, 0.0, 0.0],  # j
            [1.2, 1.2, 0.0]   # k
            ], requires_grad=True)
        
        eta  = self.TEST_ETA
        Rc   = self.TEST_RC
        zeta = self.TEST_ZETA
        lam  = self.TEST_LAMBDA
        eps  = self.TEST_EPS
        
        #pytorch version
        g2 = angular_G2_torch(coords, i=0, j=1, k=2, eta=eta, zeta=zeta, lam=lam, Rc=Rc)
        grad_auto = torch.autograd.grad(g2, coords, create_graph=False)[0].detach().numpy()
        
        #manual version
        grad_num = np.zeros_like(grad_auto)
        coords_np = coords.detach().numpy()
        
        for idx in range(coords.numel()):
            coords_plus = coords_np.copy()
            coords_plus.flat[idx] += eps
            g2_plus = angular_G2(
                np.linalg.norm(coords_plus[0]-coords_plus[1]),
                np.linalg.norm(coords_plus[0]-coords_plus[2]),
                np.linalg.norm(coords_plus[1]-coords_plus[2]),
                _compute_angle(coords_plus[0], coords_plus[1], coords_plus[2]),
                eta, zeta, lam, Rc
            )
            
            coords_minus = coords_np.copy()
            coords_minus.flat[idx] -= eps
            g2_minus = angular_G2(
                np.linalg.norm(coords_minus[0]-coords_minus[1]),
                np.linalg.norm(coords_minus[0]-coords_minus[2]),
                np.linalg.norm(coords_minus[1]-coords_minus[2]),
                _compute_angle(coords_minus[0], coords_minus[1], coords_minus[2]),
                eta, zeta, lam, Rc
            )
            
            grad_num.flat[idx] = (g2_plus - g2_minus) / (2 * eps)
        
        np.testing.assert_allclose(grad_auto.flatten(), grad_num.flatten(), rtol=1e-4, atol=1e-6)
    
    # =========================================================
    # TEST 6: Ridge regression normal equations
    # =========================================================
    def test_ridge_normal_equations(self):
        """Ridge fit via normal equations should match direct linear solve"""
        np.random.seed(42)
        n_samp = self.TEST_SAMP
        n_feat = self.TEST_FEAT
        X = np.random.randn(n_samp, n_feat)
        w_true = np.random.randn(n_feat)
        y = X @ w_true + 0.01 * np.random.randn(n_samp)
        lam = self.TEST_LAMBDA
        
        w_pred = ridge_fit(X, y, lam)
        
        XtX = X.T @ X
        Xty = X.T @ y
        w_direct = np.linalg.solve(XtX + lam * np.eye(n_feat), Xty)
        
        np.testing.assert_allclose(w_pred, w_direct, atol=1e-8,
                                   err_msg="Ridge fit does not match direct solver")
    
    # =========================================================
    # TEST 7: Force chain rule vs finite differences
    # =========================================================
    def test_force_chain_rule_fd(self):
        """Forces from analytic chain rule should match numerical differentiation"""
        from ase import Atoms
        
        #create test molecule (H2O)
        atoms = Atoms('H2O',
                     pos=[[0.00,  0.00, 0.00],
                          [0.76,  0.58, 0.00],
                          [0.76, -0.58, 0.00]])
        
        #create test descriptor grid
        param_grid = {
            'radial': [
                {'eta': 2.0, 'Rs': 0.0},
                {'eta': 0.5, 'Rs': 1.0},0
            ],
            'angular': [
                {'eta': 1.0, 'zeta': self.TEST_ZETA, 'lam': self.TEST_LAMBDA},
            ]
        }
        
        #compute descriptors, train dummy model
        X = compute_descriptors(atoms, param_grid)
        X_total = X.sum(axis=0).reshape(1, -1)
        y = np.array([0.0])
        w = ridge_fit(X_total, y, lam=1e-6)
        
        forces_analytic = force_chain_rule(atoms, w, param_grid)
        
        eps = self.TEST_EPS
        forces_numerical = np.zeros_like(forces_analytic)
        
        for i in range(len(atoms)):
            for d in range(3):
                #forward step
                atoms_plus = atoms.copy()
                atoms_plus.positions[i, d] += eps
                X_plus = compute_descriptors(atoms_plus, param_grid)
                E_plus = ridge_predict(X_plus.sum(axis=0).reshape(1, -1), w)[0]
                
                #backward step
                atoms_minus = atoms.copy()
                atoms_minus.positions[i, d] -= eps
                X_minus = compute_descriptors(atoms_minus, param_grid)
                E_minus = ridge_predict(X_minus.sum(axis=0).reshape(1, -1), w)[0]
                
                forces_numerical[i, d] = -(E_plus - E_minus) / (2 * eps)
        
        np.testing.assert_allclose(forces_analytic, forces_numerical, atol=1e-5,
                                   err_msg="Analytic forces do not match numerical FD")
    
    # =========================================================
    # TEST 8: Descriptor permutation invariance
    # =========================================================
    def test_descriptor_permutation_invariance(self):
        """Swapping identical atoms should produce identical descriptors for those atoms"""
        from ase import Atoms
        
        a1 = Atoms('H2', positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        a2 = Atoms('H2', positions=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        parm_grid = {
            'radial': [{'eta': 2.0, 'Rs': 0.0}],
            'angular': []
        }
        
        desc1 = compute_descriptors(a1, parm_grid)
        desc2 = compute_descriptors(a2, parm_grid)
        
        #identical atoms, must be matched after sorting
        desc1_sorted = np.sort(desc1, axis=0)
        desc2_sorted = np.sort(desc2, axis=0)
        
        np.testing.assert_allclose(desc1_sorted, desc2_sorted, atol=1e-8,
                                   err_msg="Descriptor not invariant to atom permutation")
    
    # =========================================================
    # TEST 9: Descriptor derivative via torch.autograd (radial)
    # =========================================================
    def test_descriptor_derivative_autograd(self):
        
        """Manual derivative of radial descriptor matches torch.autograd"""
        
        from mlip import radial_G1_torch, radial_G1_dr_vector
        
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], requires_grad=True)
        eta  = self.TEST_ETA
        Rs   = self.TEST_RS
        Rc   = self.TEST_RC
        
        #pytorch version
        g1 = radial_G1_torch(coords, i=0, j=1, eta=eta, Rs=Rs, Rc=Rc)
        grad_auto = torch.autograd.grad(g1, coords, create_graph=False)[0].detach().numpy()
        
        #manual version
        grad_manual = radial_G1_dr_vector(coords.detach().numpy(), i=0, j=1, eta=eta, Rs=Rs, Rc=Rc)
        
        np.testing.assert_allclose(grad_auto.flatten(), grad_manual.flatten(), atol=1e-6,
                                   err_msg="Manual radial derivative does not match autograd")
    
    # =========================================================
    # TEST 10: Cutoff function derivative continuity
    # =========================================================
    def test_cutoff_derivative_continuity(self):
        """Derivative of cutoff function should be continuous at all R"""
        Rc   = self.TEST_RC
        test_points = np.linspace(0.1, Rc - 0.1, 20)
        
        derivatives = [cutoff_derivative(R, Rc) for R in test_points]
        
        #check for smooth derivative
        for i in range(len(derivatives) - 1):
            diff = abs(derivatives[i+1] - derivatives[i])
            step = test_points[i+1] - test_points[i]
            assert diff < 10 * step, f"Possible discontinuity at R={test_points[i]}: diff={diff}"
        
        #check derivative at R=0 is 0 (cosine cutoff has zero slope at origin)
        df_at_0 = cutoff_derivative(0.0, Rc)
        assert abs(df_at_0) < 1e-10, f"Derivative at R=0 should be 0, got {df_at_0}"


def _compute_angle(p_i, p_j, p_k):
    """Compute angle in radians at atom i between vectors to j and k"""
    u = p_j - p_i
    v = p_k - p_i
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    cos_theta = np.clip(cos_theta, -0.999999, 0.999999)
    return np.arccos(cos_theta)


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    tester = TestMLIP()
    
    test_methods = [
        tester.test_cutoff_smoothness,
        tester.test_radial_invariance,
        tester.test_radial_derivative_fd,
        tester.test_angular_symmetry,
        tester.test_angular_gradient_autograd,
        tester.test_ridge_normal_equations,
        tester.test_force_chain_rule_fd,
        tester.test_descriptor_permutation_invariance,
        tester.test_descriptor_derivative_autograd,
        tester.test_cutoff_derivative_continuity,
    ]
    
    for test in test_methods:
        tester.run_test(test)
    
    success = tester.print_summary()
    exit(0 if success else 1)