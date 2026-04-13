"""
Unit tests for MLIP implementation
Run with: python test_mlip.py
"""

import numpy as np
import torch
from mlip import (
    cutoff, cutoff_derivative,
    radial_G1, radial_G1_derivative,
    angular_G2, angular_G2_derivative,
    compute_descriptors
)
from regression import ridge_fit, ridge_predict


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
        r2  = 2.0   # different radius

        g1_original = radial_G1(r1, Rs, Rc, eta)
        g1_same = radial_G1(r1, Rs, Rc, eta)
        g1_diff = radial_G1(r2, Rs, Rc, eta)
        
        assert abs(g1_original - g1_same) < 1e-10, "G1 not invariant to orientation"
        assert abs(g1_original - g1_diff) > 1e-6, "G1 should change with distance"
    
    # =========================================================
    # TEST 3: Angular G2 symmetry (angle sensitivity)
    # =========================================================
    def test_angular_symmetry(self):
        """G2 should be maximal at theta=0 for lambda=1, and at theta=pi for lambda=-1"""
        eta  = self.TEST_ETA
        Rc   = self.TEST_RC
        zeta = self.TEST_ZETA
        rij  = rik = 1.5     #placeholder i <-> j, i <-> k at theta=0
        rjk  = 2.5           #placeholder j <-> k at theta=0
        
        #lambda=+1: maximum @ theta=0
        g2_0    = angular_G2(np.cos(0.0), rij, rik, rjk, Rc, zeta, 1, eta)
        g2_90   = angular_G2(np.cos(np.pi/2), rij, rik, rjk, Rc, zeta, 1, eta)
        g2_180  = angular_G2(np.cos(np.pi), rij, rik, rjk, Rc, zeta, 1, eta)

        #lambda=-1: maximum @ theta=pi
        g2_0_neg    = angular_G2(np.cos(0.0), rij, rik, rjk, Rc, zeta, -1, eta)
        g2_90_neg   = angular_G2(np.cos(np.pi/2), rij, rik, rjk, Rc, zeta, -1, eta)
        g2_180_neg  = angular_G2(np.cos(np.pi), rij, rik, rjk, Rc, zeta, -1, eta)

        assert g2_0 > g2_90, f"G2(lam=+1) should peak at 0°, got {g2_0} vs {g2_90}"
        assert g2_0 > g2_180, f"G2(lam=+1) should be larger at 0° than 180°"
        assert g2_180_neg > g2_90_neg, f"G2(lam=-1) should peak at 180°"
        assert g2_180_neg > g2_0_neg, f"G2(lam=-1) should be larger at 180° than 0°"
    
    # =========================================================
    # TEST 4: Angular G2 gradient via torch.autograd
    # =========================================================
    def test_angular_gradient_autograd(self):
        """Check analytic gradient of G2 against PyTorch autograd"""

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
        
        g2 = angular_g2_torch(coords, 0, 1, 2, eta, zeta, lam, Rc)
        grad_auto = torch.autograd.grad(g2, coords, create_graph=False)[0].detach().numpy()
        
        #manual version
        grad_num = np.zeros_like(grad_auto)
        coords_np = coords.detach().numpy()
    
        for idx in range(coords.numel()):
            coords_plus = coords_np.copy()
            coords_plus.flat[idx] += eps
            
            r_ij_plus = np.linalg.norm(coords_plus[0] - coords_plus[1])
            r_ik_plus = np.linalg.norm(coords_plus[0] - coords_plus[2])
            r_jk_plus = np.linalg.norm(coords_plus[1] - coords_plus[2])
            cos_theta_plus = _compute_angle(coords_plus[0], coords_plus[1], coords_plus[2])
            cos_theta_plus = np.cos(cos_theta_plus) 
            g2_plus = angular_G2(cos_theta_plus, r_ij_plus, r_ik_plus, r_jk_plus, Rc, zeta, lam, eta)
            
            coords_minus = coords_np.copy()
            coords_minus.flat[idx] -= eps
            r_ij_minus = np.linalg.norm(coords_minus[0] - coords_minus[1])
            r_ik_minus = np.linalg.norm(coords_minus[0] - coords_minus[2])
            r_jk_minus = np.linalg.norm(coords_minus[1] - coords_minus[2])
            cos_theta_minus = _compute_angle(coords_minus[0], coords_minus[1], coords_minus[2])
            cos_theta_minus = np.cos(cos_theta_minus)
            g2_minus = angular_G2(cos_theta_minus, r_ij_minus, r_ik_minus, r_jk_minus, Rc, zeta, lam, eta)
            
            grad_num.flat[idx] = (g2_plus - g2_minus) / (2 * eps)
        
        np.testing.assert_allclose(grad_auto.flatten(), grad_num.flatten(), rtol=1e-4, atol=1e-6)
        
    # =========================================================
    # TEST 5: Ridge regression normal equations
    # =========================================================
    def test_ridge_normal_equations(self):
        """Ridge fit via normal equations should match direct linear solve"""
        np.random.seed(42)
        n_samp = self.TEST_SAMP
        n_feat = self.TEST_FEAT
        X = np.random.randn(n_samp, n_feat)
        w_true = np.random.randn(n_feat)
        y = X @ w_true + 0.01 * np.random.randn(n_samp)
        lam = self.TEST_PEN
        
        w_pred = ridge_fit(X, y, lam)
        
        X_with_bias = np.column_stack([np.ones(n_samp), X])
        n_feat_with_bias = X_with_bias.shape[1]
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y

        ridge_matrix = XtX.copy()
        ridge_matrix[1:, 1:] += lam * np.eye(n_feat_with_bias - 1)

        w_direct = np.linalg.solve(ridge_matrix, Xty)
        
        np.testing.assert_allclose(w_pred, w_direct, atol=1e-8,
                                   err_msg="Ridge fit does not match direct solver")
    
    # =========================================================
    # TEST 6: Cutoff function derivative continuity
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

def cutoff_torch(r, Rc):
                return torch.where(r <= Rc, 0.5 * (torch.cos(np.pi * r / Rc) + 1), torch.tensor(0.0))

def angular_g2_torch(coords, i, j, k, eta, zeta, lam, Rc):        

            r_ij = torch.norm(coords[i] - coords[j])
            r_ik = torch.norm(coords[i] - coords[k])
            r_jk = torch.norm(coords[j] - coords[k])
            
            vec_ij = coords[j] - coords[i]
            vec_ik = coords[k] - coords[i]
            cos_theta = torch.dot(vec_ij, vec_ik) / (r_ij * r_ik + 1e-12)
            cos_theta = torch.clamp(cos_theta, -0.999999, 0.999999)
        
        
            fc_ij = cutoff_torch(r_ij, Rc)
            fc_ik = cutoff_torch(r_ik, Rc)
            fc_jk = cutoff_torch(r_jk, Rc)
            
            G2 = (2**(1 - zeta) * 
                (1 + lam * cos_theta)**zeta * 
                torch.exp(-eta * (r_ij**2 + r_ik**2 + r_jk**2)) * 
                fc_ij * fc_ik * fc_jk)
            return G2

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    tester = TestMLIP()
    
    test_methods = [
        tester.test_cutoff_smoothness,
        tester.test_radial_invariance,
        tester.test_angular_symmetry,
        tester.test_angular_gradient_autograd,
        tester.test_ridge_normal_equations,
        tester.test_cutoff_derivative_continuity,
    ]
    
    for test in test_methods:
        tester.run_test(test)
    
    success = tester.print_summary()
    exit(0 if success else 1)