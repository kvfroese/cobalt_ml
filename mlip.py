"""
Implementation of all necessary functions for the MLIP potential, including
    cutoff, cutoff_derivative,
    radial_G1, radial_G1_derivative,
    angular_G2, angular_G2_derivative,
    compute_descriptors, ridge_fit, ridge_predict,
    force_chain_rule
"""

import numpy as np

def cutoff(r_ij, r_cut):
    if r_ij <= r_cut:
        fc = 0.5 * (np.cos(np.pi*r_ij/r_cut) + 1)
        return fc
    elif r_ij > r_cut:
        fc = 0.0
        return fc
    else:
        raise ValueError("Unexpected value of r_ij, this should never happen")

def cutoff_derivative(r_ij, r_cut):
    if r_ij <= r_cut:
        dfc_dr = (-0.5*np.pi/r_cut) * np.sin(np.pi*r_ij/r_cut)
        return dfc_r
    elif r_ij > r_cut:
        dfc_r = 0.0
        return dfc_dr
    else:
        raise ValueError("Unexpected value of r_ij, this should never happen")

def radial_G1(r_ij, r_s, eta):
    G1 = np.exp(-eta*(r_ij - r_s)**2) * cutoff(r_ij)
    return G1

def radial_G1_derivative(r_ij, r_s, r_cut, eta):
    exp = np.exp(-eta*(r_ij - r_s)**2)
    exp_deriv = exp * (-2*eta*(r_ij - r_s))
    dG1_dr = exp_deriv * cutoff(r_ij, r_cut) + exp * cutoff_derivative(r_ij, r_cut)
    return dG1_dr

def angular_G2(cos_theta, r_ij, r_ik, r_jk, r_cut, zeta, lambda_, eta):
    fc_ij = cutoff(r_ij, r_cut)
    fc_ik = cutoff(r_ik, r_cut)
    fc_jk = cutoff(r_jk, r_cut)
    G2 = 2**(1-zeta) * (1 + lambda_ * cos_theta)**zeta * np.exp(-eta * (r_ij**2 + r_ik**2 + r_jk**2)) * fc_ij*fc_ik*fc_jk
    return G2

def angular_G2_derivative(cos_theta, theta, r_ij, r_ik, r_jk, r_cut, zeta, lambda_, eta):
    # Big boys
    const = 2**(1-zeta)
    exp_squared_sum = np.exp(-eta * (r_ij**2 + r_ik**2 + r_jk**2))
    cos_term = (1 + lambda_ * cos_theta)**zeta # constant with respect to r_ij, r_ik, r_jk

    # Cutoffs
    fc_r_ij = cutoff(r_ij, r_cut)
    fc_r_ik = cutoff(r_ik, r_cut)
    fc_r_jk = cutoff(r_jk, r_cut)
    dfc_r_ij = cutoff_derivative(r_ij, r_cut)
    dfc_r_ik = cutoff_derivative(r_ik, r_cut)
    dfc_r_jk = cutoff_derivative(r_jk, r_cut)

    # Derivative with respect to cos_theta
    dG2_dtheta = const * (zeta*(1 + lambda_*cos_theta)**(zeta - 1)) * (-lambda_*np.sin(theta)) * (exp_squared_sum * fc_r_ij*fc_r_ik*fc_r_jk)

    # r derivatives
    dG2_dr_ij = const * (cos_term * fc_r_ik*fc_r_jk * (dfc_r_ij * exp_squared_sum + (exp_squared_sum * (-2*eta*r_ij) * fc_r_ij)))
    dG2_dr_ik = const * (cos_term * fc_r_ij*fc_r_jk * (dfc_r_ik * exp_squared_sum + (exp_squared_sum * (-2*eta*r_ik) * fc_r_ik)))
    dG2_dr_jk = const * (cos_term * fc_r_ij*fc_r_ik * (dfc_r_jk * exp_squared_sum + (exp_squared_sum * (-2*eta*r_jk) * fc_r_jk)))

    return dG2_dtheta, dG2_dr_ij, dG2_dr_ik, dG2_dr_jk

