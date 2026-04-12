"""
Implementation of all necessary functions for the MLIP potential, including
    cutoff, cutoff_derivative,
    radial_G1, radial_G1_derivative,
    angular_G2, angular_G2_derivative,
    compute_descriptors, ridge_fit, ridge_predict,
    force_chain_rule
"""

from pathlib import Path
import numpy as np
import configargparse
import pickle

from utility import path_handler

## Parser Setup

def parser_client():
    parser = configargparse.ArgParser(
        description='Highly recommend to use a config file instead of manual arguments. If unspecififed, a default one will be used!',
        default_config_files=[Path('config/default.ini')]
    )

    # Arguments
    ## File stuff
    parser.add_argument('-c',
                        '--my-config',
                        is_config_file=True,
                        help="Path of custom config file"
                        )
    parser.add_argument('--pair_file_name',
                        type=str,
                        help="Name of file that contains all pair geometry values"
                        )
    parser.add_argument('--triplet_file_name',
                        type=str,
                        help="Name of file that contains all triplet/matched pairs geometry values")
    parser.add_argument('--geometry_folder',
                        type=str,
                        help="Folder path to all geometry values. Recommend do not change")
    
    ## Values
    parser.add_argument('--eta',
                        type=float,
                        help="Hyperparameter for narrowness of peaks, large eta = narrow peak")
    parser.add_argument('--zeta',
                        type=float,
                        help="Hyperparameter for angular resolution of peaks, larger value = sharper angles")
    parser.add_argument('--r_s',
                        type=float,
                        help="Radial shift value for radial descriptor")
    parser.add_argument('--r_cut',
                        type=float,
                        help='Cutoff value for cutoff function, angstroms')
    parser.add_argument('--lambda_',
                        type=float,
                        help="Signed value for angular descriptor")
    
    args, unknown_args = parser.parse_known_args()
    return args

if __name__ == '__main__':
    parser = parser_client()

    pair_file_name = parser.pair_file_name
    triplet_file_name = parser.triplet_file_name
    geometry_folder = parser.geometry_folder
    eta = parser.eta
    zeta = parser.zeta
    r_s = parser.r_s
    r_cut = parser.r_cut
    lambda_ = parser.lambda_

## Opening Atomic Geomtry

try:
    triplet_file_path, _ = path_handler(triplet_file_name, geometry_folder)
    with open(triplet_file_path, 'rb') as f:
        triplet_geometry = pickle.load(f)
        print("Triplet geometry loaded successfully")
    pair_file_path, _ = path_handler(pair_file_name, geometry_folder)
    with open(pair_file_path, 'rb') as f:
        pair_geometry = pickle.load(f)
        print("Pair geometry loaded successfully")
except Exception as e:
    print(f"Error in opening file due to:\n{e}")

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
        return dfc_dr
    elif r_ij > r_cut:
        dfc_dr = 0.0
        return dfc_dr
    else:
        raise ValueError("Unexpected value of r_ij, this should never happen")

def radial_G1(r_ij, r_s, r_cut, eta):
    G1 = np.exp(-eta*(r_ij - r_s)**2) * cutoff(r_ij, r_cut)
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

    grad = (dG2_dtheta, (dG2_dr_ij, dG2_dr_ik, dG2_dr_jk))
    return grad
'''
triplet_geometry[a][b][c][d][e],
- a structure
- b=0,1 for file name or list of values for each atom
- c is one set of data values
- d=0,1 for angle values or distance values
- e=0,1 (cos_theta_ijk, theta_ijk) or 0,1,2 (r_ij, r_ik, r_jk) for choosing said values
'''
'''
pair_geometry[a][b][c][d]
- a structure
- b=0,1 for file name or list of values for each atom
- c for each atom
- d=0,1 for id or value
'''

print(pair_geometry[0][0])


def compute_descriptors(triplet_geometry, pair_geometry, eta, zeta, lambda_, r_cut, r_s):
    '''
    Compute descriptors for all atoms in each structure.
    
    Parameters:
    - triplet_geometry, pair_geometry: loaded data structures
    - eta, zeta, lambda_, r_cut: hyperparameters
    - r_s, shift value for G1, hyperparameter

    Returns:
    - ang_descriptor: for each structure we have a set of the summed angular descriptor,
    as well as the elementwise summed 4D gradient vector (theta, plus 3 r's)
    - rad_descriptor: for each structure we have a set of the summed radial descriptor
    and the summed derivative
    '''
    
    ang_descriptor = []
    for struct in triplet_geometry: # onto structures, level a
        data_values = struct[1]  # onto data values, level b

        ang_G = 0
        theta_deriv = 0
        r_ij_deriv = 0
        r_ik_deriv = 0
        r_jk_deriv = 0
        for vals in data_values: # iterating through each atom, level c
            r_ij, r_ik, r_jk = vals[1][0], vals[1][1], vals[1][2]
            cos_theta, theta = vals[0][0], vals[0][1] # angle, theta
            ang_G += angular_G2(cos_theta, r_ij, r_ik, r_jk, r_cut, zeta, lambda_, eta)
            ang_G_deriv = angular_G2_derivative(cos_theta, theta, r_ij, r_ik, r_jk, r_cut, zeta, lambda_, eta)
            theta_deriv += ang_G_deriv[0]
            r_ij_deriv += ang_G_deriv[1][0]
            r_ik_deriv += ang_G_deriv[1][1]
            r_jk_deriv += ang_G_deriv[1][2]

        ang_descriptor.append((struct[0], ang_G, (theta_deriv, (r_ij_deriv, r_ik_deriv, r_jk_deriv))))
    
    rad_descriptor = []
    for struct in pair_geometry: # onto structures, level a
        data_values = struct[1] # onto data values, level b
        
        rad_G = 0
        rad_G_deriv = 0
        for vals in data_values: # iterating through each atom, level c
            r = vals[1]
            rad_G += radial_G1(r, r_s, r_cut, eta)
            rad_G_deriv += radial_G1_derivative(r, r_s, r_cut, eta)
        
        rad_descriptor.append((struct[0], rad_G, rad_G_deriv))
    
    return ang_descriptor, rad_descriptor

ang_descriptor, rad_descriptor  = compute_descriptors(triplet_geometry, pair_geometry, eta, zeta, lambda_, r_cut, r_s)
print(f"Radial Descriptor:\n{str(rad_descriptor):.200} ...\nAngular Descriptr:\n{str(ang_descriptor):200} ...")