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

from utility import path_handler, file_saver, file_loader

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
                        help="Path of custom config file")
    parser.add_argument('--geometry_file_names',
                        type=str,
                        help="Base name of files that contain geometry values"
                        )
    parser.add_argument('--geometry_folder',
                        type=str,
                        help="Folder path to all geometry values. Recommend do not change")
    parser.add_argument('--descriptor_file_names',
                        type=str,
                        help="Base name of descriptor outputs")
    parser.add_argument('--descriptor_folder',
                        type=str,
                        help="Location of descriptor output. Recoomend do not change")
    parser.add_argument('--orca_read_folder',
                        type=str,
                        help="Path of where your Orca .out's should be")
    
    ## Parameters
    parser.add_argument('--epsilon',
                        type=float,
                        help="Shift value to prevent errors during Cartesian projection")
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

parser = parser_client()

geometry_file_name = parser.geometry_file_names
geometry_folder = parser.geometry_folder
descriptor_file_name = parser.descriptor_file_names
descriptor_folder = parser.descriptor_folder
orca_outputs = parser.orca_read_folder

epsilon = parser.epsilon
eta = parser.eta
zeta = parser.zeta
r_s = parser.r_s
r_cut = parser.r_cut
lambda_ = parser.lambda_

## Loading Atomic Geomtry

pair_geometry = file_loader(geometry_file_name, geometry_folder, "pair")
triplet_geometry = file_loader(geometry_file_name, geometry_folder, "triplet")

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

def radial_G1_derivative(r_ij, r_s, eta, cutoff_deriv):
    exp_term = np.exp(-eta*(r_ij - r_s)**2)
    dG1_dr = -2*eta*(r_ij - r_s) * exp_term + exp_term
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
    exp_term = np.exp(-eta * (r_ij**2 + r_ik**2 + r_jk**2))
    cos_term = (1 + lambda_ * cos_theta)**zeta # constant with respect to r_ij, r_ik, r_jk

    # Cutoffs
    fc_r_ij = cutoff(r_ij, r_cut)
    fc_r_ik = cutoff(r_ik, r_cut)
    fc_r_jk = cutoff(r_jk, r_cut)
    f_c_tot = fc_r_ij * fc_r_ik * fc_r_jk
    
    dfc_r_ij = cutoff_derivative(r_ij, r_cut)
    dfc_r_ik = cutoff_derivative(r_ik, r_cut)
    dfc_r_jk = cutoff_derivative(r_jk, r_cut)

    # Derivative with respect to theta
    dG2_dtheta = (const * exp_term * f_c_tot) * zeta*lambda_*(1 + lambda_*cos_theta)**(zeta - 1) * (-np.sin(theta))
    # r derivatives
    dG2_dr_ij = const*cos_term * (fc_r_ik*fc_r_jk) * ((exp_term*(-2*eta*r_ij)*fc_r_ij) + exp_term*dfc_r_ij)
    dG2_dr_ik = const*cos_term * (fc_r_ij*fc_r_jk) * ((exp_term*(-2*eta*r_ik)*fc_r_ik) + exp_term*dfc_r_ik)
    dG2_dr_jk = const*cos_term * (fc_r_ij*fc_r_ik) * ((exp_term*(-2*eta*r_jk)*fc_r_jk) + exp_term*dfc_r_jk)

    grad = (dG2_dtheta, dG2_dr_ij, dG2_dr_ik, dG2_dr_jk)
    return grad

def r_internal_cart_proj(dF_dr, r_unit_vec):
    # Simple application of chain rule
    # r_ij_unit_vec is vector with form (x, y, z) and internal_grad_r_ij is a scalar
    proj_grad = -r_unit_vec * dF_dr
    return proj_grad

def theta_internal_cart_proj(cos_theta, theta, r_ij, r_ik, r_ij_unit_vec, r_ik_unit_vec, dF_dtheta):
    proj = r_ij_unit_vec/(np.sin(theta) + epsilon)*(1/r_ik + cos_theta/r_ij) + r_ik_unit_vec/(np.sin(theta) + epsilon)*(1/r_ij + cos_theta/r_ik)
    proj_grad = proj * dF_dtheta
    return proj_grad

def compute_descriptor_grads(triplet_geometry, pair_geometry, eta, zeta, lambda_, r_cut, r_s):
    rad_desc_grads = []
    
    for struct in pair_geometry: # onto structures, level a
        data_values = struct[1]  # onto data values, level b
        unit_values = struct[2]  # onto displacements, level b
        
        rad_G_grad = np.array([0.0, 0.0, 0.0])  # summed Cartesian gradient
        
        for i, dist_val in enumerate(data_values):  # iterate through each pair, level c
            r_ij = dist_val[1]  # scalar distance
            r_ij_unit_vec = np.array(unit_values[i][1]) # unit vector from displacement

            dfc_dr = cutoff_derivative(r_ij, r_cut)
            dG1_dr = radial_G1_derivative(r_ij, r_s, eta, dfc_dr)
            
            # Project scalar derivative to Cartesian coordinates
            cart_grad = r_internal_cart_proj(dG1_dr, r_ij_unit_vec)
            rad_G_grad += cart_grad
        
        rad_desc_grads.append((struct[0], rad_G_grad))
    
    # chain rule for angular symmetry derivative:
    # dG2/dalpha = dG2/dtheta * dtheta/dalpha
    #             + dG2/dr_ij * dr_ij/dalpha
    #             + dG2/dr_ik * dr_ik/dalpha
    #             + dG2/dr_jk * dr_jk/dalpha
    ang_desc_grads = []

    for struct in triplet_geometry: # onto structures, level a
        data_values = struct[1] # onto data values, level b,
        ang_G_grad = np.array([0.0, 0.0, 0.0])

        for vals in data_values:
        
            cos_theta = vals[0][0]
            theta = vals[0][1]
            r_ij, r_ik, r_jk = vals[1] # scalars

            dG2_dtheta, dG2_dr_ij, dG2_dr_ik, dG2_dr_jk = angular_G2_derivative(
                cos_theta, theta, r_ij, r_ik, r_jk,
                r_cut, zeta, lambda_, eta
            )

            if len(vals) > 2:
                r_ij_unit_vec = np.array(vals[2][0])
                r_ik_unit_vec = np.array(vals[2][1])
                r_jk_unit_vec = np.array(vals[2][2])
            else:
                raise ValueError(
                    "Triplet data must include displacement unit vectors for Cartesian projection"
                )

            theta_grad = theta_internal_cart_proj(
                cos_theta, theta, r_ij, r_ik,
                r_ij_unit_vec, r_ik_unit_vec,
                dG2_dtheta
            )
            r_ij_grad = r_internal_cart_proj(dG2_dr_ij, r_ij_unit_vec)
            r_ik_grad = r_internal_cart_proj(dG2_dr_ik, r_ik_unit_vec)
            r_jk_grad = r_internal_cart_proj(dG2_dr_jk, r_jk_unit_vec)

            ang_G_grad += theta_grad + r_ij_grad + r_ik_grad + r_jk_grad

        ang_desc_grads.append((struct[0], ang_G_grad))

    return rad_desc_grads, ang_desc_grads

rad_desc_grads, ang_desc_grads = compute_descriptor_grads(triplet_geometry, pair_geometry, eta, zeta, lambda_, r_cut, r_s)
'''
triplet_geometry[a][b][c][d][e],
- a structure
- b=0,1,2 for file name or list of values for each atom
- c is one set of data values
- d=0,1 for angle values or distance values
- e=0,1 (cos_theta_ijk, theta_ijk) or 0,1,2 (r_ij, r_ik, r_jk) for choosing said values
'''
'''
pair_geometry[a][b][c][d]
- a structure
- b=0,1,2 for file name, distance, or displacement, per atom
- c for each atom
- d=0,1 for id or values
- e=0,1,2 for i, j, k/x, y, z (displacements only)
'''


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
        for vals in data_values: # iterating through each atom, level c
            r_ij, r_ik, r_jk = vals[1][0], vals[1][1], vals[1][2]
            cos_theta, theta = vals[0][0], vals[0][1] # angle, theta
            ang_G += angular_G2(cos_theta, r_ij, r_ik, r_jk, r_cut, zeta, lambda_, eta)

        ang_descriptor.append((struct[0], ang_G)) # theta_deriv, r_ij_deriv, r_ik_deriv, r_jk_deriv))
    
    rad_descriptor = []
    for struct in pair_geometry: # onto structures, level a
        data_values = struct[1] # onto data values, level b
        
        rad_G = 0
        for vals in data_values: # iterating through each atom, level c
            r_scal = vals[1]
            rad_G += radial_G1(r_scal, r_s, r_cut, eta)
        
        rad_descriptor.append((struct[0], rad_G)) # rad_G_deriv))
    
    return ang_descriptor, rad_descriptor

ang_descriptor, rad_descriptor  = compute_descriptors(triplet_geometry, pair_geometry, eta, zeta, lambda_, r_cut, r_s)
print(f"Radial Descriptor:\n{str(rad_descriptor):.200} ...\nAngular Descriptor:\n{str(ang_descriptor):.200} ...")
print(f"Rad Grads:\n{str(rad_desc_grads):.200} ...\nAng Grads:\n{str(ang_desc_grads):.200} ...")

# Saving Descriptor Info

if __name__ == '__main__':
    file_saver(descriptor_file_name, descriptor_folder, "ang", ang_descriptor)
    file_saver(descriptor_file_name, descriptor_folder, "rad", rad_descriptor)
    file_saver(descriptor_file_name, descriptor_folder, "rad_grad", rad_desc_grads)
    file_saver(descriptor_file_name, descriptor_folder, "ang_grad", ang_desc_grads)