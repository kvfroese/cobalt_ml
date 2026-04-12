import torch
import numpy as np
from pathlib import Path
import configargparse
import pickle

from utility import path_handler, file_saver, file_loader

def parser_client():
    parser = configargparse.ArgParser(
        description='For all required config arguments, see the default.ini file. You can specify a different config file with --config_file or -c, or override any argument with --argname value.',
        default_config_files=[Path('config/default.ini')]
    )

    # Arguments
    ## File Stuff
    parser.add_argument('-c',
                        '--my-config',
                        is_config_file=True,
                        help="Path of custom config file")
    parser.add_argument('--descriptor_file_names',
                        type=str,
                        help="Base name of descriptor outputs")
    parser.add_argument('--descriptor_folder',
                        type=str,
                        help="Location of descriptor output. Recoomend do not change")
    
    ## Parameters
    parser.add_argument('--alpha',
                        type=float,
                        help="alpha is a regularization hyperparameter")


    
    args, unknown_args = parser.parse_known_args()
    return args

parser = parser_client()

descriptor_file_name = parser.descriptor_file_names
descriptor_folder = parser.descriptor_folder

alpha = parser.alpha

ang_descriptor = file_loader(descriptor_file_name, descriptor_folder, "ang")
rad_descriptor = file_loader(descriptor_file_name, descriptor_folder, "rad")

'''
The radial descriptor has tuple form (struct, ang_G, theta_deriv, r_ij_deriv, r_ik_deriv, r_jk_deriv)
The angular descriptor has tuple form (struct, rad_G, rad_G_deriv)
We are abandoning the labels for math, of course, ergo we will end up with
N x 7
[rad_G, rad_G_deriv, ang_G, theta_deriv, r_ij_deriv, r_ik_deriv, r_jk_deriv] (for each structure)
'''
def concatenate_features(ang_descriptor, rad_descriptor):
    # TODO adjust with projected grads
    # Convert list-of-tuples descriptors to dicts keyed by file name
    def descriptor_to_dict(descriptor):
        return {item[0]: np.array(item[1:], dtype=float) for item in descriptor}

    ang_dict = descriptor_to_dict(ang_descriptor)
    rad_dict = descriptor_to_dict(rad_descriptor)

    common_structs = sorted(set(ang_dict.keys()) & set(rad_dict.keys()))
    if not common_structs:
        raise ValueError("No matching structures between ang and rad descriptors")

    features_list = []
    labels = []
    for struct in common_structs:
        ang_feat = ang_dict[struct]
        rad_feat = rad_dict[struct]
        concatenated = np.concatenate([rad_feat, ang_feat])
        features_list.append(concatenated)
        labels.append(struct)

    X = np.vstack(features_list)
    return X, labels

X, labels = concatenate_features(ang_descriptor, rad_descriptor)
print(X.shape)

print(X, labels)

def ridge_regression(X, y, alpha):
    # Stuff here
    None