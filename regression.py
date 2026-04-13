import torch
import numpy as np
from pathlib import Path
import configargparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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
    parser.add_argument('--energy_file',
                        type=str,
                        help='File containing DFT energies (eV)')
    
    ## Parameters
    parser.add_argument('--alpha',
                        type=float,
                        default=0.1,
                        help="alpha is a regularization hyperparameter")
    parser.add_argument('--test_frac',
                        type=float,
                        default=0.15,
                        help="fraction of data for testing")
    parser.add_argument('--valid_frac',
                        type=float,
                        default=0.15,
                        help="fraction of data for validation")
    parser.add_argument('--train_frac',
                        type=float,
                        default=0.7,
                        help="fraction of data for training")
    
    args, unknown_args = parser.parse_known_args()
    return args

parser = parser_client()

descriptor_file_name = parser.descriptor_file_names
descriptor_folder = parser.descriptor_folder

alpha = parser.alpha
train_frac = parser.train_frac
valid_frac = parser.valid_frac
test_frac = parser.test_frac

print('Loading descriptors...')
ang_descriptor = file_loader(descriptor_file_name, descriptor_folder, "ang")
rad_descriptor = file_loader(descriptor_file_name, descriptor_folder, "rad")

'''
The radial descriptor has tuple form (struct, ang_G, theta_deriv, r_ij_deriv, r_ik_deriv, r_jk_deriv)
The angular descriptor has tuple form (struct, rad_G, rad_G_deriv)
We are abandoning the labels for math, of course, ergo we will end up with
N x 7
[rad_G, rad_G_deriv, ang_G, theta_deriv, r_ij_deriv, r_ik_deriv, r_jk_deriv] (for each structure)
'''
try:
    y = file_loader(descriptor_file_name, descriptor_folder, 'energies')
    print(f'Loaded {len(y)} DFT energies')
except:
    print('ERROR: could not load energy file. Creating dummy for testing...')
    y = np.random.randn(len(ang_descriptor)) * 0.1
    
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

        ang_feat = np.array(ang_feat).flatten()
        rad_feat = np.array(rad_feat).flatten()

        concatenated = np.concatenate([rad_feat, ang_feat])
        features_list.append(concatenated)
        labels.append(struct)

    X = np.vstack(features_list)
    return X, labels

print('Building feature matrix...')
X, labels = concatenate_features(ang_descriptor, rad_descriptor)
print(f'Feature matrix shape: {X.shape}')

if len(y) != len(labels):
    print(f'Warning: {len(y)} energies but {len(labels)} structures')

def ridge_fit(X, y, alpha):
    """
    Ridge regression from numpy normal equations

    parms:
    ----------
    X: np.ndarray w/ shape (n_samples, n_features)
    y: np.ndarray w/ shape (n_samples,)
    alpha: float

    returns:
    ----------
    w: np.ndarray w/ shape (n_features)
    """
    #ensure intercept terms are present (column of ones)
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
       
    #normal equations: (X^T X + aI) w = X^T y
    n_features = X_with_bias.shape[1]
    xtx = X_with_bias.T @ X_with_bias
    xty = X_with_bias.T @ y

    #adding ridge penalty to diagonal (excluding intercept)
    ridge_matrix = xtx.copy()
    ridge_matrix[1:, 1:] += alpha * np.eye(n_features - 1) 

    #solve linear system
    try:
        w = np.linalg.solve(ridge_matrix, xty)
    except np.linalg.LinAlgError:
        print('Matrix singular, using pseudo-inverse to compute w...')
        w = np.linalg.pinv(ridge_matrix) @ xty
    
    return w

def ridge_predict(X, w):
    """
    Predicts using trained ridge regression coefficients

    Parms:
    ----------
    X: np.ndarray w/ shape (n_samples, n_features)
    w: np.ndarray w/ shape (n_features + 1,)

    Returns:
    ----------
    y_pred: np.ndarray w/ shape (n_samples,)
    """
    if w.shape[0] == X.shape[1] + 1:
        X = np.column_stack([np.ones(X.shape[0]), X])
    
    return X @ w

#train model
print(f'\nSplitting data: \ntrain_frac= {train_frac:.1%} \nvalid_frac= {valid_frac:.1%} \ntest_frac= {test_frac:.1%}')

#split 1: test from train & valid
X_temp, X_test, y_temp, y_test, names_temp, names_test = train_test_split(
    X, y, labels, test_size=test_frac, random_state=42)

#split 2: train from valid
val_frac_rel = valid_frac / (1.0 - test_frac)
X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
    X_temp, y_temp, names_temp, 
    test_size=val_frac_rel, 
    random_state=42
)

w = ridge_fit(X_train, y_train, alpha)
print(f'Training set: {X_train.shape[0]} samples')
print(f'Validation set: {X_val.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')

#tune hyperparameters w/ validation set
alpha_candidates = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
best_alpha = None
best_val_mae = float('inf')
best_w = None

for a in alpha_candidates:
    w_temp = ridge_fit(X_train, y_train, a)
    y_val_pred = ridge_predict(X_val, w_temp)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f'alpha = {a:.1e}: validation MAE = {val_mae:.6f} eV')

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_a = a
        best_w = w_temp
print(f'\nBest alpha: {best_a:.1e} with validation MAE = {best_val_mae:.6f} eV')


#final training w/ best alpha (train + val combined)
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])
names_train_val = names_train + names_val

print(f'Final training set size: {X_train_val.shape[0]} samples')

w_final = ridge_fit(X_train_val, y_train_val, best_a)
print(f'Number of coefficients: {len(w_final)}')
print(f'Intercept: {w_final[0]:.6f}')
print(f'Coefficient range: [{w_final[1:].min():.6f}, {w_final[1:].max():.6f}]')

#evaluate training
y_test_pred = ridge_predict(X_test, w_final)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Test MAE: {test_mae:.6f} eV')
print(f'Test R²:  {test_r2:.6f}')

#evaluate on training set for comparison
y_train_pred = ridge_predict(X_train, w_final)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f'\nTraining MAE: {test_mae:.6f} eV')
print(f'Training R²:  {test_r2:.6f}')

# Validation set performance with final model
y_val_pred = ridge_predict(X_val, w_final)
val_mae_final = mean_absolute_error(y_val, y_val_pred)
val_r2_final = r2_score(y_val, y_val_pred)

print(f'\nValidation MAE (final model): {val_mae_final:.6f} eV')
print(f'Validation R² (final model):  {val_r2_final:.6f}')

#save model and results
results = {
    #parameters
    'coefficients': w,
    'intercept': w[0],
    'feature_weights': w[1:],
    'best_a': best_a,

    #train/val/test split
    'train_mae': train_mae,
    'val_mae': val_mae_final,
    'test_mae': test_mae,
    'train_r2': train_r2,
    'val_r2': val_r2_final,
    'test_r2': test_r2,


    #predictions
    'y_train_pred': y_train_pred,
    'y_val_pred': y_val_pred,
    'y_test_pred': y_test_pred,

    #true values
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,

    #structure names
    'labels_train': names_train,
    'labels_val': names_val,
    'labels_test': names_test,

    #coefficients
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test
}

output_file = Path(descriptor_folder) / f'{descriptor_file_name}_ridge_results.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(results, f)
print(f'\nResults saved to {output_file}')
