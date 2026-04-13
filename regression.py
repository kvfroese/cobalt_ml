"""
Ridge regression for MLIP model training and evaluation.
Works with per-chunk descriptor files.
Run directly to train model, or import ridge_fit/ridge_predict for testing.
"""

import numpy as np
from pathlib import Path
import configargparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from utility import file_loader

def parser_client():
    parser = configargparse.ArgParser(
        description='For all required config arguments, see the default.ini file.',
        default_config_files=[Path('config/default.ini')]
    )

    # Arguments
    parser.add_argument('-c',
                        '--my-config',
                        is_config_file=True,
                        help="Path of custom config file")
    parser.add_argument('--descriptor_file_names',
                        type=str,
                        default='desc',
                        help="Base name of descriptor outputs")
    parser.add_argument('--descriptor_folder',
                        type=str,
                        default='descriptors',
                        help="Location of descriptor output")
    parser.add_argument('--energy_file',
                        type=str,
                        default='energies',
                        help='File containing DFT energies (eV)')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.1,
                        help="Ridge regularization hyperparameter")
    parser.add_argument('--test_frac',
                        type=float,
                        default=0.15,
                        help="Fraction of data for testing")
    parser.add_argument('--valid_frac',
                        type=float,
                        default=0.15,
                        help="Fraction of data for validation")
    parser.add_argument('--train_frac',
                        type=float,
                        default=0.7,
                        help="Fraction of data for training")
    
    args, unknown_args = parser.parse_known_args()
    return args

def ridge_fit(X, y, alpha):
    """
    Ridge regression from numpy normal equations.

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray, shape (n_samples,)
        Target values
    alpha : float
        Regularization parameter

    Returns:
    --------
    w : np.ndarray, shape (n_features + 1,)
        Coefficients including intercept
    """
    # Add intercept column
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Normal equations: (X^T X + αI) w = X^T y
    n_features = X_with_bias.shape[1]
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y

    # Add ridge penalty to diagonal (excluding intercept)
    ridge_matrix = XtX.copy()
    ridge_matrix[1:, 1:] += alpha * np.eye(n_features - 1)

    # Solve linear system
    try:
        w = np.linalg.solve(ridge_matrix, Xty)
    except np.linalg.LinAlgError:
        print('Matrix singular, using pseudo-inverse...')
        w = np.linalg.pinv(ridge_matrix) @ Xty
    
    return w

def ridge_predict(X, w):
    """
    Predict using ridge regression coefficients.

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    w : np.ndarray, shape (n_features + 1,)
        Coefficients including intercept

    Returns:
    --------
    y_pred : np.ndarray, shape (n_samples,)
        Predicted values
    """
    if w.shape[0] == X.shape[1] + 1:
        X = np.column_stack([np.ones(X.shape[0]), X])
    return X @ w

def load_descriptors_from_chunks(descriptor_folder, base_name):
    """
    Load descriptors from per-chunk files.
    
    Parameters:
    -----------
    descriptor_folder : Path
        Folder containing chunk descriptor files
    base_name : str
        Base name of descriptor files (e.g., 'desc')
    
    Returns:
    --------
    ang_descriptor : list
        Angular descriptors for all structures
    rad_descriptor : list
        Radial descriptors for all structures
    """
    folder = Path(descriptor_folder)
    
    # Find all chunk files
    ang_files = sorted(folder.glob(f'{base_name}_ang_chunk_*.pkl'))
    rad_files = sorted(folder.glob(f'{base_name}_rad_chunk_*.pkl'))
    
    if not ang_files:
        # Try alternative naming pattern
        ang_files = sorted(folder.glob(f'{base_name}_ang_chunk*.pkl'))
        rad_files = sorted(folder.glob(f'{base_name}_rad_chunk*.pkl'))
    
    if not ang_files:
        raise FileNotFoundError(f"No descriptor chunk files found in {folder}")
    
    print(f"Found {len(ang_files)} angular descriptor chunks")
    print(f"Found {len(rad_files)} radial descriptor chunks")
    
    # Load and concatenate all chunks
    ang_descriptor = []
    rad_descriptor = []
    
    for ang_file, rad_file in zip(ang_files, rad_files):
        with open(ang_file, 'rb') as f:
            ang_chunk = pickle.load(f)
        with open(rad_file, 'rb') as f:
            rad_chunk = pickle.load(f)
        
        ang_descriptor.extend(ang_chunk)
        rad_descriptor.extend(rad_chunk)
        
        print(f"  Loaded {len(ang_chunk)} structures from {ang_file.name}")
    
    print(f"Total structures loaded: {len(ang_descriptor)}")
    return ang_descriptor, rad_descriptor


def load_energies(descriptor_folder, base_name):
    """
    Load energies from the extracted energies file.
    
    Parameters:
    -----------
    descriptor_folder : Path
        Folder containing the energies file
    base_name : str
        Base name of the energies file (e.g., 'desc')
    
    Returns:
    --------
    y : np.ndarray
        Array of energy values in eV
    struct_names : list
        List of structure names in same order
    """
    folder = Path(descriptor_folder)
    
    possible_files = [
        folder / f"{base_name}_energies.pkl",
        folder / "desc_energies.pkl",
        folder / "energies.pkl",
        Path("desc_out") / f"{base_name}_energies.pkl",
        Path("desc_out") / "desc_energies.pkl",
    ]
    
    energy_data = None
    used_file = None
    
    for energy_file in possible_files:
        if energy_file.exists():
            print(f"Found energy file: {energy_file}")
            with open(energy_file, 'rb') as f:
                energy_data = pickle.load(f)
            used_file = energy_file
            break
    
    if energy_data is None:
        raise FileNotFoundError(f"No energy file found in {descriptor_folder} or desc_out/")
    
    if isinstance(energy_data, list):
        if len(energy_data) > 0 and isinstance(energy_data[0], tuple):
            struct_names = [item[0] for item in energy_data]
            y = np.array([item[1] for item in energy_data])
        else:
            struct_names = [f"struct_{i}" for i in range(len(energy_data))]
            y = np.array(energy_data)
    elif isinstance(energy_data, dict):
        struct_names = list(energy_data.keys())
        y = np.array(list(energy_data.values()))
    elif isinstance(energy_data, np.ndarray):
        y = energy_data
        struct_names = [f"struct_{i}" for i in range(len(y))]
    else:
        raise ValueError(f"Unknown energy data format: {type(energy_data)}")
    
    print(f"Loaded {len(y)} energies from {used_file}")
    return y, struct_names

def concatenate_features(ang_descriptor, rad_descriptor, struct_names_from_energies=None):
    """
    Convert descriptor lists to feature matrix X.
    
    Parameters:
    -----------
    ang_descriptor : list of tuples
        Each tuple: (structure_name, ang_G_value)
    rad_descriptor : list of tuples
        Each tuple: (structure_name, rad_G_value)
    struct_names_from_energies : list, optional
        Structure names from energies file to filter by
    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (each row is [rad_G, ang_G])
    labels : list
        Structure names in same order
    """
    # Convert to dictionaries keyed by structure name
    ang_dict = {item[0]: item[1] for item in ang_descriptor}
    rad_dict = {item[0]: item[1] for item in rad_descriptor}
    
    # Find common structures (should be all of them)
    common_structs = sorted(set(ang_dict.keys()) & set(rad_dict.keys()))

    if struct_names_from_energies is not None:
        energy_set = set(struct_names_from_energies)
        common_structs = [s for s in common_structs if s in energy_set]


    if not common_structs:
        raise ValueError("No matching structures between ang and rad descriptors")
    
    features_list = []
    labels = []
    
    for struct in common_structs:
        rad_val = rad_dict[struct]
        ang_val = ang_dict[struct]
        
        # Ensure they are scalars or 1D arrays
        rad_val = np.array(rad_val).flatten()
        ang_val = np.array(ang_val).flatten()
        
        # Concatenate radial and angular descriptors
        concatenated = np.concatenate([rad_val, ang_val])
        features_list.append(concatenated)
        labels.append(struct)
    
    X = np.vstack(features_list)
    return X, labels

# ============================================================
# Main execution - only runs when script is called directly
# ============================================================
if __name__ == "__main__":
    parser = parser_client()

    descriptor_file_names = parser.descriptor_file_names
    descriptor_folder = parser.descriptor_folder

    alpha = parser.alpha
    train_frac = parser.train_frac
    valid_frac = parser.valid_frac
    test_frac = parser.test_frac

    print('=' * 60)
    print('Loading descriptors from chunks...')
    print('=' * 60)
    
    # Load descriptors from chunk files
    ang_descriptor, rad_descriptor = load_descriptors_from_chunks(
        descriptor_folder, descriptor_file_names
    )

    # Load energies
    try:
        y, energy_struct_names = load_energies(descriptor_folder, descriptor_file_names)
        print(f'Loaded {len(y)} DFT energies')
    except Exception as e:
        print(f'ERROR: could not load energy file: {e}')
        print('Creating dummy energies for testing...')
        y = np.random.randn(len(ang_descriptor)) * 0.1
        energy_struct_names = [f"struct_{i}" for i in range(len(y))]

    print('Building feature matrix...')
    X, descriptor_labels = concatenate_features(ang_descriptor, rad_descriptor)
    print(f'Feature matrix shape: {X.shape}')
    print(f'Number of descriptors: {X.shape[1]}')

    # Align y with X order
    # Create a dictionary mapping structure name to energy
    energy_dict = dict(zip(energy_struct_names, y))
    y_aligned = np.array([energy_dict[name] for name in descriptor_labels])

    print(f'\nSplitting data: train_frac={train_frac:.1%}, valid_frac={valid_frac:.1%}, test_frac={test_frac:.1%}')

    # Split 1: test from train & valid
    X_temp, X_test, y_temp, y_test, names_temp, names_test = train_test_split(
        X, y_aligned, descriptor_labels, test_size=test_frac, random_state=42)

    # Split 2: train from valid
    val_frac_rel = valid_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
        X_temp, y_temp, names_temp, test_size=val_frac_rel, random_state=42)

    print(f'Training set: {X_train.shape[0]} samples')
    print(f'Validation set: {X_val.shape[0]} samples')
    print(f'Test set: {X_test.shape[0]} samples')

    # Tune hyperparameters with validation set
    alpha_candidates = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 
                        1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_val_mae = float('inf')
    best_w = None

    print('\n' + '=' * 60)
    print('Tuning alpha (ridge regularization)...')
    print('=' * 60)
    
    for a in alpha_candidates:
        w_temp = ridge_fit(X_train, y_train, a)
        y_val_pred = ridge_predict(X_val, w_temp)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        print(f'  alpha = {a:.1e}: validation MAE = {val_mae:.6f} eV')
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = a
            best_w = w_temp

    print(f'\nBest alpha: {best_alpha:.1e} with validation MAE = {best_val_mae:.6f} eV')

    # Learning curve generation
    print('\n' + '=' * 60)
    print('Generating learning curve...')
    print('=' * 60)
    
    train_size_fractions = np.linspace(0.1, 1.0, 10)
    train_errors = []
    val_errors = []
    actual_train_sizes = []

    for frac in train_size_fractions:
        n_samples = max(1, int(frac * len(X_train)))
        actual_train_sizes.append(n_samples)
        
        X_train_sub = X_train[:n_samples]
        y_train_sub = y_train[:n_samples]
        
        w_sub = ridge_fit(X_train_sub, y_train_sub, best_alpha)
        
        y_train_pred_sub = ridge_predict(X_train_sub, w_sub)
        y_val_pred_sub = ridge_predict(X_val, w_sub)
        
        train_mae_sub = mean_absolute_error(y_train_sub, y_train_pred_sub)
        val_mae_sub = mean_absolute_error(y_val, y_val_pred_sub)
        
        train_errors.append(train_mae_sub)
        val_errors.append(val_mae_sub)
        
        print(f'  {n_samples} samples: Train MAE = {train_mae_sub:.6f} eV, Val MAE = {val_mae_sub:.6f} eV')

    # Final training with best alpha (train + val combined)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    names_train_val = names_train + names_val

    print(f'\nFinal training set size: {X_train_val.shape[0]} samples')
    w_final = ridge_fit(X_train_val, y_train_val, best_alpha)

    # Predictions
    y_train_pred = ridge_predict(X_train, w_final)
    y_val_pred = ridge_predict(X_val, w_final)
    y_test_pred = ridge_predict(X_test, w_final)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae_final = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2_final = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print('\n' + '=' * 60)
    print('Final Results')
    print('=' * 60)
    print(f'  Training MAE: {train_mae:.6f} eV')
    print(f'  Training R²:  {train_r2:.6f}')
    print(f'  Validation MAE: {val_mae_final:.6f} eV')
    print(f'  Validation R²:  {val_r2_final:.6f}')
    print(f'  Test MAE: {test_mae:.6f} eV')
    print(f'  Test R²:  {test_r2:.6f}')

    results = {
        'coefficients': w_final,
        'intercept': w_final[0],
        'feature_weights': w_final[1:],
        'best_alpha': best_alpha,
        'train_mae': train_mae,
        'val_mae': val_mae_final,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'val_r2': val_r2_final,
        'test_r2': test_r2,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'labels_train': names_train,
        'labels_val': names_val,
        'labels_test': names_test,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'train_sizes': actual_train_sizes,
        'train_errors': train_errors,
        'val_errors': val_errors,
    }

    output_file = Path(descriptor_folder) / f'{descriptor_file_names}_ridge_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nResults saved to {output_file}')