"""
Generate and save figures from other scripts
Run with: python generate_figures.py
Run after mlip.py produces trained results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

#set figure style
plt.style.use('style.mplstyle')

# ===================================================================
# Load training results
# ===================================================================
results = np.load('temp.npz')

forces_true = results['forces_test']
forces_pred = results['forces_pred']
energies_true = results['energies_test']
energies_pred = results['energies_pred']
coefficients = results['coefficients']
descriptor_labels = results['descriptor_labels']

# ===================================================================
# Fig 1: Force correlation plot
# ===================================================================
fig1, ax1 = plt.subplots(figsize=(7, 7))

f_true_flat = forces_true.flatten()
f_pred_flat = forces_pred.flatten()

ax1.scatter(f_true_flat, f_pred_flat, s=10, alpha=0.5, edgecolors='none', c='steelblue')
min_val = min(f_true_flat.min(), f_pred_flat.min())
max_val = max(f_true_flat.max(), f_pred_flat.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

r2 = r2_score(f_true_flat, f_pred_flat)
ax1.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=ax1.transAxes, 
         fontsize=14, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_xlabel(r'DFT Forces (eV/\r{A})')
ax1.set_ylabel(r'MLIP Predicted Forces (eV/\r{A})')
ax1.set_title(r'Force Correlation')
ax1.legend(loc='lower right')

figname = 'force_correlation'
plt.tight_layout()
plt.savefig(figname)
print(f'Saved: {figname}.pdf')

# ===================================================================
# Fig 2: Energy MAE comparison
# ===================================================================
fig2, ax2 = plt.subplots(figsize=(6, 5))

mae_train = results.get('mae_train', 0.05)
mae_test  = results.get('mae_test', 0.05)
labels    = ['Training', 'Test']
maes      = [mae_train, mae_test]
colors    = ['forestgreen', 'coral']

bars = ax2.bar(labels, maes, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel(r'Energy MAE (eV)')
ax2.set_title(r'Energy Prediction Error')

for bar, val in zip(bars, maes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=12)

figname = 'mae_comparison'
plt.tight_layout()
plt.savefig(figname)
print(f'Saved: {figname}.pdf')

# ===================================================================
# Fig 3: Top n Descriptor Coefficients
# ===================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))
n_top = 10  #TEMP: amount of coefficients recognized

top_idxs = np.argsort(np.abs(coefficients))[-n_top:][::-1]
top_labels = [descriptor_labels[i] for i in top_idxs]
top_coeffs = coefficients[top_idxs]
colors_coeff = ['darkred' if c < 0 else 'darkblue' for c in top_coeffs]

ax3.barh(range(len(top_labels)), top_coeffs, color=colors_coeff, edgecolor='black', linewidth=0.8)
ax3.set_yticks(range(len(top_labels)))
ax3.set_yticklabels(top_labels, fontsize=10)
ax3.set_xlabel(r'Coefficient Value')
ax3.set_title(rf'Top {n_top} Most Impactful Descriptors')
ax3.axvline(x=0, color='black', linewidth=1, linestyle='-')

figname = 'top_coefficients'
plt.tight_layout()
plt.savefig(figname)
print(f'Saved: {figname}.pdf')

# ===================================================================
# Fig 4: Learning Curves
# ===================================================================
fig4, ax4 = plt.subplots(figsize=(8, 6))

train_sizes = results['train_sizes']
train_errs = results['train_errs']
val_errs = results['val_errs']

ax4.plot(train_sizes, train_errs, 'o-', label='Training', color='black', linewidth=2)
ax4.plot(train_sizes, val_errs, 's-', label='Validation', color='red', linewidth=2)
ax4.set_xlabel(r'Number of Training Configurations')
ax4.set_ylabel(r'Force MAE (eV/\r{A})')
ax4.set_title(r'Learning Curves')
ax4.legend()

figname = 'learning_curves'
plt.tight_layout()
plt.savefig(figname)
print(f'Saved: {figname}.pdf')

# ===================================================================
# Fig 5: Hyperparameter Heatmap (TEMP: hopefully possible it would be cool i think)
# ===================================================================
if 'hyperparm_grid' in results and 'hyperparm_scores_2d' in results:
    fig5, ax5 = plt.subplots(figsize=(7, 7))

    grid = results['hyperparm_grid']
    im = ax5.imshow(results['hyperparm_scores'], aspect='auto', origin='lower')

    ax5.set_xticks(range(len(grid['eta_values'])))
    ax5.set_xticklabels([f'{e:.1f}' for e in grid['eta_values']])
    ax5.set_yticks(range(len(grid['Rc_values'])))
    ax5.set_yticklabels([f'{r:.1f}' for r in grid['Rc_values']])
    ax5.set_xlabel(r'$\eta$ (\r{A}$^{-2}$)')
    ax5.set_ylabel(r'$R_c$ (\r{A})')
    ax5.set_title(r'Validation Force MAE vs. Hyperparameters')
    plt.colorbar(im, label='Force MAE (eV/Å)')

    figname = 'hyperparm_heatmap'
    plt.tight_layout()
    plt.savefig(figname)
    print(f'Saved: {figname}.pdf')

print('\nAll figures successfully generated!')