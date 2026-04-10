"""
Generate DFT configuration inputs from provided starting geometry for HCo(CO)4
Run with: python generate_data_v1.py
"""
import os
import sys
import numpy as np
from ase import Atoms
from ase.io import write, read
import subprocess
import time
import argparse

## reference geometry is HCo(CO)4 from Veillard et. Al (1990)

ax_co_len = 1.764   #Co <-> axial C bond len
eq_co_len = 1.764   #Co <-> equitorial C bond len
co_len    = 1.15    #C  <-> O bond len
h_len     = 1.556   #Co <-> H bond len

#order: Co, H, C_ax, O_ax, C_eq1, O_eq1, C_eq2, O_eq2, C_eq3, O_eq3, C_eq4, O_eq4 
symbols = ['Co', 'H', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O']

#coordinates (Angstroms):
coords = np.zeros((10,3))                   #0: Co at origin
coords[1] = [0.0, 0.0, h_bond]              #1: H along +z
coords[2] = [0.0, 0.0, -ax_co_len]          #2: C_ax along -z
coords[3] = [0.0, 0.0, -ax_co_len - co_len] #3: O_ax along -z

angles = [0, 120, 240] #eq CO groups in xy-plane at 120° intervals
for i, ang in enumerate(angles):
    rad = np.radians(ang)
    coords[4+2*i] = [eq_co_len * np.cos(rad), eq_co_len * np.sin(rad), 0.0]
    direction = np.array([np.cos(rad), np.sin(rad), 0.0])
    coords[5+2*i] = coords[4+2*i] + direction * co_len

#crease reference Atoms object
ref_atoms = Atoms(symbols=symbols, positions=coords)

def gen_distortions(ref_atoms, n_configs = 200, pert_scale = 0.05, stretch_range=(0.9, 1.1), angle_std=np.radians(5)):
    """
    Generates distorted geometries for HCo(CO)4 reference molecule, applies
    random displacement via noise to atomic pos & random bond stretches
    
    Parms:
    ref_atoms       : ASE Atoms object
    n_configs       : int
    pert_scale      : float
    stretch_range   : tuple
    angle_std       : float

    Returns list of ASE Atoms objects of n_configs length
    """
    dists = []
    pos_ref = ref_atoms.positions.copy()
    for i in range(n_configs):
        atoms = ref_atoms.copy()
        #random displacement to all atoms
        d = np.random.normal(0, pert_scale, atoms.positions.shape)
        atoms.positions += d

        #random stretch to Co-H
        h_idx = 1
        co_idx = 0
        vec_coh = atoms.positions[h_idx] - atoms.positions[co_idx]
        stretch_coh = np.random.uniform(*stretch_range)
        atoms.positions[h_idx] = atoms.positions[co_idx] + vec_coh * stretch_coh

        #random stretch to Co-C
        for c_idx in [2,4,6,8]:
            vec_coc = atoms.positions[c_idx] - atoms.positions[co_idx]
            stretch_coc = np.random.uniform(*stretch_range)
            atoms.positions[h_idx] = atoms.positions[co_idx] + vec_coh * stretch_coh
        dists.append(atoms)
    return dists

def write_orca(atoms_list, prefix='hco_co4', output_dir='orca_ins'):
    """
    Writes ORCA input files for all atoms in atoms_list

    naming scheme is {prefix}_{idx:04d}.inp, all written to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    template = """! B3LYP def2-SVP D3BJ Opt Grid4 FinalGrid5 Engrad NormalPrint
    %maxcore 2000
    %pal nprocs 4 end
    * xyz 0 1
    <REPLACE>
    *
    """
    input_files = []
    for idx, atoms in enumerate(atoms_list):
        filename = f'{prefix}_{idx:04d}.inp'
        filepath = os.path.join(output_dir, filename)

        xyz_block = f'{len(atoms)}\n'
        xyz_block += f' {prefix}_{idx}\n'
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
            xyz_block += f' {sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n'
        inp_content = template.replace('<REPLACE>', xyz_block)
        with open(filepath, 'w') as f:
            f.write(inp_content)
        input_files.append(filepath)
    return input_files

def write_slurm(input_dir='orca_ins', script_name='run_cobalt.slurm', time='24:00:00', mem='4G', ntasks=1, cpus_per_task=4):
    """
    Writes a SLURM script that loops ofer all .inp files in input_dir and submits each as seperate jobs
    """
    inp_files = sorted(glob.glob(os.path.join(input_dir, '*.inp')))
    n_jobs = len(inp_files)
    if n_jobs == 0:
        print('No .inp files found in ', input_dir)
        return None
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=orca_hco_co4
#SBATCH --output=orca_%A_%a.out
#SBATCH --error=orca_%A_%a.err
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --array=1-{n_jobs}

# Load ORCA module
module purge
module load ORCA/5.0.4

# Get inp for this array task
INPUT_FILE=$(ls {input_dir}/*.inp | sed -n ${{SLURM_ARRAY_TASK_ID}}p)
INPUT_DIR=$(dirname "$INPUT_FILE")
BASENAME=$(basename "$INPUT_FILE" .inp)
cd "$INPUT_DIR"

# Run ORCA
orca $BASENAME.inp > $BASENAME.out

echo "Finished $BASENAME"
"""
    script_path = os.path.join(os.getcwd(), script_name)
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    os.chmod(script_path, 0o755)
    print(f'SLURM script written to {script_path} (run: sbatch {script_path})')
    return script_path

def parse_orca_out(outfile, ha_ev_conv=27.2113862459988, ha_bohr_ev_a_conv=51.422086190832):
    """
    Parses an ORCA .out file and the corresponding .engrad file to extract 
    total energy (Ha) and atomic forces (Ha/Bohr)

    Returns (energy, forces)
    """
    engrad_file = outfile.replace('.out', '.engrad')
    if not os.path.exists(engrad_file):
        raise FileNotFoundError(f'Could not find {engrad_file}')

    with open(engrad_file, 'r') as f:
        lines = f.readlines()
    
    energy_ha = float(lines[1].strip()) #on line 2 of .engrad file
    forces_lines = []
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '' and i > 2:
            start_idx = i+1
            break
    if start_idx is None:
        raise ValueError('Could not find forces in .engrad file')
    n_atoms = (len(lines) - start_idx) // 3
    forces_ha_bohr = np.zeros((n_atoms, 3))
    for a in range(n_atoms):
        fx = float(lines[start_idx + 3*a].strip())
        fy = float(lines[start_idx + 3*a + 1].strip())
        fz = float(lines[start_idx + 3*a + 2].strip())
        forces_ha_bohr[a] = [fx, fy, fz]

        #Ha      -> eV:   1 Ha = 27.211386245988 eV
        #Ha/Bohr -> eV/A: 1 Ha/Bhor = 51.422086190832 eV/A
        energy_ev = energy_ha * ha_ev_conv        
        forces_ev_a = forces_Ha_bohr * ha_bohr_ev_a_conv

        return energy_ev, forces_ev_a

def collect_results(output_dir='.', prefix='hco_co4'):
    """
    Scans for all ORCA output files and collect energies and forces
    Returns a dict with keys: posns, energies, forces, atomic_nums
    """
    out_fles = sorted(glob.glob(os.path.join(output_dir, f'{prefix}_*.out')))
    if not out_files:
        print('No output files found')
        return None
    all_energies = []
    all_forces   = []
    all_posn     = []
    atomic_nums  = None
    for outfile in out_files:
        try:
            energy, forces = parse_orca_output(outfile)
            all_energies.append(energy)
            all_forces.append(forces)
            inpfile = outfile.replace('.out', '.inp')

            with open(inpfile, 'r') as f:
                lines = f.readlines()
            xyz_start = None
            for i, line in enumerate(lines):
                if '* xyz 0 1' in line:
                    xyz_start = i+1
                    break
            if xyz_start is None:
                raise ValueError('Could not find XYZ block in inp file')
                n_atoms = int(lines[xyz_start].strip())
                posn = []
                for a in range(n_atoms):
                    parts = lines[xyz_start+1+a].split()
                    posn.append([float(parts[1]),
                                float(parts[2]),
                                float(parts[3])])
                all_posn.append(np.array(posn))
                if atomic_nums is None:
                    from ase.data import atomic_nums as ase_atomic_nums
                    syms_first = [parts[0] for parts in [lines[xyz_start+1+a.split]
                                  for a in range(n_atoms)]]
                    atomic_nums = np.array([ase_atomic_nums[sym]] for sym in syms_first)
        except Exception as e:
            print(f'Failed to parse {outfile}:.{e}')
            continue

    all_energies = np.array(all_energies)
    all_forces   = np.array(all_forces)
    all_posn     = np.array(all_posn)
    return {
        'positions': all_posn,
        'energies': all_energies,
        'forces': all_forces,
        'atomic numbers': atomic_nums
    }

# --- Main execution ---
def main():
    parser = argparse.ArgumentParser(description='Generate HCo(CO)4 DFT data using ORCA')
    parser.add_argument('--nconfigs', type=int, default=200, help='Number of distorted configurations to generate')
    parser.add_argument('--perturb', type=float, default=0.05, help='Positional perturbation scale (Å)')
    parser.add_argument('--stretch_min', type=float, default=0.9, help='Minimum bond stretch factor')
    parser.add_argument('--stretch_max', type=float, default=1.1, help='Maximum bond stretch factor')
    parser.add_argument('--write_ins', action='store_true', help='Only write input files')
    parser.add_argument('--collect', action='store_true', help='Collect results from existing ORCA outputs')
    args = parser.parse_args()

    if args.collect:
        print('Collecting results from ORCA outputs...')
        data = collect_results()
        if data is not None:
            filename = 'hco_co4_dataset.npz'
            np.savez(filename, **data)
            print(f'Saved dataset to {filename} with {len(data['energies'])} configurations')
        return
    
    print('Generating distorted configurations...')
    dists = gen_distortions(ref_atoms, n_configs=args.nconfigs,
                                       pert_scale=args.perturb,
                                       stretch_range=(args.stretch_min, args.stretch_max))
    print(f'Generated {len(distortions)} configurations')

    print('Writing ORCA input files...')
    input_files = write_orca(dists, prefix='hco_co4', output_dir='orca_ins')
    print(f'Wrote {len(input_files)} input files to orca_inputs/')
    
    if args.write_inputs:
        print('Input files written. Exiting.')
        return

    print('Writing slurm script...')
    write_slurm(input_dir='orca_inputs', script_name='run_cobalt.slurm', time='24:00:00', mem='4G', ntasks=1, cpus_per_task=4)
    print(f'Slurm written to orca_inputs/')
    print('After jobs complete, rerun this script with the --collect flag to parse results')

    if __name__ == "__main__":
        main()