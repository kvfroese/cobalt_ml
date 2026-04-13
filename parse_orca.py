from pathlib import Path
import numpy as np
import itertools
import configargparse
import pickle
import random
import gc
from tqdm import tqdm

from utility import path_handler, file_saver

## Parser Setup

def parser_client():
    parser = configargparse.ArgParser(
        description='For all required config arguments, see the default.ini file. You can specify a different config file with --config_file, or override any argument with --argname value.',
        default_config_files=[Path('config/default.ini')]
    )

    # Arguments
    parser.add_argument('-c',
                        '--my-config',
                        is_config_file=True,
                        help="Path of custom config file")
    parser.add_argument('--input_file_names',
                        type=str,
                        help="Base name of ORCA files that contain geometry and energy values")
    parser.add_argument('--input_folder',
                        type=str,
                        help="Folder path to all ORCA files. Recommend do not change")
    parser.add_argument('--output_file_names',
                        type=str,
                        help="Base name of descriptor outputs")
    parser.add_argument('--output_folder',
                        type=str,
                        help="Location of descriptor output. Recoomend do not change")
    parser.add_argument('--chunk_size',
                        type=int,
                        default=50,
                        help='Number of structures to process before saving checkpoint')
    parser.add_argument('--max_structures',
                        type=int,
                        default=1000,
                        help='Maximum structures sampled for MLIP')
    parser.add_argument('--skip_parsing',
                        action='store_true',
                        help="Skip parsing phase and use saved checkpoint")
    
    args, unknown_args = parser.parse_known_args()
    return args

parser = parser_client()

input_file_names = parser.input_file_names
input_folder = Path(parser.input_folder)
output_file_names = parser.output_file_names
output_folder = Path(parser.output_folder)
chunk_size = parser.chunk_size
max_structures = parser.max_structures
skip_parsing = parser.skip_parsing

parsed_checkpoint = output_folder / f'{output_file_names}_parsed_checkpoint.pkl'
geo_checkpoint = output_folder / f'{output_file_names}_geo_checkpoint.pkl'

## Parsing
def open_file(file_path):
    with open(Path(file_path), 'r') as f:
        lines = f.readlines()
    return lines

def extract_energy(lines, filename, ha_to_ev=27.211386245988):
    """
    extracts FINAL SINGLE POINT ENERGY
    """
    for line in lines:
        if 'FINAL SINGLE POINT ENERGY' in line:
            parts = line.split()
            try:
                energy_ha = float(parts[4])
                energy_ev = energy_ha * ha_to_ev
                return energy_ev
            except (IndexError, ValueError) as e:
                print(f'Could not parse energy for {filename}')
                return None
    print(f'Could not find final SPE for {filename}')
    return None

def extract_coords(lines, filename):
    """
    extract coords from ORCA *.out
    """
    start_line = None

    for i, line in enumerate(lines):
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            start_line = i + 2
            break

    if start_line is None:
        print(f'Could not find coordinates for {filename}')
        return None

    coordinates = []
    id_counter = 0

    for j in range(start_line, len(lines)):
        line = lines[j].strip()
        if not line or line.startswith('-') or line.startswith('---'):
            break
        
        parts = line.split()
        if len(parts) >= 4:
            try:
                atom_sym = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                coordinates.append((id_counter, atom_sym, x, y, z))
                id_counter += 1
            except ValueError:
                continue
    if len(coordinates) == 0:
        return None
    return tuple(coordinates)

if skip_parsing and parsed_checkpoint.exists():
    print(f'Found existing parsed checkpoint: {parsed_checkpoint}')

    with open(parsed_checkpoint, 'rb') as f:
        checkpoint_data = pickle.load(f)

    atom_files = checkpoint_data['atom_files']
    energy_files = checkpoint_data['energy_files']
    failed_files = checkpoint_data['failed_files']

    print(f"   Loaded {len(atom_files)} successfully parsed files")
    print(f"   Loaded {len(failed_files)} failed files")
else:
    out_files = list(input_folder.glob('*.out'))

    if len(out_files) == 0:
        raise ValueError(f'No .out files found in {input_folder}')
    print(f'Found {len(out_files)} .out files in {input_folder}')
    print('=' * 60)

    atom_files   = []   # (filename, atom_info, energy_ev)
    energy_files = []   # (filename, energy_ev)
    failed_files = []   # (filename, reason)

    for file in tqdm(out_files, desc='Parsing ORCA files', unit='file'):
        filename = file.name
        try:
            lines = open_file(file)
            energy_ev = extract_energy(lines, filename)
            if energy_ev is None:
                failed_files.append((filename, 'Missing SPE'))
                continue
            atom_info = extract_coords(lines, filename)
            if atom_info is None:
                failed_files.append((filename, 'Missing coords'))
                continue
            
            atom_files.append((filename, atom_info, energy_ev))
            energy_files.append((filename, energy_ev))

        except Exception as e:
            print(f'ERROR: {e}')
            failed_files.append((filename, str(e)))

    print("\nSaving parsing checkpoint...")
    checkpoint_data = {
        'atom_files': atom_files,
        'energy_files': energy_files,
        'failed_files': failed_files,
        'input_folder': str(input_folder),
        'total_files': len(out_files)
    }
    with open(parsed_checkpoint, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Saved to {parsed_checkpoint}")

    print("\n" + "=" * 60)
    print(f"Success: {len(atom_files)} files")
    print(f"Failed:  {len(failed_files)} files")

if len(atom_files) == 0:
    raise ValueError('No files successfully processed')

if max_structures is not None and max_structures > 0 and max_structures < len(atom_files):
    print(f"\nRandomly sampling {max_structures} structures from {len(atom_files)} total")
    random.seed(42)
    indices = random.sample(range(len(atom_files)), max_structures)
    atom_files = [atom_files[i] for i in indices]
    energy_files = [energy_files[i] for i in indices]
    print(f"   Sampled {len(atom_files)} structures")

print(f"\nCreating chunks of {chunk_size} structures each...")
print("=" * 60)

# Split into chunks
chunks = []
for i in range(0, len(atom_files), chunk_size):
    chunks.append((atom_files[i:i+chunk_size], energy_files[i:i+chunk_size]))

print(f"Created {len(chunks)} chunks")

# Save each chunk as a separate file
for idx, (atom_chunk, energy_chunk) in enumerate(chunks):
    chunk_data = {
        'atom_files': atom_chunk,
        'energy_files': energy_chunk,
        'chunk_id': idx + 1,
        'total_chunks': len(chunks),
        'chunk_size': len(atom_chunk)
    }
    chunk_file = output_folder / f"{output_file_names}_chunk_{idx+1}.pkl"
    with open(chunk_file, 'wb') as f:
        pickle.dump(chunk_data, f)
    print(f"Saved chunk {idx+1}: {len(atom_chunk)} structures to {chunk_file.name}")

