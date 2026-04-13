"""
Process a single chunk of structures
Run with: python process_chunk.py --chunk 1
"""

from pathlib import Path
import numpy as np
import itertools
import configargparse
import pickle
import gc
from tqdm import tqdm

from utility import file_saver

def parser_client():
    parser = configargparse.ArgParser()
    parser.add_argument('--chunk', type=int, required=True, help="Chunk number (1-4)")
    parser.add_argument('--output_file_names', type=str, default='desc')
    parser.add_argument('--output_folder', type=str, default='desc_out')
    args, _ = parser.parse_known_args()
    return args

args = parser_client()

output_folder = Path(args.output_folder)
output_file_names = args.output_file_names
chunk_num = args.chunk

# Load the chunk data
chunk_file = output_folder / f"{output_file_names}_chunk_{chunk_num}.pkl"
with open(chunk_file, 'rb') as f:
    chunk_data = pickle.load(f)

atom_files = chunk_data['atom_files']
energy_files = chunk_data['energy_files']
total_chunks = chunk_data['total_chunks']

print(f"Processing chunk {chunk_num}/{total_chunks} with {len(atom_files)} structures")
print("=" * 60)

# ============================================================
# Geometry processing functions (copied from parse_orca.py)
# ============================================================

def unique_atom_pairs(atom_info):
    return list(itertools.combinations(atom_info, 2))

def unique_atom_triplets(atom_info):
    atom_triplets = []
    for combo in itertools.combinations(atom_info, 3):
        i, j, k = combo
        atom_triplets.append((i, j, k))
        atom_triplets.append((j, i, k))
        atom_triplets.append((k, i, j))
    return atom_triplets

def calculate_distance(atom1, atom2):
    pair_ids = (atom1[0], atom2[0])
    x1, y1, z1 = atom1[2], atom1[3], atom1[4]
    x2, y2, z2 = atom2[2], atom2[3], atom2[4]
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    r_scal = np.sqrt(dx*dx + dy*dy + dz*dz)
    return np.float64(r_scal), pair_ids

def calculate_displacement(atom1, atom2):
    pair_ids = (atom1[0], atom2[0])
    x1, y1, z1 = atom1[2], atom1[3], atom1[4]
    x2, y2, z2 = atom2[2], atom2[3], atom2[4]
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    r_scal = np.sqrt(dx*dx + dy*dy + dz*dz)
    if r_scal > 0:
        r_unit_vec = (dx/r_scal, dy/r_scal, dz/r_scal)
    else:
        r_unit_vec = (0.0, 0.0, 0.0)
    return r_unit_vec, pair_ids

def calculate_angle(atom1, atom2, atom3):
    triplet_ids = (atom1[0], atom2[0], atom3[0])
    x1, y1, z1 = atom1[2], atom1[3], atom1[4]
    x2, y2, z2 = atom2[2], atom2[3], atom2[4]
    x3, y3, z3 = atom3[2], atom3[3], atom3[4]

    a_vec = (x1 - x2, y1 - y2, z1 - z2)
    b_vec = (x3 - x2, y3 - y2, z3 - z2)
    
    dot_num = a_vec[0]*b_vec[0] + a_vec[1]*b_vec[1] + a_vec[2]*b_vec[2]
    norm_a = np.sqrt(a_vec[0]*a_vec[0] + a_vec[1]*a_vec[1] + a_vec[2]*a_vec[2])
    norm_b = np.sqrt(b_vec[0]*b_vec[0] + b_vec[1]*b_vec[1] + b_vec[2]*b_vec[2])
    
    if norm_a * norm_b > 0:
        cos_theta = np.clip(dot_num / (norm_a * norm_b), -1.0, 1.0)
    else:
        cos_theta = 1.0
    
    theta = np.arccos(cos_theta)
    return cos_theta, theta, triplet_ids

def process_structure(struct_name, atom_info):
    """Process a single structure and return its pair and triplet geometries."""
    
    # Generate pairs
    atom_pairs = unique_atom_pairs(atom_info)
    distances = []
    displacements = []
    
    for pair in atom_pairs:
        r, ids = calculate_distance(pair[0], pair[1])
        distances.append((ids, r))
        u, _ = calculate_displacement(pair[0], pair[1])
        displacements.append((ids, u))
    
    # Create lookup dictionaries
    dist_dict = {tuple(sorted(ids)): r for ids, r in distances}
    disp_dict = {tuple(sorted(ids)): d for ids, d in displacements}
    
    # Generate triplets
    atom_triplets = unique_atom_triplets(atom_info)
    angle_distances = []
    
    for triplet in atom_triplets:
        cos_theta, theta, ids = calculate_angle(triplet[0], triplet[1], triplet[2])
        id1, id2, id3 = ids
        d12 = dist_dict.get(tuple(sorted((id1, id2))), 0.0)
        d23 = dist_dict.get(tuple(sorted((id2, id3))), 0.0)
        d13 = dist_dict.get(tuple(sorted((id1, id3))), 0.0)
        u12 = disp_dict.get(tuple(sorted((id1, id2))), (0.0, 0.0, 0.0))
        u23 = disp_dict.get(tuple(sorted((id2, id3))), (0.0, 0.0, 0.0))
        u13 = disp_dict.get(tuple(sorted((id1, id3))), (0.0, 0.0, 0.0))
        angle_distances.append(((cos_theta, theta), (d12, d23, d13), (u12, u23, u13)))

    return (struct_name, distances, displacements), (struct_name, angle_distances)

# ============================================================
# Process the chunk
# ============================================================
print(f"\nProcessing {len(atom_files)} structures in chunk {chunk_num}...")
print("=" * 60)

atom_pair_geometry = []
atom_triplet_geometry = []

for idx, (filename, atom_info, energy_eV) in enumerate(tqdm(atom_files, desc=f"Chunk {chunk_num}")):
    try:
        pair_entry, triplet_entry = process_structure(filename, atom_info)
        atom_pair_geometry.append(pair_entry)
        atom_triplet_geometry.append(triplet_entry)
        
    except Exception as e:
        print(f"\n  Error processing {filename}: {e}")
        continue

# Save chunk 
result_file = output_folder / f"{output_file_names}_chunk{chunk_num}_results.pkl"
with open(result_file, 'wb') as f:
    pickle.dump({
        'pair': atom_pair_geometry,
        'triplet': atom_triplet_geometry,
        'energies': energy_files,
        'chunk': chunk_num,
        'count': len(atom_pair_geometry)
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nChunk {chunk_num} complete! Processed {len(atom_pair_geometry)} structures")