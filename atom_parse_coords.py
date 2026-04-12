from pathlib import Path
import numpy as np
import itertools
import configargparse
import pickle

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
    parser.add_argument('--geometry_file_names',
                        type=str,
                        help="Base name of files that contain geometry values")
    parser.add_argument('--geometry_folder',
                        type=str,
                        help="Folder path to all geometry values. Recommend do not change")
    
    args, unknown_args = parser.parse_known_args()
    return args

parser = parser_client()

geometry_file_name = parser.geometry_file_names
geometry_folder = parser.geometry_folder


## Parsing
def open_file(file_path):
    with open(Path(file_path), 'r') as f:
        lines = f.readlines()
    return lines

def atoms(lines):
    # Find starting line
    for i, line in enumerate(lines):
        if line.lstrip().startswith('*** FINAL ENERGY EVALUATION'):
            start_line = i + 6
            print(f"Found line {start_line}")
            break
    else:
        raise ValueError("Could not find '*** FINAL' in the file.")

    # Find ending line
    for j, line in enumerate(lines[start_line:]):
        if line.startswith('-'):
            end_line = start_line + j - 1 # exclude blank line after coordinates
            print(f"Found line {end_line}")
            break
    else:
        raise ValueError("Could not find end line (line starting with '-') after '*** FINAL'.")

    return start_line, end_line

atom_info = []
def extract_atom_info(lines, start_line, end_line):
    for line in lines[start_line:end_line]:
        atom_sym, x, y, z = line.split() # separates by whitespace
        atom_sym = atom_sym.strip() # remove whitespace
        x, y, z = float(x), float(y), float(z)
        atom_info.append((atom_sym, x, y, z))
    return atom_info

folder_path = Path('orca_outputs')


'''
We prduce a list of tuples of tuples: The order of atom, x, y, z and
order of the atom lines are preserved because we need to keep track
of atom is which for radial and angular descriptors. The output is a list of tuples,
where each tuple contains
    the file name,
    a tuple of atom information (atom symbol and coordinates),
    and the number of atoms.
'''
atom_files = []
for file in folder_path.glob('*'):
    try:
        lines = open_file(file)
        start_line, end_line = atoms(lines)

        atom_info = []  # reset per file
        id_counter = 0
        for line in lines[start_line:end_line]:
            atom_sym, x, y, z = line.split()  # separates by whitespace
            atom_sym = atom_sym.strip()  # remove whitespace
            x, y, z = float(x), float(y), float(z)
            atom_info.append((id_counter, atom_sym, x, y, z))
            id_counter += 1
        atom_info = tuple(atom_info)

        atom_number = len(atom_info)
        atom_files.append((file.name, atom_info, atom_number))
    except Exception as e:
        print(f"Error processing {file.name}: {e}")
        continue

print(f"{len(atom_files)} files processed, with example output:\n{str(atom_files[0]):.50} ...")

## Paired/Triplet Atomic Values

# Generate unique atom pairs, per file, for radial descriptors
def unique_atom_pairs(atom_info, ):
    atom_pairs = list(itertools.combinations(atom_info, 2))
    return atom_pairs

atom_pairs_list = []
for structure in atom_files:
    set_pairs = unique_atom_pairs(structure[1])
    num_pairs = len(set_pairs)
    atom_pairs_list.append((set_pairs, num_pairs))

print(f"You have {len(atom_pairs_list[0][0])} unique atom distance descriptions for the first structure, with example:\n {atom_pairs_list[0][0][2]}")

'''
The goal is to create unique atom triplets for angular descriptors,
that are invariant to frame of reference.
The summation rules are such that ∠|BAC| = ∠|CAB|, however ∠|ABC| is different.
We can generate unique triplets by first generating unique combinatorial pairs,
and then adding a third atom that is not in the pair.
This way we ensure that we do not have duplicate triplets
that differ only by the order of the first and third atoms.
'''
def unique_atom_triplets(structure):
    atom_triplets = []
    outer_pairs = list(itertools.combinations(structure, 2))
    for outer in outer_pairs:
        for inner in structure:
            if inner not in outer:
                atom_triplets.append((outer[0], inner, outer[1]))
    return atom_triplets

atom_triplets_list = []
for structure in atom_files:
    set_triplets = unique_atom_triplets(structure[1])
    num_triplets = len(set_triplets)
    atom_triplets_list.append((set_triplets, num_triplets))

atom_triplets_list = unique_atom_triplets(atom_files[0][1])
print(f"You have {len(atom_triplets_list)} atom angle descriptions for the first structure, with example:\n {str(atom_triplets_list[0]):.50} ... ")


## Calculation of Distance and Angle

def calculate_distance(atom1, atom2):
    # 0 is id, 1 is atom symbol, 2-4 are x, y, z coordinates
    pair_ids = atom1[0], atom2[0]
    x1, y1, z1 = atom1[2], atom1[3], atom1[4]
    x2, y2, z2 = atom2[2], atom2[3], atom2[4]
    r_scal = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
    r_scal = np.float64(r_scal)
    return r_scal, pair_ids

def calculate_displacement(atom1, atom2):
    pair_ids = atom1[0], atom2[0]
    x1, y1, z1 = atom1[2], atom1[3], atom1[4]
    x2, y2, z2 = atom2[2], atom2[3], atom2[4]
    r_i = x2 - x1
    r_j = y2 - y1
    r_k = z2 - z1
    r_scal = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
    r_unit_vec = r_i/r_scal, r_j/r_scal, r_k/r_scal
    return r_unit_vec, pair_ids
    

def calculate_angle(atom1, atom2, atom3):
    triplet_ids = atom1[0], atom2[0], atom3[0]
    x1, y1, z1 = atom1[2], atom1[3], atom1[4]
    x2, y2, z2 = atom2[2], atom2[3], atom2[4] # central atom
    x3, y3, z3 = atom3[2], atom3[3], atom3[4]

    a_vec = (x1 - x2, y1 - y2, z1 - z2)
    b_vec = (x3 - x2, y3 - y2, z3 - z2)

    dot_num = np.dot(a_vec, b_vec)
    dot_denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    cos_theta = np.clip(dot_num / dot_denom, -1.0, 1.0)  # clip to avoid numerical issues
    theta = np.arccos(cos_theta)
    theta = np.clip(theta, 0.0, np.pi)  # clip to valid range

    return cos_theta, theta, triplet_ids

def calculate_all_distances(atom_pairs):
    distances = []
    displaces = []
    for pair in atom_pairs:
        r_scal, pair_ids = calculate_distance(pair[0], pair[1])
        distances.append((pair_ids, r_scal))
        r_vec, pair_ids = calculate_displacement(pair[0], pair[1])
        displaces.append((pair_ids, r_vec))
    return distances, displaces

def calculate_all_angles(atom_triplets):
    angles = []
    for triplet in atom_triplets:
        cos_theta, theta, triplet_ids = calculate_angle(triplet[0], triplet[1], triplet[2])
        angles.append((triplet_ids, (cos_theta, theta)))
    return angles

# Unnecessary now, but useful for seeing ID numbers and checking outputs
atom_pair_geometry = []
for i, structure in enumerate(atom_files):
    atom_pairs = unique_atom_pairs(structure[1])
    distances, displaces = calculate_all_distances(atom_pairs)
    atom_pair_geometry.append((structure[0], distances, displaces))

print(f"You have {len(atom_pair_geometry[0][0])} number of pairs in the first structure, example:\n{str(atom_pair_geometry):.50}")
## Sorting, final data structure

atom_triplet_geometry = []
for i, structure in enumerate(atom_files):
    distances, displaces = calculate_all_distances(unique_atom_pairs(structure[1]))
    angles = calculate_all_angles(unique_atom_triplets(structure[1]))
    
    # Create distance dictionary with sorted pair IDs as keys
    distance_dict = {tuple(sorted(pair_ids)): r for pair_ids, r in distances}
    displace_dict = {tuple(sorted(pair_ids)): r for pair_ids, r in displaces}
    
    # For each angle, collect the three associated distances
    angle_distances = []
    for triplet_ids, angle_tuple in angles:
        cos_theta, theta = angle_tuple
        id1, id2, id3 = triplet_ids
        dist1 = distance_dict[tuple(sorted((id1, id2)))]
        dist2 = distance_dict[tuple(sorted((id2, id3)))]
        dist3 = distance_dict[tuple(sorted((id1, id3)))]
        disp1 = displace_dict[tuple(sorted((id1, id2)))]
        disp2 = displace_dict[tuple(sorted((id2, id3)))]
        disp3 = displace_dict[tuple(sorted((id1, id3)))]
        angle_distances.append(((cos_theta, theta), (dist1, dist2, dist3), (disp1, disp2, disp3)))
    
    atom_triplet_geometry.append((structure[0], angle_distances))

print(f"Atom geometries:\n{str(atom_triplet_geometry):.300} ...")

'''
The final data structure is a list of tuples. Each tuple contains:
- the file name (per structure)
- a list of geometric tuples, where each tuple contains:
    - the angle (technically cos theta) for a unique triplet of atoms
    - a list of the three distances associated with that angle

We use the angle and the three distances for the angular descriptor
We use just the distance (one at time) for the radial descriptor
'''

## Saving Geometry Info
if __name__ == '__main__':
    file_saver(geometry_file_name, geometry_folder, "triplet", atom_triplet_geometry)
    file_saver(geometry_file_name, geometry_folder, "pair", atom_pair_geometry)