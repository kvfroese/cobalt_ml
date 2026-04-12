from pathlib import Path
import pickle
import configargparse

def parser_client():
    parser = configargparse.ArgParser(
        description='Highly recommend to use a config file instead of manual arguments. If unspecififed, a default one will be used!',
        default_config_files=[Path('config/default.ini')]
    )
    parser.add_argument('--run_id',
                        '-i',
                        type=str,
                        help="Attaches ID cookie to all files saved. Useful for doing multiple runs")
    args, unknown_args = parser.parse_known_args()
    return args

parser = parser_client()

cookie = parser.run_id

def path_handler(file_name, folder):
    file = Path(file_name)
    file = file.with_suffix(".pkl")

    file_folder = Path(folder)
    file_path = file_folder.joinpath(file)
    return file_path, file_folder

def file_saver(file_name, folder, accent, to_save):
    try:
        unique_file_name = file_name + "_" + accent + "__" + cookie
        file_path, file_folder = path_handler(unique_file_name, folder)
        if not file_folder.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{str(unique_file_name)} folder has been created")
        with open(file_path, 'wb') as f:
            pickle.dump(to_save, f)
        print(f"File {unique_file_name} is saved to {folder}")
    except Exception as e:
        print(f"File unable to be saved due to:\n{e}")

def file_loader(file_name, folder, accent):
    try:
        unique_file_name = file_name + "_" + accent + "__" + cookie
        file_path, _ = path_handler(unique_file_name, folder)
        with open(file_path, 'rb') as f:
            loaded_name = pickle.load(f)
            print(f"{unique_file_name} successfully loaded from {folder}")
        return loaded_name
    except Exception as e:
        print(f"Error in opening file due to:\n{e}")