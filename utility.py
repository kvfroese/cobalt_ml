from pathlib import Path


def path_handler(file_name, folder):
    file = Path(file_name)
    file = file.with_suffix(".pkl")

    file_folder = Path(folder)
    file_path = file_folder.joinpath(file)
    return file_path, file_folder