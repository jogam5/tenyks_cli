import roboflow_datasets as roboflow
import os
import shutil

def make_dir(new_dir_path):
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    print(f'{new_dir_path} created')

def copy_from_to(file_path_from, file_path_to):
    shutil.copy(file_path_from, file_path_to)

def count_files_in_dir(dir_path):
    i = 0
    for _ in os.listdir(dir_path):
        i += 1
    print(i)

def get_common_files_n_dir(folder_paths):
    common_files = None
    file_extension = '.txt'

    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]

        if common_files is None:
            common_files = set(files)
        else:
            common_files = common_files.intersection(files)

    if common_files is None:
        common_files = set()

    print(f"Common files in folders: {len(common_files)}")
    return common_files

def copy_imgs_to_subset_dir(common_filenames, source_dir_path, dest_dir_path):
    os.makedirs(dest_dir_path, exist_ok=True)
    for filename in common_filenames:
        if filename.endswith('.txt'):
            copy_from_to(f'{source_dir_path}/{filename.replace(".txt",".jpg")}',
                                  dest_dir_path)
    count_files_in_dir(dest_dir_path)

def copy_anns_to_subset_dir(common_filenames, source_dir_path, dest_dir_path):
    os.makedirs(dest_dir_path, exist_ok=True)
    for filename in common_filenames:
        if filename.endswith('.txt'):
            copy_from_to(f'{source_dir_path}/{filename}', dest_dir_path)
    count_files_in_dir(dest_dir_path)
    
def eliminate_duplicates_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        #filepath = ''
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            remove_duplicate_lines(filepath)
        
def remove_duplicate_lines(file_path):
    unique_lines = set()

    # Read the file and collect unique lines
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line not in unique_lines:
                unique_lines.add(line)

    # Write back the unique lines to the file
    with open(file_path, 'w') as file:
        for line in unique_lines:
            file.write(line + '\n')
            
def get_nested_subfolder_paths(folder_path):
    subfolder_paths = []

    # Iterate over the contents of the current folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # Check if the item is a subfolder
        if os.path.isdir(item_path):
            subfolder_paths.append(item_path)

            # Recursively explore nested subfolders
            subfolder_paths.extend(get_nested_subfolder_paths(item_path))
    
    return subfolder_paths

def ingesting_data(experiment_dir=None, predictions_dir=None, imgs_dir=None, 
                   annotations_dir=None, dataset_dir=None):
    # caveat: only supports 1 model
    common_files = get_common_files_n_dir([predictions_dir])
    
    copy_imgs_to_subset_dir(common_files, imgs_dir, f'{experiment_dir}/images')
    copy_anns_to_subset_dir(common_files, annotations_dir, f'{experiment_dir}/anns')
    copy_anns_to_subset_dir(common_files, predictions_dir, f'{experiment_dir}/preds')
    
    roboflow.get_annotations(f'{experiment_dir}/images', f'{experiment_dir}/anns', 
                         experiment_dir, dataset_dir)
    roboflow.get_predictions(f'{experiment_dir}/images', f'{experiment_dir}/preds', 
                         experiment_dir, dataset_dir)
    print('ingesting data completed')
