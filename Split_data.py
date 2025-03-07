import os
import json
import shutil

# split data according to the json file

# Paths
json_file = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\WLASL_dataset\\WLASL_v0.3.json"
train_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_train_sample_FINAL"
val_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_val_sample_FINAL"
mhi_folder = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_MHI_FINAL"

# Ensure base train/val directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# json file stuff
with open(json_file, 'r') as f:
    data = json.load(f)

count = 0
n_count = 0

# process data (json file setup)

for entry in data:
    gloss = entry["gloss"]  # Gloss label
    for instance in entry["instances"]:
        video_id = instance["video_id"]  # vid  ID
        split = instance["split"]  # 'train' or 'val'

        # naming convention... video_number_mhi.jpg 
        mhi_image_filename = f"{video_id}_mhi.jpg"
        source_path = os.path.join(mhi_folder, mhi_image_filename)

        if os.path.exists(source_path):
            # target directory (where to put it: train/split)
            target_dir = os.path.join(train_dir if split == 'train' else val_dir, gloss)

            os.makedirs(target_dir, exist_ok=True)

            # Copy file
            shutil.copy(source_path, os.path.join(target_dir, mhi_image_filename))
            count +=1
            
        else:
            # print(f"Warning: {source_path} not found!")
            n_count += 1

print(f"Image classification complete! {count} files copied.")
print(n_count)

# Ensure that the val and training directories match 
# (note some of the videos didn't exist so want to filter those files out and esnure there are no folders that exist in one and not the other)

subfolders_folder1 = set(os.listdir(val_dir))  # subfolders in validation
subfolders_folder2 = set(os.listdir(train_dir))  # subfolders in training

# folders that are in 1 but not 2... and vice versa
folders_to_delete_from_folder1 = subfolders_folder1 - subfolders_folder2
folders_to_delete_from_folder2 = subfolders_folder2 - subfolders_folder1

# loop and delete for both// exist in one but not the other
for subfolder in folders_to_delete_from_folder1:
    folder_to_delete = os.path.join(val_dir, subfolder)
    print(f"Deleting folder from folder1: {folder_to_delete}")
    shutil.rmtree(folder_to_delete)  
for subfolder in folders_to_delete_from_folder2:
    folder_to_delete = os.path.join(train_dir, subfolder)
    print(f"Deleting folder from folder2: {folder_to_delete}")
    shutil.rmtree(folder_to_delete)  

print("Done deleting... val and training match.")
