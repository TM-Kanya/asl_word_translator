import os
import json
import shutil
import random

# Paths
json_file = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\WLASL_dataset\\WLASL_v0.3.json"
train_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_train_new_frames"
val_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_val_new_frames"
test_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_test_new_frames"
frames_folder = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_Frames"

# Ensure base directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# json file stuff
with open(json_file, 'r') as f:
    data = json.load(f)

count = 0
n_count = 0

# Dictionary to store frames for each gloss
gloss_dict = {}


print("starting... ")
# Process data (json file setup)
for entry in data:
    gloss = entry["gloss"]  # Gloss label
    class_instances = []  # Store instances for each gloss

    for instance in entry["instances"]:
        video_id = instance["video_id"]  # video ID
        
        # Assume frames are stored in folders named by video_id
        video_folder = os.path.join(frames_folder, video_id)
        
        if os.path.exists(video_folder):
            frames = sorted(os.listdir(video_folder))  # Sort to ensure proper order
            for frame in frames:
                frame_path = os.path.join(video_folder, frame)
                class_instances.append(frame_path)

        else:
            n_count += 1  # If folder doesn't exist, increment missing count

    # Store the frames in the gloss_dict
    gloss_dict[gloss] = class_instances

# Each word, split frame

for gloss, frames in gloss_dict.items():
    random.shuffle(frames)

    # Calculate 70% train, 15% validation, 15% test split
    num_instances = len(frames)
    train_split = int(num_instances * 0.8)
    val_split = int(num_instances * 0.1)

    if train_split > 0 and val_split > 0:
        # Get the respective splits
        train_frames = frames[:train_split]
        val_frames = frames[train_split:train_split + val_split]
        test_frames = frames[train_split + val_split:]

        # Copy the frames to respective directories for the gloss
        for split, split_frames in zip(['train', 'val', 'test'], [train_frames, val_frames, test_frames]):
            target_dir = {
                'train': os.path.join(train_dir, gloss),
                'val': os.path.join(val_dir, gloss),
                'test': os.path.join(test_dir, gloss),
            }[split]

            os.makedirs(target_dir, exist_ok=True)
            for frame_path in split_frames:
                frame_filename = os.path.basename(frame_path)
                shutil.copy(frame_path, os.path.join(target_dir, frame_filename))
                count += 1

print(f"Frame classification complete! {count} files copied.")
print(f"{n_count} folders were missing!")

# Ensure that the val, test, and training directories match
subfolders_folder1 = set(os.listdir(val_dir))  # subfolders in validation
subfolders_folder2 = set(os.listdir(train_dir))  # subfolders in training
subfolders_folder3 = set(os.listdir(test_dir))  # subfolders in test

# folders that are in one but not the others
folders_to_delete_from_folder1 = subfolders_folder1 - subfolders_folder2 - subfolders_folder3
folders_to_delete_from_folder2 = subfolders_folder2 - subfolders_folder1 - subfolders_folder3
folders_to_delete_from_folder3 = subfolders_folder3 - subfolders_folder1 - subfolders_folder2

# loop and delete for both: folders in one but not in others
for subfolder in folders_to_delete_from_folder1:
    folder_to_delete = os.path.join(val_dir, subfolder)
    print(f"Deleting folder from validation: {folder_to_delete}")
    shutil.rmtree(folder_to_delete)
for subfolder in folders_to_delete_from_folder2:
    folder_to_delete = os.path.join(train_dir, subfolder)
    print(f"Deleting folder from training: {folder_to_delete}")
    shutil.rmtree(folder_to_delete)
for subfolder in folders_to_delete_from_folder3:
    folder_to_delete = os.path.join(test_dir, subfolder)
    print(f"Deleting folder from test: {folder_to_delete}")
    shutil.rmtree(folder_to_delete)

print("Done deleting... val, train, and test match.")
