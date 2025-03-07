import os
import cv2
import numpy as np

# magnify img
def upscale_image(image, scale_factor=2, interpolation=cv2.INTER_LANCZOS4):
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

# folders of frames, turn each into MHI
def create_mhi_from_folder(folder_path, output_path, threshold=30, motion_threshold=1000, duration=30, scale_factor=2):
    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")])
    
    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return
    
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]), cv2.IMREAD_GRAYSCALE)
    first_image = upscale_image(first_image, scale_factor)  # scale
    height, width = first_image.shape
    motion_history = np.zeros((height, width), dtype=np.float32)
    
    prev_frame = first_image
    timestamp = 0
    
    for filename in image_files[1:]:
        current_frame = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        current_frame = upscale_image(current_frame, scale_factor)  # scale frame
        
        # diff bt frames
        frame_diff = cv2.absdiff(prev_frame, current_frame)
        _, motion_mask = cv2.threshold(frame_diff, threshold, 1, cv2.THRESH_BINARY)
        
        # amt of motion 
        motion_amount = np.sum(motion_mask)
        
        # Only update motion history if there is significant motion
        if motion_amount > motion_threshold:
            motion_history[motion_mask == 1] = timestamp  
            motion_history[motion_mask == 0] = np.maximum(motion_history[motion_mask == 0] - 1, 0) 
            prev_frame = current_frame
            timestamp += 1  
        else:
            # print(f"Ignoring frame {filename} (motion amount: {motion_amount})")
            continue
    
    # normalize for visualization
    mhi_image = np.uint8(255 * (motion_history / duration))
    
    # put in folder
    folder_name = os.path.basename(folder_path)  
    output_file = os.path.join(output_path, f"{folder_name}_mhi.jpg")
    cv2.imwrite(output_file, mhi_image)
    # print(f"MHI saved")

# file organization of data, just go through and create them
def create_mhi_for_folders(input_folder, output_folder, threshold=30, motion_threshold=1000, duration=30, scale_factor=2):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
   
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            # print(f"Processing folder: {folder_path}")
            create_mhi_from_folder(folder_path, output_folder, threshold, motion_threshold, duration, scale_factor)
        else:
            # print(f"Skipping non-folder: {folder_name}")
            continue


# TUNE THESE!!!! (guess + check?)
# parameters 
threshold = 20  # motion detection
motion_threshold = 1000  # motion needed for fram
duration = 20  
scale_factor = 3  # make it bigger, a little small


input_folder_sample = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_Frames" 
output_folder_sample = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_MHI_FINAL"  


# runnnn!!!!
create_mhi_for_folders(input_folder_sample, output_folder_sample, threshold, motion_threshold, duration, scale_factor)
