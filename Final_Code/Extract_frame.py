import cv2
import os

def extract_frames_from_videos(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4") or f.endswith(".avi") or f.endswith(".mov")]
    video_files.sort()  # sort -- for organization/ordering
    
    for i in range(0, len(video_files)):
        filename = video_files[i]
        video_path = os.path.join(input_folder, filename)
        video_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
        os.makedirs(video_output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {filename}.")
            continue
        
        frame_count = 0
        saved_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  #end, video done (no frames left)
            
            if frame_count % 5 == 0:  # save every 5 frames
                frame_filename = os.path.join(video_output_folder, f"frame_{saved_frame_count:04d}.png")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_frame_count} frames from {filename} and saved to {video_output_folder}")
        
        # # Delete the processed video file
        # os.remove(video_path)
        # print(f"Deleted video file: {filename}")
        

# Example usage
input_folder = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\WLASL_dataset\\videos_og"  # Change this to your input folder containing videos
output_folder = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_Frames"  # Change this to your desired output folder
extract_frames_from_videos(input_folder, output_folder)


print("Done")
