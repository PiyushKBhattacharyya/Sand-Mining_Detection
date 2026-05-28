import os
import cv2
import sys

def extract_frames(video_path, output_dir):
    """
    Extracts every frame from the video file and saves them to output_dir.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    print(f"\nProcessing video: {video_name}")
    print(f"Saving frames to: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")

    # Calculate frame interval for 5 seconds
    frame_interval = max(1, int(round(fps * 5)))
    print(f"Extracting 1 frame every {frame_interval} frames (~5 seconds)")

    frame_idx = 0
    saved_count = 0
    success = True
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if frame_idx % frame_interval == 0:
            # Format filename with zero-padded index (e.g. frame_000000.jpg)
            frame_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            
            # Save frame
            cv2.imwrite(frame_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1
        
        frame_idx += 1
        
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processed {frame_idx}/{total_frames} frames ({progress:.1f}%), Saved: {saved_count} frames", end='\r')
            sys.stdout.flush()

    cap.release()
    print(f"\nFinished extraction! Total frames processed: {frame_idx}, Saved: {saved_count}")

if __name__ == "__main__":
    # Define paths
    workspace_dir = r"d:\Projects\Sand-Mining_Detection"
    raw_dir = os.path.join(workspace_dir, "data", "raw")
    output_base_dir = os.path.join(workspace_dir, "data", "processed", "frames")
    
    videos = [
        "DJI_20260522133226_0026_D.MP4",
        "DJI_20260522133752_0027_D.MP4"
    ]
    
    for video in videos:
        video_path = os.path.join(raw_dir, video)
        video_basename = os.path.splitext(video)[0]
        output_dir = os.path.join(output_base_dir, video_basename)
        
        extract_frames(video_path, output_dir)
