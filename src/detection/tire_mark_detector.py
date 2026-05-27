import cv2
import numpy as np
from pathlib import Path

def detect_tire_marks(image_path_or_img, debug=False):
    """
    Advanced hybrid CV pipeline to detect linear, parallel tire mark signatures 
    in sandy riverbed terrain using Gaussian gradients, ridge extraction, 
    and Hough Transform parallel line matching.
    """
    if isinstance(image_path_or_img, (str, Path)):
        img = cv2.imread(str(image_path_or_img))
        if img is None:
            raise ValueError(f"Could not load image from {image_path_or_img}")
    else:
        img = image_path_or_img.copy()

    h, w, _ = img.shape
    
    # 1. Convert to Grayscale & Apply Smoothing to reduce fine sand noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Extract linear tracks using Sobel gradients (Ridge Approximation)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    # 3. Dynamic Thresholding to isolate prominent sand ridges
    _, thresh = cv2.threshold(magnitude, 45, 255, cv2.THRESH_BINARY)
    
    # Morphological cleaning to link linear patterns
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. Hough Line Transform to extract linear segments
    # Max line gap allows linking broken tracks in dry/wet sand transition
    lines = cv2.HoughLinesP(cleaned, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)
    
    detected_lines = []
    overlay = img.copy()
    
    if lines is not None:
        # 5. Parallel Line Matching (Tire tracks always form parallel pairs)
        # We group lines by angle and check if they have a consistent track distance
        for line1 in lines:
            x1_1, y1_1, x2_1, y2_1 = line1[0]
            angle1 = np.arctan2(y2_1 - y1_1, x2_1 - x1_1) * 180 / np.pi
            
            for line2 in lines:
                x1_2, y1_2, x2_2, y2_2 = line2[0]
                if x1_1 == x1_2 and y1_1 == y1_2:
                    continue
                    
                angle2 = np.arctan2(y2_2 - y1_2, x2_2 - x1_2) * 180 / np.pi
                
                # Check if lines are roughly parallel (angle tolerance < 10 degrees)
                if abs(angle1 - angle2) < 10:
                    # Calculate approximate perpendicular distance
                    dist = np.abs((x2_1 - x1_1)*(y1_2 - y1_1) - (x1_1 - x1_2)*(y2_1 - y1_1)) / np.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
                    
                    # Typical tire track width is bounded (e.g. 15 to 80 pixels at drone altitude)
                    if 15 < dist < 80:
                        detected_lines.append(line1[0])
                        detected_lines.append(line2[0])

        # Remove duplicate line indices
        unique_lines = []
        for line in detected_lines:
            if not any(np.array_equal(line, ul) for ul in unique_lines):
                unique_lines.append(line)

        # 6. Draw Glowing High-Aesthetic Overlays (Vibrant Purple for Tire Marks)
        for line in unique_lines:
            x1, y1, x2, y2 = line
            # Under-glow line
            cv2.line(overlay, (x1, y1), (x2, y2), (240, 0, 160), 4, cv2.LINE_AA)
            # Core line
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    # Blend overlay with original
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    return img, len(detected_lines) > 0

if __name__ == "__main__":
    # Test on a raw frame if available
    workspace_dir = Path(r"d:\Projects\Sand-Mining_Detection")
    sample_img_path = next(workspace_dir.glob("data/processed/frames/*/*.jpg"), None)
    
    if sample_img_path:
        print(f"Testing tire mark detector on sample frame: {sample_img_path}")
        result_img, found = detect_tire_marks(sample_img_path)
        print(f"Tire marks detected: {found}")
    else:
        print("No sample images found to test.")
