import cv2
import numpy as np
import os

#function for Extract a Region of Interest (ROI) from an image.
def roi_select(input_path, output_path, x, y, width, height):
    try:
        # Check if the input image exists
        if not os.path.exists(input_path):
            print(f"Error: Input image {input_path} does not exist")
            return False
        
        # Read the input image
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Error: Failed to read image {input_path}")
            return False
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Validate ROI coordinates
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            print("Error: ROI coordinates are outside image boundaries")
            # Adjust ROI to fit within image boundaries
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            width = min(width, img_width - x)
            height = min(height, img_height - y)
            print(f"Adjusted ROI to: x={x}, y={y}, width={width}, height={height}")
        
        # Extract ROI
        roi = image[y:y+height, x:x+width]
        
        # Save ROI to output path
        cv2.imwrite(output_path, roi)
        
        # Validate if file was created
        if os.path.exists(output_path):
            print(f"ROI extracted and saved to {output_path}")
            return True
        else:
            print(f"Error: Failed to save ROI to {output_path}")
            return False
            
    except Exception as e:
        print(f"Error in ROI extraction: {str(e)}")
        return False



#function for highlight a Region of Interest (ROI) in an image.
def highlight_roi(input_path, output_path, x, y, width, height):
    try:
        # Read the input image
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Error: Failed to read image {input_path}")
            return False
        
        # Create a copy of the image
        result = image.copy()
        
        # Draw rectangle around ROI
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Save the result
        cv2.imwrite(output_path, result)
        
        return True
        
    except Exception as e:
        print(f"Error in highlighting ROI: {str(e)}")
        return False