import cv2
import numpy as np
import os


#function to apply zooming
def zoomed_image(input_path, output_path, zoom_factor=2.0):
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
        
        # Get the dimensions of the image
        height, width = image.shape[:2]
        
        # Determine the center of the image
        center_x, center_y = width // 2, height // 2
        
        # Calculate the ROI size based on the zoom factor
        # Lower zoom factor for better quality
        roi_width = int(width / zoom_factor)
        roi_height = int(height / zoom_factor)
        
        # Calculate the ROI coordinates
        x = center_x - roi_width // 2
        y = center_y - roi_height // 2
        
        # Ensure ROI is within image boundaries
        x = max(0, min(x, width - roi_width))
        y = max(0, min(y, height - roi_height))
        
        # Extract the ROI
        roi = image[y:y+roi_height, x:x+roi_width]
        
        # Resize using Lanczos interpolation for better quality
        zoomed = cv2.resize(roi, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply a combination of denoising and gentle sharpening
        # First denoise to remove artifacts
        zoomed = cv2.fastNlMeansDenoisingColored(zoomed, None, 10, 10, 7, 21)
        
        # Then apply gentle sharpening
        kernel = np.array([[-0.3, -0.3, -0.3],
                          [-0.3, 3.4, -0.3],
                          [-0.3, -0.3, -0.3]])
        zoomed = cv2.filter2D(zoomed, -1, kernel)
        
        # Save the zoomed image
        cv2.imwrite(output_path, zoomed)
        
        # Validate if file was created
        if os.path.exists(output_path):
            print(f"Zoomed image saved to {output_path}")
            return True
        else:
            print(f"Error: Failed to save zoomed image to {output_path}")
            return False
            
    except Exception as e:
        print(f"Error in image zooming: {str(e)}")
        return False



#function to zoom into a specific ROI of an image
def zoom_roi(input_path, output_path, x, y, width, height, zoom_factor=2.0):
    try:
        # Read the input image
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Error: Failed to read image {input_path}")
            return False
        
        # Get the dimensions of the image
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
        
        # Apply denoising before resizing to reduce noise amplification
        roi = cv2.fastNlMeansDenoisingColored(roi, None, 5, 5, 7, 21)
        
        # Resize ROI to the original image size using Lanczos interpolation
        zoomed = cv2.resize(roi, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply subtle sharpening to enhance details without creating artifacts
        kernel = np.array([[-0.2, -0.2, -0.2],
                          [-0.2, 2.8, -0.2],
                          [-0.2, -0.2, -0.2]])
        zoomed = cv2.filter2D(zoomed, -1, kernel)
        
        # Save the zoomed image
        cv2.imwrite(output_path, zoomed)
        
        return True
        
    except Exception as e:
        print(f"Error in ROI zooming: {str(e)}")
        return False