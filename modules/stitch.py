# import cv2
# import os

# def stitch_images(images):
#     """
#     Stitch multiple overlapping images into a single high-resolution image.
#     :param images: List of images to be stitched.
#     :return: Stitched image or None if stitching fails.
#     """
#     stitcher = cv2.Stitcher_create()
#     status, stitched = stitcher.stitch(images)
    
#     if status == cv2.Stitcher_OK:
#         return stitched
#     else:
#         print("Image stitching failed!")
#         return None

# if __name__ == "__main__":
#     # Define image folder path
#     image_folder = "images"

#     # Get image file paths from the folder
#     image_paths = [os.path.join(image_folder, "image1.jpg"), 
#                    os.path.join(image_folder, "image2.jpg"),
#                    os.path.join(image_folder, "image3.jpg"),
#                    os.path.join(image_folder, "image4.jpg")]

#     # Load images
#     images = [cv2.imread(path) for path in image_paths]

#     # Check if images are loaded correctly
#     if any(img is None for img in images):
#         print("Error: One or more images failed to load. Check paths and formats.")
#     else:
#         # Perform stitching
#         result = stitch_images(images)

#         if result is not None:
#             output_path = os.path.join(image_folder, "stitched_output.jpg")
#             cv2.imwrite(output_path, result)
#             print(f"✅ Stitched image saved at: {output_path}")

#             cv2.imshow("Stitched Image", result)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()







import cv2
import numpy as np
import os


#function for stitching images
def stitched_images(image_paths, output_path):
    """
    Stitch multiple microscope images into one seamless high-resolution image.
    
    Parameters:
    - image_paths: List of paths to the input images
    - output_path: Path to save the stitched output image
    
    Returns:
    - Boolean indicating success or failure
    """
    try:
        # Check if we have at least 2 images
        if len(image_paths) < 2:
            print("Error: At least 2 images are required for stitching")
            return False
        
        # Read all images
        images = []
        for path in image_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                else:
                    print(f"Warning: Could not read image {path}")
            else:
                print(f"Warning: Image {path} does not exist")
        
        if len(images) < 2:
            print("Error: At least 2 valid images are required for stitching")
            return False
        
        # Create a stitcher object
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        
        # Perform stitching
        status, stitched_img = stitcher.stitch(images)
        
        if status != cv2.Stitcher_OK:
            # If automatic stitching fails, try a feature-based approach
            print("Automatic stitching failed, trying feature-based approach...")
            stitched_img = feature_based_stitching(images)
            
            if stitched_img is None:
                print(f"Error: Image stitching failed with status {status}")
                return False
        
        # Save the result
        cv2.imwrite(output_path, stitched_img)
        print(f"Stitched image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error in image stitching: {str(e)}")
        return False


#function to create feature-based stitching using ORB
def feature_based_stitching(images):
    try:
        # Define feature detector
        detector = cv2.ORB_create(nfeatures=2000)
        
        # Detect features in all images
        keypoints = []
        descriptors = []
        
        for img in images:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect features
            kp, des = detector.detectAndCompute(gray, None)
            keypoints.append(kp)
            descriptors.append(des)
        
        # Match features between adjacent images
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Start with the first image
        result = images[0]
        
        # Stitch each subsequent image
        for i in range(1, len(images)):
            # Match features between result and current image
            matches = matcher.match(descriptors[0], descriptors[i])
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Take only good matches
            good_matches = matches[:int(len(matches) * 0.75)]
            
            if len(good_matches) < 4:
                print(f"Not enough good matches found between images 0 and {i}")
                continue
            
            # Get matched keypoints
            src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[i][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                print(f"Could not find homography between images 0 and {i}")
                continue
            
            # Apply homography to stitch images
            h, w = result.shape[:2]
            h2, w2 = images[i].shape[:2]
            
            # Warp the first image to align with the second
            result_warped = cv2.warpPerspective(result, H, (w2 + w, max(h, h2)))
            
            # Add the second image
            result_warped[0:h2, 0:w2] = images[i]
            
            # Update result for next iteration
            result = result_warped
            
            # Update descriptors for next iteration
            descriptors[0] = descriptors[i]
        
        return result
        
    except Exception as e:
        print(f"Error in feature-based stitching: {str(e)}")
        return None