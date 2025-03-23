import cv2
import numpy as np
import os
from scipy import ndimage



#function to enhance the focus on image to increase clarity
def auto_focus(input_path, output_path):
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
        
        # Step 1: Apply initial denoising to reduce noise before processing
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
        
        # Step 2: Convert to Lab color space for better color processing
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Step 3: Apply CLAHE on the L channel with microscope-specific parameters
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced_l = clahe.apply(l)
        
        # Step 4: Apply multi-scale unsharp masking for better detail enhancement
        gaussian_1 = cv2.GaussianBlur(enhanced_l, (0, 0), 1.0)
        gaussian_2 = cv2.GaussianBlur(enhanced_l, (0, 0), 3.0)
        gaussian_3 = cv2.GaussianBlur(enhanced_l, (0, 0), 5.0)
        
        # Create multi-scale unsharp mask with weighted contributions
        unsharp_1 = cv2.addWeighted(enhanced_l, 1.5, gaussian_1, -0.5, 0)
        unsharp_2 = cv2.addWeighted(enhanced_l, 1.3, gaussian_2, -0.3, 0)
        unsharp_3 = cv2.addWeighted(enhanced_l, 1.2, gaussian_3, -0.2, 0)
        
        # Combine the different scales
        enhanced_l = cv2.addWeighted(unsharp_1, 0.4, unsharp_2, 0.3, 0)
        enhanced_l = cv2.addWeighted(enhanced_l, 0.8, unsharp_3, 0.2, 0)
        
        # Step 5: Edge enhancement specific for microscope images
        edges = cv2.Laplacian(enhanced_l, cv2.CV_8U, ksize=3)
        edges = cv2.GaussianBlur(edges, (0, 0), 0.5)  # Smooth the edges slightly
        enhanced_l = cv2.addWeighted(enhanced_l, 1.0, edges, 0.2, 0)
        
        # Step 6: Reconstruct the Lab image with enhanced luminance
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        
        # Step 7: Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 8: Apply microscope-specific detail enhancement
        kernel = np.array([[-0.5, -0.5, -0.5],
                           [-0.5,  5.0, -0.5],
                           [-0.5, -0.5, -0.5]])
        enhanced_bgr = cv2.filter2D(enhanced_bgr, -1, kernel)
        
        # Step 9: Apply targeted contrast enhancement
        enhanced_bgr = cv2.convertScaleAbs(enhanced_bgr, alpha=1.15, beta=5)
        
        # Step 10: Remove any remaining noise while preserving edges
        enhanced_bgr = cv2.bilateralFilter(enhanced_bgr, 5, 30, 30)
        
        # Step 11: Apply deconvolution-like effect to further enhance clarity (simplified)
        # Create a sharpening kernel that mimics deconvolution
        kernel_deconv = np.array([[-0.1, -0.15, -0.1],
                                  [-0.15, 2.0, -0.15],
                                  [-0.1, -0.15, -0.1]])
        deconv_effect = cv2.filter2D(enhanced_bgr, -1, kernel_deconv)
        
        # Step 12: Apply local contrast enhancement for fine structures
        for c in range(3):  # Apply to each channel
            channel = enhanced_bgr[:,:,c]
            blurred = cv2.GaussianBlur(channel, (0, 0), 10)
            enhanced_bgr[:,:,c] = cv2.addWeighted(channel, 1.5, blurred, -0.5, 0)
        
        # Step 13: Apply final contrast normalization
        enhanced_bgr = normalize_contrast(enhanced_bgr)
        
        # Save the enhanced image
        cv2.imwrite(output_path, enhanced_bgr)
        
        # Validate if file was created
        if os.path.exists(output_path):
            print(f"Enhanced image saved to {output_path}")
            return True
        else:
            print(f"Error: Failed to save enhanced image to {output_path}")
            return False
            
    except Exception as e:
        print(f"Error in focus enhancement: {str(e)}")
        return False


#function to normalize the contrast of an image
def normalize_contrast(image):
    # Convert to HSV for better color preservation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply contrast normalization to the V channel
    # Calculate percentiles for more robust normalization
    p_low, p_high = np.percentile(v, (2, 98))
    
    # Scale the values to the full range
    v_norm = np.clip((v - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
    
    # Merge the channels back
    hsv_norm = cv2.merge([h, s, v_norm])
    
    # Convert back to BGR
    return cv2.cvtColor(hsv_norm, cv2.COLOR_HSV2BGR)


#function to measure the sharpness of an image
def measure_image_sharpness(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Failed to read image {image_path}")
            return -1
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient_mag)
        
        
        sharpness = laplacian_var * 0.5 + gradient_mean * 0.5
        
        return sharpness
        
    except Exception as e:
        print(f"Error in measuring sharpness: {str(e)}")
        return -1


#function for specialized enhancement on microscopic image
def enhance_microscope_image(input_path, output_path):
    
    try:
        # First apply the standard auto_focus
        success = auto_focus(input_path, output_path)
        
        if not success:
            return False
        
        # Now read the enhanced image for additional processing
        enhanced = cv2.imread(output_path)
        
        # Apply microscope-specific techniques:
        # 1. Deconvolution-like processing to enhance fine structures
        psf = np.ones((5, 5)) / 25  # Simple point spread function
        for i in range(3):
            enhanced[:,:,i] = deconvolution_filter(enhanced[:,:,i], psf)
        
        # 2. Local texture enhancement
        enhanced = enhance_texture(enhanced)
        
        # 3. Final adaptive sharpening
        enhanced = adaptive_sharpen(enhanced)
        
        # Save the final enhanced image
        cv2.imwrite(output_path, enhanced)
        
        return True
        
    except Exception as e:
        print(f"Error in microscope image enhancement: {str(e)}")
        return False

def deconvolution_filter(channel, psf):

    # Simple implementation for image clarity enhancement
    from scipy import signal
    
    # Apply a simplified deconvolution (single iteration)
    blurred = signal.convolve2d(channel, psf, mode='same')
    # Avoid division by zero
    blurred = np.maximum(blurred, 0.01)
    # Richardson-Lucy step
    ratio = channel / blurred
    estimate = channel * signal.convolve2d(ratio, psf, mode='same')
    
    # Convert back to uint8
    return np.clip(estimate, 0, 255).astype(np.uint8)


#function to enhance local texture details of image
def enhance_texture(image):
   
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance local texture in L channel using local contrast enhancement
    local_mean = cv2.GaussianBlur(l, (0, 0), 10)
    local_detail = cv2.subtract(l, local_mean)
    enhanced_detail = cv2.multiply(local_detail, 1.5)
    enhanced_l = cv2.add(local_mean, enhanced_detail)
    
    # Merge back and convert to BGR
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


#function to add adaptive brightness based on local contrast
def adaptive_sharpen(image):
 
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate local standard deviation (measure of local contrast)
    mean, std_dev = cv2.meanStdDev(gray)
    
    # Create an adaptive kernel strength based on local contrast
    kernel_strength = 1.0 + (std_dev[0][0] / 128.0)
    
    # Create adaptive sharpening kernel
    kernel = np.array([[-0.1, -0.1, -0.1],
                        [-0.1, 1.0 + kernel_strength, -0.1],
                        [-0.1, -0.1, -0.1]])
    
    # Apply the kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Blend with original based on local contrast
    alpha = min(0.7, std_dev[0][0] / 255.0)
    return cv2.addWeighted(image, 1-alpha, sharpened, alpha, 0)