#!/usr/bin/env python3
"""
Convert PNG mask file to NPY format similar to img_0_seg.npy
"""

import numpy as np
from PIL import Image
import os
from datetime import datetime

def convert_mask_to_npy(png_path, output_path=None):
    """
    Convert PNG mask to NPY format with similar structure to img_0_seg.npy
    
    Args:
        png_path (str): Path to input PNG mask file
        output_path (str): Path for output NPY file (optional)
    
    Returns:
        str: Path to created NPY file
    """
    
    # Load the PNG image
    print(f"Loading PNG mask from: {png_path}")
    img = Image.open(png_path)
    
    # Convert to numpy array
    mask_array = np.array(img)
    print(f"Mask shape: {mask_array.shape}")
    print(f"Mask dtype: {mask_array.dtype}")
    print(f"Unique values: {np.unique(mask_array)}")
    
    # Ensure binary mask (0 and 255)
    if len(np.unique(mask_array)) == 2:
        print("Binary mask detected")
        # Convert to binary (0 and 1)
        binary_mask = (mask_array > 0).astype(np.uint8)
    else:
        print("Non-binary mask detected, converting to binary")
        binary_mask = (mask_array > 0).astype(np.uint8)
    
    # Find contours (boundaries) of the mask
    from skimage import measure
    
    # Find contours
    contours = measure.find_contours(binary_mask, 0.5)
    print(f"Found {len(contours)} contours")
    
    # Create the annotation structure similar to img_0_seg.npy
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    
    # Create annotation record
    annotation = [
        current_time,  # timestamp
        "converted from PNG",  # action description
        [],  # x coordinate arrays will go here
        []   # y coordinate arrays will go here
    ]
    
    # Add coordinate arrays for each contour
    for i, contour in enumerate(contours):
        # Extract x and y coordinates
        y_coords = contour[:, 0].astype(int)  # row coordinates
        x_coords = contour[:, 1].astype(int)  # column coordinates
        
        # Add to annotation
        annotation[2].append(x_coords)
        annotation[3].append(y_coords)
        
        print(f"Contour {i+1}: {len(x_coords)} points")
    
    # Create the final array structure
    final_array = np.array([annotation], dtype=object)
    
    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(png_path))[0]
        output_path = f"{base_name}_seg.npy"
    
    # Save as NPY file
    print(f"Saving NPY file to: {output_path}")
    np.save(output_path, final_array, allow_pickle=True)
    
    print(f"Conversion complete! Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    
    return output_path

def main():
    """Main function to run the conversion"""
    
    # Input PNG file paths
    png_paths = [
        "samples/ROI 1_1 mask.png",
        "samples/ROI 1_10.png", 
        "samples/ROI 1_20.png",
        "samples/ROI 1_62.png"
    ]
    
    # Convert each mask
    for png_path in png_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {png_path}")
        print(f"{'='*60}")
        
        # Check if file exists
        if not os.path.exists(png_path):
            print(f"Error: File not found: {png_path}")
            continue
        
        # Convert the mask
        try:
            output_file = convert_mask_to_npy(png_path)
            print(f"\nSuccessfully converted {png_path} to {output_file}")
            
            # Verify the created file
            print("\nVerifying created NPY file:")
            loaded_data = np.load(output_file, allow_pickle=True)
            print(f"Loaded array shape: {loaded_data.shape}")
            print(f"Loaded array dtype: {loaded_data.dtype}")
            print(f"Number of annotations: {len(loaded_data)}")
            
            if len(loaded_data) > 0:
                first_annotation = loaded_data[0]
                print(f"Timestamp: {first_annotation[0]}")
                print(f"Action: {first_annotation[1]}")
                print(f"Number of contours: {len(first_annotation[2])}")
                
        except Exception as e:
            print(f"Error during conversion of {png_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All conversions completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 