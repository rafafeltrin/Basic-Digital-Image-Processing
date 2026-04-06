import argparse
import sys
import numpy as np
import cv2
from utils import load_image, save_image, display_results
from transformations import rotation_90, rotation_180, rotation_270, image_elargement_replication_2_factor, image_elargement_replication_4_factor, image_elargement_replication, bit_representation
from filters import combined_gradient_magnitude, mosaic, pencil_sketch, gamma_correction, spatial_convolution, threshold_binarization, sepia_filter, monochrome_filter, bit_planes, weighted_average_monochromatic_image, negative_filter, intensity_transformed, inverted_even_rows, mirror_top_half_to_bottom_half,vertical_mirror


def main():
    parser = argparse.ArgumentParser(description="MC920 - Trabalho 1")
    
    # Positional arguments for Input and Output paths
    parser.add_argument("input", help="Path to input PNG image")
    parser.add_argument("output", help="Path to save output PNG image")

    # Task selection argument
    parser.add_argument("--task", type=str, required=True,
                        choices=[
                            "rotation_90", 
                            "rotation_180", 
                            "rotation_270",
                            "elargement_replication_2_factor",
                            "elargement_replication_4_factor",
                            "elargement_replication",
                            "pencil_sketch",
                            "gamma_correction",
                            "threshold_binarization",
                            "sepia_filter",
                            "monochrome_filter",
                            "bit_planes",
                            "weighted_average",
                            "mosaic",
                            "intensity_transformation",
                            "bit_representation",
                            "spatial_convolution"
                        ],
                        help="Select the transformation or filter to apply")

    # Optional flag to display results
    parser.add_argument("--display", action="store_true", help="Display original and processed images side-by-side")
    
    # Optional flag for monochromatic mode
    parser.add_argument("--monochromatic", action="store_true", help="Load image in monochromatic mode (grayscale)")

    args = parser.parse_args()

    #Load the image (monochromatic as per spec 1.1)
    if args.monochromatic:
        img = load_image(args.input, monochromatic=True)
    else:
        img = load_image(args.input, monochromatic=False)
    
    
    # Execution logic
    if args.task == "rotation_90":
        result = rotation_90(img)
    elif args.task == "rotation_180":
        result = rotation_180(img)
    elif args.task == "rotation_270":
        result = rotation_270(img)
    elif args.task == "elargement_replication_2_factor":
        result = image_elargement_replication_2_factor(img)
    elif args.task == "elargement_replication_4_factor":
        result = image_elargement_replication_4_factor(img)
    elif args.task == "elargement_replication":
        factor = int(input("Enter the replication factor: "))
        result = image_elargement_replication(img, factor)
    elif args.task == "pencil_sketch":
        result = pencil_sketch(img)
    elif args.task == "gamma_correction":
        gamma = float(input("Enter the gamma value: "))
        result = gamma_correction(img, gamma)
    elif args.task == "threshold_binarization":
        threshold = int(input("Enter the threshold value (0-255): "))
        result = threshold_binarization(img, threshold)
    elif args.task == "sepia_filter":
        result = sepia_filter(img)
    elif args.task == "monochrome_filter":
        result = monochrome_filter(img)
    elif args.task == "bit_planes":
        plane = int(input("Enter the bit plane to extract (0-7): "))
        result = bit_planes(img, plane)
    elif args.task == "weighted_average":
        img2_path = input("Enter the path for the second image: ")
        img2 = load_image(img2_path, monochromatic=args.monochromatic)
        weight1 = float(input("Enter the weight for the first image (0-1): "))
        weight2 = float(input("Enter the weight for the second image (0-1): "))
        result = weighted_average_monochromatic_image(img, img2, weight1, weight2)
    elif args.task == "mosaic":
        professor_array = np.array([
            [6, 11, 13, 3],
            [8, 16,  1, 9],
            [12, 14, 2, 7],
            [4, 15, 10, 5]
        ]) - 1
        result = mosaic(img, professor_array)
    elif args.task == "intensity_transformation":
        negative = negative_filter(img)
        save_image(negative, "output", f"{args.output}negative.png")

        intensity_transformed_image = intensity_transformed(img)
        save_image(intensity_transformed_image, "output", f"{args.output[:-4]}intensity_transformed.png")

        inverted_even_rows_image = inverted_even_rows(img)
        save_image(inverted_even_rows_image, "output", f"{args.output[:-4]}inverted_even_rows.png")

        mirror_top_half_to_bottom_half_image = mirror_top_half_to_bottom_half(img)
        save_image(mirror_top_half_to_bottom_half_image, "output", f"{args.output[:-4]}mirror_top_half.png")

        vertical_mirror_image = vertical_mirror(img)
        save_image(vertical_mirror_image, "output", f"{args.output[:-4]}vertical_mirror.png")

        result = vertical_mirror_image  # Just to have a final result to display
    elif args.task == "bit_representation":
        original_bit_depth = int(input("Enter the original bit depth (e.g., 8): "))
        final_bit_depth = int(input("Enter the desired bit depth (e.g., 4): "))
        result = bit_representation(img, original_bit_depth, final_bit_depth)
    elif args.task == "spatial_convolution":
        mask_to_use = int(input("Enter the mask number to use (1-12): "))

        h1_mask = np.array([[ 0,  0, -1,  0,  0],
                            [ 0, -1, -2, -1,  0],
                            [-1, -2, 16, -2, -1],
                            [ 0, -1, -2, -1,  0],
                            [ 0,  0, -1,  0,  0]], dtype=np.float32)


        h2_mask = (1.0 / 256.0) * np.array([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
            ], dtype=np.float32)

        h3_mask = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        h4_mask = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float32)

        h5_mask = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]], dtype=np.float32)

        h6_mask = (1.0 / 9.0) * np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.float32)

        h7_mask = np.array([[-1, -1,  2],
                            [-1,  2, -1],
                            [ 2, -1, -1]], dtype=np.float32)

        h8_mask = np.array([[ 2, -1, -1],
                            [-1,  2, -1],
                            [-1, -1,  2]], dtype=np.float32)

        # h9: 9x9 Identity matrix divided by 9. 
        h9_mask = (1.0 / 9.0) * np.eye(9, dtype=np.float32)

        # h10: 5x5 High-pass/Sharpening mask. Multiplied by 1/8.
        h10_mask = (1.0 / 8.0) * np.array([
            [-1, -1, -1, -1, -1],
            [-1,  2,  2,  2, -1],
            [-1,  2,  8,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]
        ], dtype=np.float32)

        h11_mask = np.array([[-1, -1, 0],
                            [-1,  0, 1],
                            [ 0,  1, 1]], dtype=np.float32)
        
        masks = {
            1: h1_mask,
            2: h2_mask,
            3: h3_mask,
            4: h4_mask,
            5: h5_mask,
            6: h6_mask,
            7: h7_mask,
            8: h8_mask,
            9: h9_mask,
            10: h10_mask,
            11: h11_mask
        }
        if mask_to_use in masks:
            selected_mask = masks[mask_to_use]
            result = spatial_convolution(img, selected_mask)
        elif mask_to_use == 12:
            result = combined_gradient_magnitude(img, h3_mask, h4_mask)
        else:
            print("Invalid mask number. Please enter a number between 1 and 11.")
            sys.exit(1)
    else:
        print("Invalid task.")
        sys.exit(1)

    # Save and optionally display
    save_image(result, "output", args.output)
    if args.display:
        display_results(img, result, title=args.task)
    
    print("Processing completed successfully.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
if __name__ == "__main__":
    main()

