import argparse
import sys
import numpy as np
import cv2
from ultils import load_image, save_image, display_results
from transformations import rotation_90, rotation_180, rotation_270, image_elargement_replication_2_factor, image_elargement_replication_4_factor, image_elargement_replication
from filters import pencil_sketch


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
                            "pencil_sketch"
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

