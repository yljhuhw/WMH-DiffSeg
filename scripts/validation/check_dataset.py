"""
Validate dataset structure and format
"""
import argparse
import os
from PIL import Image
import numpy as np


def check_dataset(data_dir):
    """
    Validate dataset structure and format

    Args:
        data_dir: path to dataset directory
    """
    print(f"Checking dataset: {data_dir}\n")

    errors = []
    warnings = []

    # Check directory structure
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            errors.append(f"Missing {split} directory")
            continue

        images_dir = os.path.join(split_dir, 'images')
        masks_dir = os.path.join(split_dir, 'masks')

        if not os.path.exists(images_dir):
            errors.append(f"Missing {split}/images directory")
        if not os.path.exists(masks_dir):
            errors.append(f"Missing {split}/masks directory")

        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            # Get file lists
            image_files = set([f for f in os.listdir(images_dir)
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
            mask_files = set([f for f in os.listdir(masks_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))])

            # Check counts
            print(f"✓ Found {len(image_files)} {split} images")
            print(f"✓ Found {len(mask_files)} {split} masks")

            # Check filename matching
            if image_files != mask_files:
                missing_masks = image_files - mask_files
                missing_images = mask_files - image_files

                if missing_masks:
                    errors.append(f"{split}: Missing masks for images: {missing_masks}")
                if missing_images:
                    errors.append(f"{split}: Missing images for masks: {missing_images}")
            else:
                print(f"✓ All {split} filenames match")

            # Check mask format (sample a few)
            sample_size = min(5, len(mask_files))
            non_binary_masks = []

            for mask_file in list(mask_files)[:sample_size]:
                mask_path = os.path.join(masks_dir, mask_file)
                try:
                    mask = np.array(Image.open(mask_path).convert('L'))
                    unique_values = np.unique(mask)

                    # Check if binary (only 0 and 255)
                    if not (len(unique_values) <= 2 and
                           all(v in [0, 255] for v in unique_values)):
                        non_binary_masks.append(mask_file)
                except Exception as e:
                    errors.append(f"Error reading {mask_file}: {str(e)}")

            if non_binary_masks:
                warnings.append(f"{split}: Non-binary masks detected: {non_binary_masks}")
            else:
                print(f"✓ All sampled {split} masks are binary")

        print()

    # Print summary
    print("="*60)
    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  ✗ {error}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    if not errors and not warnings:
        print("✓ Dataset validation passed!")
    elif not errors:
        print("✓ Dataset structure is valid (with warnings)")
    else:
        print("✗ Dataset validation failed")

    print("="*60)

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    args = parser.parse_args()

    success = check_dataset(args.data_dir)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
