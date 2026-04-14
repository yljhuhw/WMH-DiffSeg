"""
Inference script for TransUNet model
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.transunet import TransUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default='./predictions')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--model_size', type=str, default='tiny', choices=['tiny', 'small', 'base'])
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print(f"Creating TransUNet ({args.model_size})...")
    model = TransUNet(
        img_size=args.image_size,
        in_channels=3,
        out_channels=1,
        model_size=args.model_size
    ).to(device)

    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Get list of test images
    test_dir = os.path.join(args.data_dir, 'test', 'images')
    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(image_files)} images...")

    for img_file in tqdm(image_files):
        # Load and preprocess image
        img_path = os.path.join(test_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((args.image_size, args.image_size))
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output)

        # Post-process and save
        pred_mask = (pred_mask[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        output_path = os.path.join(args.out_dir, img_file)
        Image.fromarray(pred_mask).save(output_path)

    print(f"Inference complete. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
