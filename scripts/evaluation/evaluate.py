"""
Evaluation script for segmentation results
Computes Dice, IoU, Sensitivity, Specificity, Precision, and Accuracy
"""
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


def compute_metrics(pred, gt, smooth=1e-5):
    """
    Compute segmentation metrics

    Args:
        pred: predicted mask (binary, 0 or 1)
        gt: ground truth mask (binary, 0 or 1)
        smooth: smoothing factor to avoid division by zero

    Returns:
        dict with metrics: dice, iou, sensitivity, specificity, precision, accuracy
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # True Positive, False Positive, True Negative, False Negative
    TP = np.sum(pred & gt)
    FP = np.sum(pred & ~gt)
    TN = np.sum(~pred & ~gt)
    FN = np.sum(~pred & gt)

    # Dice Score
    dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)

    # IoU (Jaccard Index)
    iou = (TP + smooth) / (TP + FP + FN + smooth)

    # Sensitivity (Recall, True Positive Rate)
    sensitivity = (TP + smooth) / (TP + FN + smooth)

    # Specificity (True Negative Rate)
    specificity = (TN + smooth) / (TN + FP + smooth)

    # Precision (Positive Predictive Value)
    precision = (TP + smooth) / (TP + FP + smooth)

    # Accuracy
    accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)

    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predicted masks')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with ground truth masks')
    parser.add_argument('--out_file', type=str, default='metrics.json', help='Output JSON file')
    args = parser.parse_args()

    # Get list of prediction files
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(pred_files) == 0:
        print(f"No prediction files found in {args.pred_dir}")
        return

    print(f"Evaluating {len(pred_files)} images...")

    all_metrics = []

    for pred_file in tqdm(pred_files):
        # Load prediction
        pred_path = os.path.join(args.pred_dir, pred_file)
        pred = np.array(Image.open(pred_path).convert('L'))
        pred = (pred > 127).astype(np.uint8)

        # Load ground truth
        gt_path = os.path.join(args.gt_dir, pred_file)
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {pred_file}")
            continue

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = (gt > 127).astype(np.uint8)

        # Compute metrics
        metrics = compute_metrics(pred, gt)
        metrics['filename'] = pred_file
        all_metrics.append(metrics)

    # Compute average metrics
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
    }

    # Print results
    print("\n" + "="*50)
    print("Average Metrics:")
    print("="*50)
    for key, value in avg_metrics.items():
        print(f"{key:15s}: {value:.4f}")
    print("="*50)

    # Save results
    results = {
        'average': avg_metrics,
        'per_image': all_metrics
    }

    with open(args.out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.out_file}")


if __name__ == "__main__":
    main()
