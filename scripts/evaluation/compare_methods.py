"""
Compare results from multiple methods (MedSegDiff, WMH, TransUNet)
"""
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_metrics(json_file):
    """Load metrics from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['average']


def plot_comparison(metrics_dict, output_path):
    """
    Create comparison plot for multiple methods

    Args:
        metrics_dict: dict of {method_name: metrics_dict}
        output_path: path to save the plot
    """
    methods = list(metrics_dict.keys())
    metric_names = ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'accuracy']

    # Prepare data
    data = {metric: [metrics_dict[method][metric] for method in methods] for metric in metric_names}

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metric_names))
    width = 0.25

    for i, method in enumerate(methods):
        values = [data[metric][i] for metric in metric_names]
        ax.bar(x + i * width, values, width, label=method)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Segmentation Methods')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")


def print_comparison_table(metrics_dict):
    """Print comparison table"""
    methods = list(metrics_dict.keys())
    metric_names = ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'accuracy']

    # Print header
    print("\n" + "="*80)
    print(f"{'Method':<20}", end='')
    for metric in metric_names:
        print(f"{metric.capitalize():<12}", end='')
    print()
    print("="*80)

    # Print data
    for method in methods:
        print(f"{method:<20}", end='')
        for metric in metric_names:
            value = metrics_dict[method][metric]
            print(f"{value:<12.4f}", end='')
        print()

    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--medsegdiff', type=str, help='Path to MedSegDiff metrics JSON')
    parser.add_argument('--wmh', type=str, help='Path to WMH metrics JSON')
    parser.add_argument('--transunet', type=str, help='Path to TransUNet metrics JSON')
    parser.add_argument('--output', type=str, default='comparison.png', help='Output plot path')
    args = parser.parse_args()

    metrics_dict = {}

    if args.medsegdiff and os.path.exists(args.medsegdiff):
        metrics_dict['MedSegDiff'] = load_metrics(args.medsegdiff)

    if args.wmh and os.path.exists(args.wmh):
        metrics_dict['WMH-DiffSeg'] = load_metrics(args.wmh)

    if args.transunet and os.path.exists(args.transunet):
        metrics_dict['TransUNet'] = load_metrics(args.transunet)

    if len(metrics_dict) == 0:
        print("No valid metrics files provided")
        return

    # Print comparison table
    print_comparison_table(metrics_dict)

    # Create comparison plot
    plot_comparison(metrics_dict, args.output)


if __name__ == "__main__":
    main()
