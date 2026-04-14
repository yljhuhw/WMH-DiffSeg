"""
Training script for TransUNet baseline
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.transunet import TransUNet
from guided_diffusion.custom_dataset_loader import CustomDataset


def dice_loss(pred, target, smooth=1e-5):
    """Dice loss for binary segmentation"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target):
    """Combined BCE + Dice loss"""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch[0].to(device)
        masks = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch[0].to(device)
            masks = batch[1].to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=['DRIVE', 'CHASEDB1', 'ISIC'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--model_size', type=str, default='tiny', choices=['tiny', 'small', 'base'])
    parser.add_argument('--out_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

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

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Create datasets
    print("Loading datasets...")
    train_dataset = CustomDataset(
        data_dir=args.data_dir,
        split='train',
        image_size=args.image_size
    )
    val_dataset = CustomDataset(
        data_dir=args.data_dir,
        split='test',
        image_size=args.image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        # Save latest
        torch.save(checkpoint, os.path.join(args.out_dir, f'transunet_{args.dataset}_latest.pt'))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.out_dir, f'transunet_{args.dataset}_best.pt'))
            print(f"Saved best model with val_loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
