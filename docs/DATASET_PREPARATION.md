# Dataset Preparation Guide

This guide explains how to prepare datasets for training and evaluation with MedSegDiff, WMH-DiffSeg, and TransUNet.

## Dataset Structure

All datasets should follow this structure:

```
data/
├── DATASET_NAME/
│   ├── train/
│   │   ├── images/
│   │   │   ├── image_001.png
│   │   │   ├── image_002.png
│   │   │   └── ...
│   │   └── masks/
│   │       ├── image_001.png
│   │       ├── image_002.png
│   │       └── ...
│   └── test/
│       ├── images/
│       │   ├── image_001.png
│       │   ├── image_002.png
│       │   └── ...
│       └── masks/
│           ├── image_001.png
│           ├── image_002.png
│           └── ...
```

**Important Notes:**
- Image and mask filenames must match exactly
- Masks should be binary (0 for background, 255 for foreground)
- Images can be RGB or grayscale
- Supported formats: PNG, JPG, JPEG

## Supported Datasets

### 1. DRIVE (Retinal Vessel Segmentation)

**Download:** https://drive.grand-challenge.org/

**Statistics:**
- Training: 20 images
- Testing: 20 images
- Image size: 565×584 pixels
- Task: Blood vessel segmentation in retinal images

**Preparation Steps:**

1. Download the DRIVE dataset
2. Extract to `data/DRIVE/`
3. Organize as follows:

```bash
data/DRIVE/
├── train/
│   ├── images/          # 20 training images
│   └── masks/           # 20 manual segmentation masks
└── test/
    ├── images/          # 20 test images
    └── masks/           # 20 manual segmentation masks
```

4. Convert masks to binary format (if needed):

```python
from PIL import Image
import numpy as np

mask = Image.open('mask.png').convert('L')
mask_array = np.array(mask)
binary_mask = (mask_array > 127).astype(np.uint8) * 255
Image.fromarray(binary_mask).save('mask_binary.png')
```

### 2. CHASEDB1 (Retinal Vessel Segmentation)

**Download:** https://blogs.kingston.ac.uk/retinal/chasedb1/

**Statistics:**
- Training: 22 images
- Testing: 6 images
- Image size: 999×960 pixels
- Task: Blood vessel segmentation in retinal images

**Preparation Steps:**

1. Download the CHASEDB1 dataset
2. Extract to `data/CHASEDB1/`
3. Split into train/test (typically first 22 for training, last 6 for testing)
4. Organize following the standard structure

### 3. ISIC (Skin Lesion Segmentation)

**Download:** https://challenge.isic-archive.com/

**Statistics:**
- Training: 900 images
- Testing: 379 images
- Image size: Variable (will be resized to 256×256)
- Task: Skin lesion segmentation

**Preparation Steps:**

1. Download ISIC 2017 or 2018 dataset
2. Extract to `data/ISIC/`
3. Organize following the standard structure
4. Ensure masks are binary

## Data Preprocessing

### Image Resizing

All images will be automatically resized to 256×256 during training/inference. You can modify this in the training scripts:

```bash
python scripts/train/train_wmh.py --image_size 256
```

### Data Augmentation

Training scripts automatically apply:
- Random horizontal flips
- Random rotations (optional)
- Color jittering (optional)

You can modify augmentation in `guided_diffusion/custom_dataset_loader.py`.

## Validation

After preparing your dataset, validate the structure:

```bash
# Check dataset structure
python scripts/validation/check_dataset.py --data_dir data/DRIVE

# Expected output:
# ✓ Found 20 training images
# ✓ Found 20 training masks
# ✓ Found 20 test images
# ✓ Found 20 test masks
# ✓ All filenames match
# ✓ All masks are binary
```

## Custom Datasets

To use your own dataset:

1. Organize following the standard structure
2. Ensure masks are binary (0 and 255)
3. Update the dataset loader if needed:

```python
# In guided_diffusion/custom_dataset_loader.py
class CustomDataset(Dataset):
    def __init__(self, data_dir, split='train', image_size=256):
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'masks')
        # ... rest of implementation
```

4. Train with your dataset:

```bash
python scripts/train/train_wmh.py \
    --data_dir data/YOUR_DATASET \
    --dataset YOUR_DATASET \
    --batch_size 4 \
    --epochs 100
```

## Common Issues

### Issue: Filenames don't match

**Solution:** Rename files to ensure image and mask filenames match exactly:

```bash
cd data/DATASET/train/masks
for f in *_mask.png; do mv "$f" "${f/_mask/}"; done
```

### Issue: Masks are not binary

**Solution:** Convert masks to binary format:

```python
import os
from PIL import Image
import numpy as np

mask_dir = 'data/DATASET/train/masks'
for filename in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, filename)
    mask = np.array(Image.open(mask_path).convert('L'))
    binary_mask = (mask > 127).astype(np.uint8) * 255
    Image.fromarray(binary_mask).save(mask_path)
```

### Issue: Images are different sizes

**Solution:** Images will be automatically resized during training. No action needed.

## Next Steps

After preparing your dataset:

1. **Train models:** See training scripts in `scripts/train/`
2. **Run inference:** See inference scripts in `scripts/inference/`
3. **Evaluate results:** See evaluation scripts in `scripts/evaluation/`

For more information, see the main [README.md](../README.md).
