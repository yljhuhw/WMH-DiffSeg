"""
Inference script for WMH-DiffSeg model
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.out_dir)

    logger.log("creating WMH model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load checkpoint
    logger.log(f"loading model from {args.model_path}")
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Get list of test images
    test_dir = os.path.join(args.data_dir, 'test', 'images')
    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    logger.log(f"processing {len(image_files)} images...")

    for img_file in tqdm(image_files):
        # Load and preprocess image
        img_path = os.path.join(test_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((args.image_size, args.image_size))
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(dist_util.dev())

        # Run inference
        with torch.no_grad():
            # Start from random noise
            noise = torch.randn_like(image_tensor[:, :1, :, :])

            # Denoise using DDPM or DPM-Solver
            sample = diffusion.p_sample_loop(
                model,
                (1, 1, args.image_size, args.image_size),
                noise=noise,
                clip_denoised=True,
                model_kwargs={"image": image_tensor},
                progress=False,
            )

        # Post-process and save
        pred_mask = (sample[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        output_path = os.path.join(args.out_dir, img_file)
        Image.fromarray(pred_mask).save(output_path)

    logger.log(f"inference complete. Results saved to {args.out_dir}")


def create_argparser():
    defaults = dict(
        data_dir="",
        model_path="",
        out_dir="./predictions",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        timestep_respacing="50",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
