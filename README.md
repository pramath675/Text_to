#!/usr/bin/env python3
"""
text2image_agent.py

Small CLI "agent" that converts a text prompt into one or more images using
Stable Diffusion (diffusers). Requires HUGGINGFACE_TOKEN in the environment.

Usage example:
  HUGGINGFACE_TOKEN="hf_xxx" python text2image_agent.py \
    --prompt "A serene futuristic city at sunset, cinematic, ultra-detailed" \
    --num_images 3 --seed 42
"""

import os
import argparse
import math
from typing import List
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# -------------------------
# Simple safety / content guard (minimal — replace with policy checks if needed)
# -------------------------
DISALLOWED_TERMS = {
    # extremely minimal example; expand or replace with a robust moderation API in production
    "child", "underage", "nsfw", "porn", "sexual", "illegal", "gore", "torture"
}

def contains_disallowed(prompt: str) -> bool:
    s = prompt.lower()
    for t in DISALLOWED_TERMS:
        if t in s:
            return True
    return False

# -------------------------
# Agent core
# -------------------------
def load_pipeline(model_id: str, device: str, use_fp16: bool = True):
    """Load Stable Diffusion pipeline with a fast scheduler."""
    torch_dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32
    # Use a modern solver scheduler for quality/speed trade-off
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        safety_checker=None,  # we handle basic safety above; for production use HF safety_checker or moderation API
        torch_dtype=torch_dtype,
        revision="fp16" if torch_dtype==torch.float16 else None,
    )
    pipe = pipe.to(device)
    # optimization for faster generation (if GPU)
    if device.startswith("cuda"):
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None
    return pipe

def generate_images(
    pipe,
    prompt: str,
    num_images: int = 1,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 7.5,
    steps: int = 25,
    seed: int | None = None,
    negative_prompt: str | None = None,
    out_dir: str = "outputs",
) -> List[str]:
    """Generate images and return list of saved file paths."""
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    for i in range(num_images):
        img = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]

        # make filename deterministic if seed provided
        seed_part = f"_s{seed}" if seed is not None else ""
        fname = f"img_{i+1}{seed_part}.png"
        path = Path(out_dir) / fname
        img.save(path)
        saved_paths.append(str(path.resolve()))

        # If a seed is provided, increment for variety across images
        if generator is not None:
            seed = (seed + 1) if seed is not None else None
            if seed is not None:
                generator = torch.Generator(device=pipe.device).manual_seed(seed)

    return saved_paths

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Text → Image agent using Stable Diffusion (diffusers).")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt to generate image from.")
    p.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt (things to avoid).")
    p.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model ID.")
    p.add_argument("--num_images", type=int, default=1, help="Number of images to generate.")
    p.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8).")
    p.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8).")
    p.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    p.add_argument("--steps", type=int, default=25, help="Inference steps (more -> better quality, slower).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Directory to save images.")
    p.add_argument("--no_fp16", action="store_true", help="Disable FP16 (useful if model has no fp16 revision).")
    return p.parse_args()

def main():
    args = parse_args()

    # basic safety check
    if contains_disallowed(args.prompt) or (args.negative_prompt and contains_disallowed(args.negative_prompt)):
        print("ERROR: Prompt appears to contain disallowed content. Aborting.")
        return

    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: Please set HUGGINGFACE_TOKEN environment variable with a valid token from Hugging Face.")
        return

    # set HF token so diffusers can use private models if necessary
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    # device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        pipe = load_pipeline(args.model, device=device, use_fp16=not args.no_fp16)
    except Exception as e:
        print("Failed to load model pipeline:", e)
        return

    print("Generating image(s)...")
    paths = generate_images(
        pipe,
        prompt=args.prompt,
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        out_dir=args.out_dir,
    )

    print("Done. Saved images:")
    for pth in paths:
        print("  -", pth)

if __name__ == "__main__":
    main()
