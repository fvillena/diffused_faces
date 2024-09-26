from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import argparse
import sys
from simplejsondb import Database

sys.path.append("/workspaces/diffused_faces")

parser = argparse.ArgumentParser(description="Caption images")

parser.add_argument(
    "--input_dir",
    "-i",
    type=Path,
    default="selected_faces/",
    help="Directory containing images",
    required=False,
)

parser.add_argument(
    "--output_file",
    "-o",
    type=Path,
    default="captions.json",
    help="Output file for captions",
    required=False,
)

parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="Salesforce/blip2-opt-6.7b",
    help="Model to use for captioning",
    required=False,
)

args = parser.parse_args()

captioner_processor = AutoProcessor.from_pretrained(args.model)
captioner = Blip2ForConditionalGeneration.from_pretrained(
    args.model, device_map="auto", torch_dtype=torch.float16
)


def caption_image(image):
    inputs = captioner_processor(image, return_tensors="pt").to("cuda", torch.float16)
    generated_ids = captioner.generate(
        **inputs,
        min_length=1,
        max_length=128,
        do_sample=False,
        num_beams=3,
    )
    generated_text = captioner_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    return generated_text


captions = Database("captions.json", default=dict(), save_at_exit=False)

image_paths = list(args.input_dir.rglob("*.png"))
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    caption = caption_image(image)
    captions.data[image_path.stem] = caption
    captions.save(indent=4)
