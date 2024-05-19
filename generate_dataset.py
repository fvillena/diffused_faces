from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from diffusers import DiffusionPipeline
import torch
from FFHQFaceAlignment.align import *
import argparse
import sys

sys.path.append("/workspaces/diffused_faces")

parser = argparse.ArgumentParser(
    description="Generate a dataset of real and synthetic faces"
)

parser.add_argument(
    "--input_dir",
    "-i",
    type=Path,
    default="images1024x1024",
    help="Directory containing real images",
    required=False,
)

parser.add_argument(
    "--output_dir",
    "-o",
    type=Path,
    default="images",
    help="Output directory for image pairs",
    required=False,
)

parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="sdxl",
    help="Model to use for generation",
    required=False,
)

args = parser.parse_args()

if args.model == "sdxl":
    generator_base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    generator_refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=generator_base.text_encoder_2,
        vae=generator_base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    def generate_image(prompt):
        image = generator_base(
            prompt=prompt,
            num_inference_steps=60,
            output_type="latent",
        ).images[0]
        image = generator_refiner(
            prompt=prompt,
            num_inference_steps=60,
            image=image,
        ).images[0]
        return image

elif args.model == "realvisxl":
    from diffusers import DPMSolverMultistepScheduler

    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    def generate_image(prompt):
        image = pipe(
            prompt=prompt,
            negative_prompt="face asymmetry, eyes asymmetry, deformed eyes",
        ).images[0]
        return image

else:
    raise ValueError("Invalid model")


captioner_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
captioner = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b", torch_dtype=torch.float32
)
captioner.to("cpu")

le = LandmarksEstimation(type="2D")


def caption_image(image):
    inputs = captioner_processor(image, return_tensors="pt").to("cpu", torch.float32)
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


def align_and_crop(image):
    image = np.array(image)
    img_tensor = torch.tensor(np.transpose(image, (2, 0, 1))).float().cuda()
    with torch.no_grad():
        landmarks, detected_faces = le.detect_landmarks(
            img_tensor.unsqueeze(0), detected_faces=None
        )
    if len(landmarks) > 0:
        image = align_crop_image(
            image=image,
            landmarks=np.asarray(landmarks[0].detach().cpu().numpy()),
            transform_size=512,
        )
    else:
        raise ValueError("No landmarks found in image")
    image = Image.fromarray(image)
    return image


image_paths = list(args.input_dir.rglob("*.png"))
for image_path in image_paths:
    image = Image.open(image_path)
    prompt = caption_image(image) + ", color headshot portrait photo "
    print(f"IMAGE: {image_path.stem}")
    print(f"PROMPT: {prompt}")
    synthetic_image = generate_image(prompt)
    max_attempts = 5
    attempts = 0
    while True:
        if attempts >= max_attempts:
            print("WARNING: Maximum attempts reached")
            continue
        try:
            synthetic_image_cropped = align_and_crop(synthetic_image)
            break
        except ValueError as e:
            print(f"WARNING: {e}")
            synthetic_image = generate_image(prompt)
            attempts += 1
    if attempts >= max_attempts:
        continue
    synthetic_image_cropped_exif = synthetic_image_cropped.getexif()
    synthetic_image_cropped_exif[0x9286] = prompt
    synthetic_image_cropped.save(
        args.output_dir / f"{image_path.stem}_synthetic.jpg",
        format="JPEG",
        quality=75,
        exif=synthetic_image_cropped_exif,
    )
    image.save(
        args.output_dir / f"{image_path.stem}_original.jpg",
        format="JPEG",
        quality=75,
    )
    # synthetic_image.save(
    #     args.output_dir / f"{image_path.stem}_synthetic_nocrop.jpg",
    #     format="JPEG",
    #     quality=75,
    # )
