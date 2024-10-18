from pathlib import Path
from PIL import Image
from diffusers import DiffusionPipeline
import torch
from FFHQFaceAlignment.align import *
import argparse
import sys
from simplejsondb import Database

sys.path.append("/workspaces/diffused_faces")

parser = argparse.ArgumentParser(description="Generate images")

parser.add_argument(
    "--output_dir",
    "-o",
    type=Path,
    default="generated_images/",
    help="Output directory for generated images",
    required=False,
)

parser.add_argument(
    "--captions",
    "-c",
    type=Path,
    default="captions.json",
    help="File containing captions",
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

parser.add_argument(
    "-n",
    type=int,
    default=1,
    help="Number of images to generate per caption",
    required=False,
)

parser.add_argument(
    "--start_from",
    type=str,
    default=None,
    help="Start from a specific filename",
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

le = LandmarksEstimation(type="2D")

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
            transform_size=1024,
        )
    else:
        raise ValueError("No landmarks found in image")
    image = Image.fromarray(image)
    return image

captions = Database(args.captions)

def generate_and_save_image(filename, prompt, n=1):
    for i in range(n):
        synthetic_image = generate_image(prompt)
        max_attempts = 5
        attempts = 0
        while True:
            if attempts >= max_attempts:
                print("WARNING: Maximum attempts reached")
                break
            try:
                synthetic_image_cropped = align_and_crop(synthetic_image)
                break
            except ValueError as e:
                print(f"WARNING: {e}")
                synthetic_image = generate_image(prompt)
                attempts += 1
        if attempts >= max_attempts:
            # synthetic_image_cropped is a black image of the same dimensions
            synthetic_image_cropped = Image.new("RGB", (1024, 1024), (0, 0, 0))
        synthetic_image_cropped_exif = synthetic_image_cropped.getexif()
        synthetic_image_cropped_exif[0x9286] = prompt
        if n == 1:
            synthetic_image_cropped.save(
                args.output_dir / f"{filename}_synthetic.png",
                format="PNG",
                exif=synthetic_image_cropped_exif,
            )
        else:
            synthetic_image_cropped.save(
                args.output_dir / f"{filename}_synthetic_{i}.png",
                format="PNG",
                exif=synthetic_image_cropped_exif,
            )

filenames = list(captions.data.keys())
if args.start_from:
    start_index = filenames.index(args.start_from)
    filenames = filenames[start_index:]

for filename in filenames:
    caption = captions.data[filename]
    #is the image is in the directory, then we skip it
    if (args.output_dir / f"{filename}_synthetic.png").exists():
        continue
    prompt = caption + ", color headshot portrait photo "
    print(f"IMAGE: {filename}")
    print(f"PROMPT: {prompt}")
    generate_and_save_image(filename, prompt, args.n)