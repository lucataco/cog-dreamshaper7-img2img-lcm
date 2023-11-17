# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

MODLE_CACHE = "model-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            MODLE_CACHE
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Astronauts in a jungle, cold color palette, muted colors, detailed, 8k",
        ),
        image: Path = Input(description="Input image"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=20, default=4
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=10, default=1
        ),
        strength: float = Input(
            description="Prompt strength, 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        init_image = Image.open(image)

        image = self.pipe(
            prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator
        ).images[0]

        image = image.convert("RGB")
        output_path = "/tmp/output.jpg"
        image.save(output_path)

        return Path(output_path)
