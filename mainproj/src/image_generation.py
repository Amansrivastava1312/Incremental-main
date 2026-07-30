import os
import time
import torch

from diffusers import StableDiffusionPipeline
from huggingface_hub import login

class ImageGenerator:
    def __init__(
        self,
        model_name="segmind/tiny-sd",
        hf_token=None,
        output_dir="Artifacts/image_generation",
        device=None,
    ):
        """
        Initialize Stable Diffusion Pipeline.

        Args:
            model_name (str): Hugging Face model ID.
            hf_token (str): Hugging Face access token.
            output_dir (str): Directory to save generated images.
            device (str): "cuda" or "cpu". Auto-selected if None.
        """

        if hf_token:
            login(token=hf_token)

        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print("=" * 60)
        print("Loading Stable Diffusion Model")
        print(f"Model  : {model_name}")
        print(f"Device : {self.device}")
        print(f"Dtype  : {self.dtype}")
        print("=" * 60)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            use_safetensors=False,
        )

        self.pipe = self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✓ xFormers enabled")
            except Exception:
                print("xFormers not installed. Skipping.")

        print("\nModel loaded successfully.\n")

    def generate_image(
        self,
        prompt,
        negative_prompt=None,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=None,
    ):
        """
        Generate image from prompt.

        Returns:
            output_path (str)
        """

        generator = None

        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        filename = f"generated_{int(time.time())}.png"

        output_path = os.path.join(
            self.output_dir,
            filename,
        )

        image.save(output_path)

        return output_path