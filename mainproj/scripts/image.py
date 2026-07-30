import os

from dotenv import load_dotenv

from src.image_generation import ImageGenerator


def main():

    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found. Please add it to your .env file."
        )

    generator = ImageGenerator(
        model_name="segmind/tiny-sd",
        hf_token=hf_token,
        output_dir="Artifacts/image_generation",
    )

    print("\n========================================")
    print(" Stable Diffusion Image Generator")
    print(" Type 'exit' to quit")
    print("========================================")

    while True:

        prompt = input("\nEnter Prompt: ").strip()

        if prompt.lower() == "exit":
            print("\nExiting...")
            break

        if not prompt:
            print("Prompt cannot be empty.")
            continue

        image_path = generator.generate_image(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, watermark, text",
            num_inference_steps=20,
            guidance_scale=7.5,
        )

        print("\n✅ Image generated successfully!")
        print(f"📁 Saved at: {image_path}")

        return image_path


if __name__ == "__main__":
    main()