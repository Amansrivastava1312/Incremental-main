from __future__ import annotations

import os
import sys
from pathlib import Path


# This file is in ``mainproj/scripts``; the project root is its parent.
PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

MODEL_PATH = PROJECT_DIR / "model" / "best.pt"
IMAGES_DIR = PROJECT_DIR / "images"
ARTIFACTS_DIR = PROJECT_DIR / "Artifacts"


def _relative_path(path: Path) -> str:
    """Return a user-friendly path relative to the project directory."""
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _choose_image() -> Path | None:
    """Choose an image; the detector stores external uploads in ``images``."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ) if IMAGES_DIR.exists() else []

    if samples:
        print("\nAvailable sample images:")
        for number, image_path in enumerate(samples, start=1):
            print(f"  {number}. {_relative_path(image_path)}")

    answer = input("\nImage number or image path (Enter to cancel): ").strip()
    if not answer:
        return None

    if answer.isdigit() and 1 <= int(answer) <= len(samples):
        return samples[int(answer) - 1]

    image_path = Path(answer).expanduser()
    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path

    if not image_path.is_file():
        print(f"\nImage not found: {image_path}")
        return None
    if image_path.suffix.lower() not in image_extensions:
        print("\nPlease select a supported image file.")
        return None

    return image_path


def run_detection() -> None:
    """Run fashion-object detection for a selected image."""
    if not MODEL_PATH.is_file():
        print(f"\nModel file not found: {_relative_path(MODEL_PATH)}")
        return

    image_path = _choose_image()
    if image_path is None:
        return

    try:
        from mainproj.src.detector import FashionDetector

        print("\nLoading detector and analysing image...")
        detector = FashionDetector(str(MODEL_PATH))
        _, summary, output_image, json_file = detector.detect_objects(
            image_path,
            save_dir=str(ARTIFACTS_DIR / "detection"),
            images_dir=IMAGES_DIR,
        )
    except ImportError as error:
        print(f"\nA detection dependency is missing: {error}")
        print("Install requirements with: pip install -r requirements.txt")
        return
    except Exception as error:
        print(f"\nDetection failed: {error}")
        return

    print("\n========== Detection Summary ==========")
    print(f"Image: {_relative_path(Path(summary['source_image_path']))}")
    print(f"Objects detected: {summary['total_objects']}")
    if summary["object_counts"]:
        print("Object counts:")
        for name, count in summary["object_counts"].items():
            print(f"  - {name}: {count}")
    else:
        print("No objects detected.")
    print(f"Annotated image: {_relative_path(Path(output_image))}")
    print(f"Detection data: {_relative_path(Path(json_file))}")

    paths = {"Image" : _relative_path(Path(output_image)), 
             "Json" : _relative_path(Path(json_file))}
    
    return paths


def run_image_generation() -> None:
    """Generate an image from a text prompt."""
    prompt = input("\nDescribe the image to generate (Enter to cancel): ").strip()
    if not prompt:
        return

    try:
        from dotenv import load_dotenv
        from mainproj.src.image_generation import ImageGenerator

        env_path = PROJECT_DIR / ".env"
        load_dotenv(env_path if env_path.is_file() else ROOT_DIR / ".env")
        token = os.getenv("HF_TOKEN")
        print("\nLoading image-generation model. This can take a moment...")
        generator = ImageGenerator(
            hf_token=token,
            output_dir=str(ARTIFACTS_DIR / "image_generation"),
        )
        output_path = generator.generate_image(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, watermark, text",
        )
        print(f"\nImage generated successfully: {_relative_path(Path(output_path))}")
    except ImportError as error:
        print(f"\nAn image-generation dependency is missing: {error}")
        print("Install requirements with: pip install -r requirements.txt")
    except Exception as error:
        print(f"\nImage generation failed: {error}")

    return _relative_path(Path(output_path))


def run_audio_generation() -> None:
    """Convert entered text to an MP3 file."""
    speech = input("\nEnter text to convert to speech (Enter to cancel): ").strip()
    if not speech:
        return

    try:
        from mainproj.src.audio_generation import AudioGenerator

        generator = AudioGenerator(ARTIFACTS_DIR / "audio_generation")
        output_path = generator.generate_audio(speech)
        print(f"\nAudio generated successfully: {_relative_path(Path(output_path))}")
    except ImportError as error:
        print(f"\nAn audio-generation dependency is missing: {error}")
        print("Install requirements with: pip install -r requirements.txt")
    except Exception as error:
        print(f"\nAudio generation failed: {error}")

    _relative_path(Path(output_path))
