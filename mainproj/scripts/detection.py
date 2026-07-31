from pathlib import Path

from mainproj.src.detector import FashionDetector


def main(image_path):

    project_dir = Path(__file__).resolve().parent.parent
    detector = FashionDetector(project_dir / "model" / "best.pt")

    # image_path = input("Enter image path: ").strip()

    annotated_image, summary, output_image_path, json_path = detector.detect_objects(
        image_path,
        save_dir=project_dir / "Artifacts" / "detection",
        images_dir=project_dir / "images",
    )

    print("\n========== Detection Summary ==========")

    print(f"Image Name      : {summary['image_name']}")
    print(f"Total Objects   : {summary['total_objects']}")

    print("\nObject Counts:")

    if summary["object_counts"]:
        for name, count in summary["object_counts"].items():
            print(f"  {name}: {count}")
    else:
        print("  No objects detected.")

    print("\nOutput Image:")
    print(output_image_path)

    print("\nJSON File:")
    print(json_path)

    paths = {
        "Image": output_image_path,
        "Json": json_path
    }

    return paths


if __name__ == "__main__":
    image_path = input("Enter the path: ").strip()
    main(image_path)

