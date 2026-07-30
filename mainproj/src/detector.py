import json
import cv2

from datetime import datetime
from pathlib import Path

from collections import Counter
from ultralytics import YOLO


class FashionDetector:

    def __init__(self, model_path):
        """
        Load YOLO model only once.
        """
        self.model = YOLO(model_path)

    @staticmethod
    def save_image(image, images_dir=None):
        """Save an uploaded image in ``images`` and return its local path.

        ``image`` can be a path, an OpenCV/Numpy image, raw image bytes, or a
        file-like upload object.  Images already inside ``images`` are reused
        instead of copied again.
        """
        project_dir = Path(__file__).resolve().parent.parent
        images_dir = Path(images_dir) if images_dir else project_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if isinstance(image, (str, Path)):
            source = Path(image).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Image not found: {source}")
            if source.suffix.lower() not in image_extensions:
                raise ValueError("Unsupported image format.")
            try:
                source.relative_to(images_dir.resolve())
                return source
            except ValueError:
                suffix = source.suffix.lower()
                data = source.read_bytes()
                original_name = source.stem
        elif isinstance(image, bytes):
            data = image
            suffix = ".jpg"
            original_name = "upload"
        elif hasattr(image, "read"):
            data = image.read()
            name = Path(getattr(image, "name", "upload.jpg"))
            suffix = name.suffix.lower() if name.suffix.lower() in image_extensions else ".jpg"
            original_name = name.stem or "upload"
        else:
            # OpenCV/Numpy image arrays are encoded as a PNG to avoid lossy
            # conversion before the detector reads the saved upload.
            suffix = ".png"
            original_name = "upload"
            ok, encoded = cv2.imencode(suffix, image)
            if not ok:
                raise ValueError("Could not encode the uploaded image.")
            data = encoded.tobytes()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = images_dir / f"{original_name}_{timestamp}{suffix}"
        destination.write_bytes(data)
        return destination

    def detect_objects(self, image, save_dir=None, images_dir=None):
        """
        Detect fashion objects in an image.

        Returns:
            annotated_image (numpy.ndarray): Image with bounding boxes.
            summary (dict): Detection summary.
            output_image_path (str): Path to saved annotated image.
            json_path (str): Path to saved JSON file.
        """

        project_dir = Path(__file__).resolve().parent.parent
        save_dir = Path(save_dir) if save_dir else project_dir / "Artifacts" / "detection"
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.save_image(image, images_dir)

        image_name = image_path.name
        image_base = image_path.stem

        results = self.model(str(image_path))
        result = results[0]

        annotated_image = result.plot()

        output_image_path = save_dir / f"detected_{image_name}"

        cv2.imwrite(str(output_image_path), annotated_image)

        detected_classes = []
        detections = []

        for box in result.boxes:

            cls = int(box.cls)
            conf = float(box.conf)

            class_name = self.model.names[cls]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detected_classes.append(class_name)

            detections.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "bounding_box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

        object_counts = Counter(detected_classes)

        json_path = save_dir / f"{image_base}.json"

        summary = {
            "image_name": image_name,
            "source_image_path": str(image_path),
            "total_objects": len(detected_classes),
            "object_counts": dict(object_counts),
            "detections": detections,
            "annotated_image_path": str(output_image_path),
            "json_file_path": str(json_path)
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return annotated_image, summary, str(output_image_path), str(json_path)
