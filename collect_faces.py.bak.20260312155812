from pathlib import Path
import argparse
import time

import cv2

from face_utils import IMAGE_SIZE, build_face_detector, detect_and_preprocess, ensure_dir


def collect(
    label: str,
    output_dir: Path,
    target_count: int,
    camera_index: int,
    delay: float,
    use_dnn: bool = True,
    use_alignment: bool = True,
) -> None:
    """
    Capture des visages pour le dataset.
    
    Args:
        label: "authorized", "others", ou "user_X" pour multi-utilisateurs
        output_dir: Dossier racine du dataset
        target_count: Nombre d'images à capturer
        camera_index: Index de la caméra
        delay: Délai entre captures (secondes)
        use_dnn: Utiliser le détecteur DNN si disponible
        use_alignment: Activer l'alignement de visage
    """
    detector = build_face_detector(use_dnn=use_dnn)
    save_dir = output_dir / label
    ensure_dir(save_dir)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    captured = 0
    last_face_time = 0
    
    try:
        while captured < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            face = detect_and_preprocess(
                frame, detector, image_size=IMAGE_SIZE, use_alignment=use_alignment
            )
            
            if face is not None:
                current_time = time.time()
                # Éviter les captures trop rapprochées
                if current_time - last_face_time >= delay:
                    filename = save_dir / f"{int(time.time() * 1000)}.png"
                    cv2.imwrite(str(filename), face)
                    captured += 1
                    last_face_time = current_time
                    print(f"[{captured}/{target_count}] saved {filename}")

            # Afficher le compteur sur la frame
            cv2.putText(
                frame,
                f"Capture: {captured}/{target_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            
            cv2.imshow("Capture visage", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n[OK] Capture terminee: {captured} images sauvegardees dans {save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture faces for dataset.")
    parser.add_argument(
        "--label",
        default="authorized",
        help='Destination label folder (e.g., "authorized", "others", "user_1", "user_2").',
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Root dataset directory.")
    parser.add_argument("--count", type=int, default=40, help="Number of face samples to capture.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index.")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds to wait between saved frames.")
    parser.add_argument("--use-dnn", action="store_true", default=True, help="Use DNN face detector if available")
    parser.add_argument("--use-alignment", action="store_true", default=True, help="Enable face alignment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(
        args.label,
        args.output_dir,
        args.count,
        args.camera_index,
        args.delay,
        use_dnn=args.use_dnn,
        use_alignment=args.use_alignment,
    )
