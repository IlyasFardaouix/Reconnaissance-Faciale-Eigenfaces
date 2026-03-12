from pathlib import Path
import argparse
import time
from collections import deque

import cv2
import joblib
import numpy as np

from face_utils import build_face_detector, detect_and_preprocess, draw_status


class DecisionBuffer:
    """Buffer circulaire pour vote majoritaire sur plusieurs frames."""
    
    def __init__(self, max_size: int = 20):
        """
        Args:
            max_size: Nombre de frames à garder en mémoire pour le vote
        """
        self.max_size = max_size
        self.distances = deque(maxlen=max_size)
        self.predictions = deque(maxlen=max_size)  # 0=refus, 1=autorisé, user_id pour multi-user
        self.user_ids = deque(maxlen=max_size)  # Pour multi-user
    
    def add(self, distance: float, threshold: float, user_id: int = 1) -> None:
        """Ajoute une mesure."""
        self.distances.append(distance)
        is_authorized = distance <= threshold
        self.predictions.append(1 if is_authorized else 0)
        self.user_ids.append(user_id)
    
    def get_majority_decision(self) -> tuple:
        """
        Retourne (decision, confidence, median_distance, detected_user_id)
        decision: 0=refus, 1=autorisé, user_id pour multi-user
        """
        if len(self.predictions) == 0:
            return None, 0.0, None, None
        
        median_dist = float(np.median(self.distances))
        
        # Vote majoritaire
        if len(self.user_ids) > 0 and len(set(self.user_ids)) > 1:
            # Mode multi-user: trouver l'utilisateur le plus fréquent
            from collections import Counter
            user_counts = Counter(self.user_ids)
            most_common_user, count = user_counts.most_common(1)[0]
            confidence = count / len(self.user_ids)
            
            # Vérifier si la majorité est autorisée pour cet utilisateur
            authorized_count = sum(1 for p, uid in zip(self.predictions, self.user_ids) if p == 1 and uid == most_common_user)
            if authorized_count > len(self.predictions) * 0.6:  # 60% de votes positifs
                return most_common_user, confidence, median_dist, most_common_user
            else:
                return 0, confidence, median_dist, None
        else:
            # Mode simple: vote binaire
            authorized_count = sum(self.predictions)
            confidence = authorized_count / len(self.predictions)
            decision = 1 if authorized_count > len(self.predictions) * 0.6 else 0
            return decision, confidence, median_dist, self.user_ids[0] if len(self.user_ids) > 0 else None
    
    def clear(self) -> None:
        """Réinitialise le buffer."""
        self.distances.clear()
        self.predictions.clear()
        self.user_ids.clear()
    
    def is_ready(self, min_frames: int = 8) -> bool:
        """Vérifie si on a assez de frames pour prendre une décision."""
        return len(self.predictions) >= min_frames


def find_nearest_user(embedding: np.ndarray, user_centroids: dict, thresholds: dict) -> tuple:
    """
    Trouve l'utilisateur le plus proche et retourne (user_id, distance, is_authorized).
    """
    if not user_centroids:
        return None, None, False
    
    min_dist = float('inf')
    nearest_user = None
    
    for user_id, centroid in user_centroids.items():
        dist = np.linalg.norm(embedding - centroid)
        if dist < min_dist:
            min_dist = dist
            nearest_user = user_id
    
    threshold = thresholds.get(nearest_user, thresholds.get(1, float('inf')))
    is_authorized = min_dist <= threshold
    
    return nearest_user, min_dist, is_authorized


def run_demo(
    model_path: Path,
    camera_index: int,
    min_confidence: float,
    use_dnn: bool = True,
    use_alignment: bool = True,
    buffer_size: int = 20,
    min_frames: int = 8,
) -> None:
    """
    Démo live avec vote majoritaire amélioré et support multi-utilisateurs.
    
    Args:
        model_path: Chemin vers le modèle entraîné
        camera_index: Index de la caméra
        min_confidence: Facteur de multiplication du seuil
        use_dnn: Utiliser le détecteur DNN si disponible
        use_alignment: Activer l'alignement de visage
        buffer_size: Taille du buffer pour vote majoritaire
        min_frames: Nombre minimum de frames avant décision
    """
    model = joblib.load(model_path)
    pca = model["pca"]
    
    # Support multi-utilisateurs
    multi_user = model.get("multi_user", False)
    if multi_user and "user_centroids" in model:
        user_centroids = model["user_centroids"]
        thresholds_dict = model.get("thresholds", {})
        # Ajuster les seuils avec min_confidence
        thresholds_dict = {uid: th * min_confidence for uid, th in thresholds_dict.items()}
        label_names = model.get("label_names", {})
    else:
        # Mode simple: un seul utilisateur
        authorized_centroid = model.get("authorized_centroid")
        if authorized_centroid is None:
            user_centroids_dict = model.get("user_centroids", {})
            authorized_centroid = user_centroids_dict.get(1) if user_centroids_dict else None
        if authorized_centroid is None:
            raise ValueError("Modèle invalide: pas de centroïde autorisé trouvé")
        user_centroids = {1: authorized_centroid}
        threshold = float(model.get("threshold", 0)) * min_confidence
        thresholds_dict = {1: threshold}
        label_names = model.get("label_names", {1: "authorized", 0: "others"})

    detector = build_face_detector(use_dnn=use_dnn)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    banner_text = "VEUILLEZ VOUS APPROCHER DE LA CAMERA"
    banner_color = (80, 80, 80)
    last_distance = None
    detected_user_name = None

    decision_buffer = DecisionBuffer(max_size=buffer_size)
    consecutive_no_face = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face = detect_and_preprocess(
                frame, detector, image_size=model["image_size"], use_alignment=use_alignment
            )
            
            if face is not None:
                consecutive_no_face = 0
                embedding = pca.transform(face.reshape(1, -1))
                
                # Trouver l'utilisateur le plus proche
                nearest_user, dist, is_auth = find_nearest_user(embedding, user_centroids, thresholds_dict)
                
                if nearest_user is not None:
                    decision_buffer.add(dist, thresholds_dict[nearest_user], user_id=nearest_user)
                    last_distance = dist
                    
                    # Phase d'analyse avant décision finale
                    if not decision_buffer.is_ready(min_frames=min_frames):
                        banner_text = "ANALYSE EN COURS..."
                        banner_color = (90, 90, 90)
                    else:
                        # Vote majoritaire
                        decision, confidence, median_dist, detected_user_id = decision_buffer.get_majority_decision()
                        last_distance = median_dist
                        
                        if decision == 0:
                            banner_text = "ACCES REFUSE"
                            banner_color = (0, 0, 200)
                            detected_user_name = None
                        else:
                            if multi_user and detected_user_id:
                                user_name = label_names.get(detected_user_id, f"User {detected_user_id}")
                                banner_text = f"ACCES AUTORISE - {user_name.upper()}"
                            else:
                                banner_text = "ACCES AUTORISE"
                            banner_color = (0, 180, 0)
                            detected_user_name = label_names.get(detected_user_id, "Authorized") if detected_user_id else None
                else:
                    decision_buffer.clear()
                    banner_text = "VEUILLEZ VOUS APPROCHER DE LA CAMERA"
                    banner_color = (80, 80, 80)
            else:
                consecutive_no_face += 1
                if consecutive_no_face > 5:  # Après 5 frames sans visage, réinitialiser
                    decision_buffer.clear()
                    banner_text = "VEUILLEZ VOUS APPROCHER DE LA CAMERA"
                    banner_color = (80, 80, 80)
                    last_distance = None
                    detected_user_name = None

            draw_status(frame, banner_text, banner_color)
            
            # Affichage des informations de debug
            info_lines = []
            if last_distance is not None:
                threshold_display = thresholds_dict.get(nearest_user if nearest_user else 1, 0)
                info_lines.append(f"distance: {last_distance:.3f} / seuil: {threshold_display:.3f}")
            
            if decision_buffer.is_ready():
                _, conf, _, _ = decision_buffer.get_majority_decision()
                info_lines.append(f"confiance: {conf:.1%}")
            
            if detected_user_name:
                info_lines.append(f"utilisateur: {detected_user_name}")
            
            y_offset = frame.shape[0] - 15
            for i, line in enumerate(reversed(info_lines)):
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("Eigenfaces - Controle d'acces", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            time.sleep(0.01)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live demo using trained Eigenfaces model.")
    parser.add_argument("--model-path", type=Path, default=Path("models/eigenfaces.joblib"), help="Path to trained model.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for cv2.VideoCapture.")
    parser.add_argument(
        "--confidence-scale",
        type=float,
        default=1.0,
        help="Multiply decision threshold ( <1.0 = plus permissif, >1.0 = plus strict ).",
    )
    parser.add_argument("--use-dnn", action="store_true", default=True, help="Use DNN face detector if available")
    parser.add_argument("--use-alignment", action="store_true", default=True, help="Enable face alignment")
    parser.add_argument("--buffer-size", type=int, default=20, help="Size of decision buffer for majority voting")
    parser.add_argument("--min-frames", type=int, default=8, help="Minimum frames before making decision")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        args.model_path,
        args.camera_index,
        args.confidence_scale,
        use_dnn=args.use_dnn,
        use_alignment=args.use_alignment,
        buffer_size=args.buffer_size,
        min_frames=args.min_frames,
    )
