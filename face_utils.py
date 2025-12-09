from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict, Any
import os

import cv2
import numpy as np


IMAGE_SIZE: Tuple[int, int] = (200, 200)

# URLs pour télécharger les modèles DNN OpenCV (optionnel, avec fallback Haar)
DNN_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
DNN_MODEL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def ensure_dir(path: Path) -> None:
    """Create directory tree if missing."""
    path.mkdir(parents=True, exist_ok=True)


class FaceDetector:
    """Wrapper pour détecteurs de visage avec fallback automatique."""
    
    def __init__(self, use_dnn: bool = True):
        self.use_dnn = use_dnn
        self.dnn_net = None
        self.haar_detector = None
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialise les détecteurs avec fallback."""
        # Essayer DNN d'abord si demandé
        if self.use_dnn:
            try:
                # Chercher les fichiers de modèle localement
                proto_path = Path("models/opencv_face_detector.pbtxt")
                model_path = Path("models/res10_300x300_ssd_iter_140000.caffemodel")
                
                if proto_path.exists() and model_path.exists():
                    # Essayer Caffe d'abord (format le plus commun)
                    try:
                        self.dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
                        print("[OK] Detecteur DNN (Caffe) charge")
                        return
                    except:
                        # Essayer TensorFlow
                        try:
                            self.dnn_net = cv2.dnn.readNetFromTensorflow(str(model_path), str(proto_path))
                            print("[OK] Detecteur DNN (TensorFlow) charge")
                            return
                        except:
                            pass
            except Exception as e:
                pass  # Fallback silencieux vers Haar
        
        # Fallback vers Haar cascade (toujours disponible)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar_detector = cv2.CascadeClassifier(cascade_path)
        if self.haar_detector.empty():
            raise RuntimeError(f"Unable to load Haar cascade from {cascade_path}")
        if not self.use_dnn:
            print("[OK] Detecteur Haar charge")
        else:
            print("[OK] Detecteur Haar charge (fallback depuis DNN)")
    
    def detect(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Détecte les visages et retourne liste de (x, y, w, h).
        Utilise DNN si disponible, sinon Haar.
        """
        if self.dnn_net is not None:
            return self._detect_dnn(gray)
        else:
            return self._detect_haar(gray)
    
    def _detect_dnn(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Détection via DNN OpenCV (format Caffe ou TensorFlow)."""
        h, w = gray.shape
        
        # Créer le blob pour le réseau (300x300 pour ResNet-SSD)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(gray, (300, 300)),
            1.0,
            (300, 300),
            [104.0, 117.0, 123.0]  # Valeurs de normalisation BGR
        )
        
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        # Format de sortie: [batch, class_id, confidence, x1, y1, x2, y2]
        # ou [batch, 1, num_detections, 7] où 7 = [class_id, confidence, x1, y1, x2, y2, ?]
        detection_shape = detections.shape
        
        if len(detection_shape) == 4 and detection_shape[2] == 1:
            # Format TensorFlow: [batch, 1, num_detections, 7]
            for i in range(detection_shape[3]):
                confidence = detections[0, 0, 0, i * 7 + 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, 0, i * 7 + 3] * w)
                    y1 = int(detections[0, 0, 0, i * 7 + 4] * h)
                    x2 = int(detections[0, 0, 0, i * 7 + 5] * w)
                    y2 = int(detections[0, 0, 0, i * 7 + 6] * h)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        else:
            # Format Caffe/SSD standard: [batch, 1, num_detections, 7]
            # ou [batch, num_detections, 7]
            for i in range(detections.shape[2] if len(detections.shape) == 4 else detections.shape[1]):
                if len(detections.shape) == 4:
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append((x1, y1, x2 - x1, y2 - y1))
                else:
                    confidence = detections[0, i, 2]
                    if confidence > 0.5:
                        x1 = int(detections[0, i, 3] * w)
                        y1 = int(detections[0, i, 4] * h)
                        x2 = int(detections[0, i, 5] * w)
                        y2 = int(detections[0, i, 6] * h)
                        faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def _detect_haar(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Détection via Haar cascade."""
        faces = self.haar_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]


def align_face(face_gray: np.ndarray, eyes: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> np.ndarray:
    """
    Aligne le visage horizontalement en utilisant les yeux.
    Si les yeux ne sont pas fournis, utilise une détection simple basée sur la géométrie.
    """
    h, w = face_gray.shape
    
    if eyes is not None:
        left_eye, right_eye = eyes
        # Calculer l'angle de rotation
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Centre du visage
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(face_gray, M, (w, h), flags=cv2.INTER_LINEAR)
        return aligned
    
    # Fallback: détection simple des yeux avec Haar cascade
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes_detected = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    
    if len(eyes_detected) >= 2:
        # Prendre les deux yeux les plus grands
        eyes_sorted = sorted(eyes_detected, key=lambda e: e[2] * e[3], reverse=True)[:2]
        eye_centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in eyes_sorted]
        
        # Déterminer gauche/droite
        if eye_centers[0][0] < eye_centers[1][0]:
            left_eye, right_eye = eye_centers[0], eye_centers[1]
        else:
            left_eye, right_eye = eye_centers[1], eye_centers[0]
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(face_gray, M, (w, h), flags=cv2.INTER_LINEAR)
        return aligned
    
    # Pas d'alignement possible, retourner l'original
    return face_gray


def build_face_detector(use_dnn: bool = True) -> FaceDetector:
    """Retourne un détecteur de visage (DNN avec fallback Haar)."""
    return FaceDetector(use_dnn=use_dnn)


def detect_and_preprocess(
    frame_bgr: np.ndarray,
    detector: Union[FaceDetector, cv2.CascadeClassifier],
    image_size: Tuple[int, int] = IMAGE_SIZE,
    use_alignment: bool = True,
) -> Optional[np.ndarray]:
    """
    Détecte le plus grand visage, aligne, redimensionne, applique CLAHE.
    
    Args:
        frame_bgr: Image BGR
        detector: FaceDetector ou CascadeClassifier
        image_size: Taille de sortie
        use_alignment: Activer l'alignement de visage
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Détection
    if isinstance(detector, FaceDetector):
        faces = detector.detect(gray)
    else:
        # Compatibilité avec l'ancien code
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        faces = [(x, y, w, h) for (x, y, w, h) in faces]
    
    if len(faces) == 0:
        return None
    
    # Garder la plus grande détection
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Extraire le visage avec un peu de marge
    margin = int(min(w, h) * 0.1)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(gray.shape[1] - x, w + 2 * margin)
    h = min(gray.shape[0] - y, h + 2 * margin)
    
    face = gray[y : y + h, x : x + w]
    
    # Alignement (optionnel mais recommandé)
    if use_alignment:
        face = align_face(face)
    
    # Redimensionnement
    face = cv2.resize(face, image_size)
    
    # CLAHE pour robustesse à l'éclairage
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)
    
    return face


def _iter_image_files(folder: Path) -> Iterable[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in exts:
            yield entry


def load_labeled_faces(
    data_dir: Path,
    detector: Union[FaceDetector, cv2.CascadeClassifier],
    image_size: Tuple[int, int] = IMAGE_SIZE,
    use_alignment: bool = True,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Charge les visages depuis data/authorized et data/others.
    Supporte maintenant plusieurs dossiers autorisés pour multi-utilisateurs.
    """
    authorized_dir = data_dir / "authorized"
    others_dir = data_dir / "others"

    images: List[np.ndarray] = []
    labels: List[int] = []

    # Charger les autorisés (label 1+ pour multi-utilisateurs)
    if authorized_dir.exists():
        for img_path in _iter_image_files(authorized_dir):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            face = detect_and_preprocess(img_bgr, detector, image_size=image_size, use_alignment=use_alignment)
            if face is None:
                continue
            images.append(face)
            labels.append(1)  # Label 1 pour autorisé
    
    # Charger les autres (label 0)
    if others_dir.exists():
        for img_path in _iter_image_files(others_dir):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            face = detect_and_preprocess(img_bgr, detector, image_size=image_size, use_alignment=use_alignment)
            if face is None:
                continue
            images.append(face)
            labels.append(0)

    return images, labels


def load_multi_user_faces(
    data_dir: Path,
    detector: Union[FaceDetector, cv2.CascadeClassifier],
    image_size: Tuple[int, int] = IMAGE_SIZE,
    use_alignment: bool = True,
) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """
    Charge les visages avec support multi-utilisateurs.
    Structure attendue: data/user_1/, data/user_2/, etc. et data/others/
    
    Returns:
        images, labels, label_to_name: labels commencent à 1 pour users, 0 pour others
    """
    images: List[np.ndarray] = []
    labels: List[int] = []
    label_to_name: Dict[int, str] = {0: "others"}
    
    # Chercher les dossiers user_*
    user_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("user_")])
    
    user_id = 1
    for user_dir in user_dirs:
        user_name = user_dir.name
        label_to_name[user_id] = user_name
        
        for img_path in _iter_image_files(user_dir):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            face = detect_and_preprocess(img_bgr, detector, image_size=image_size, use_alignment=use_alignment)
            if face is None:
                continue
            images.append(face)
            labels.append(user_id)
        
        user_id += 1
    
    # Fallback: si pas de user_*, utiliser authorized/ comme user_1
    authorized_dir = data_dir / "authorized"
    if len(user_dirs) == 0 and authorized_dir.exists():
        label_to_name[1] = "authorized"
        for img_path in _iter_image_files(authorized_dir):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            face = detect_and_preprocess(img_bgr, detector, image_size=image_size, use_alignment=use_alignment)
            if face is None:
                continue
            images.append(face)
            labels.append(1)
        user_id = 2
    
    # Charger les autres (label 0)
    others_dir = data_dir / "others"
    if others_dir.exists():
        for img_path in _iter_image_files(others_dir):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            face = detect_and_preprocess(img_bgr, detector, image_size=image_size, use_alignment=use_alignment)
            if face is None:
                continue
            images.append(face)
            labels.append(0)

    return images, labels, label_to_name


def flatten_images(images: Iterable[np.ndarray]) -> np.ndarray:
    """Flatten list of images into matrix of shape (n_samples, n_features)."""
    stacked = np.stack(images, axis=0)
    return stacked.reshape(len(stacked), -1)


def draw_status(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    """Overlay status banner on the frame."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, thickness=-1)
    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
