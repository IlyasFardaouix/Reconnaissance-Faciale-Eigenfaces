from pathlib import Path
import argparse
import json

import joblib
import numpy as np
from sklearn.decomposition import PCA

from face_utils import IMAGE_SIZE, build_face_detector, ensure_dir, flatten_images, load_labeled_faces, load_multi_user_faces


def compute_threshold(auth_dists: np.ndarray, other_dists: np.ndarray) -> float:
    """
    Choose a distance threshold:
    - If negatives exist: use mid-point between 95e percentile (authorized) and 5e percentile (others).
    - Fallbacks to a conservative mean + 2*std when no negatives.
    """
    if other_dists.size:
        q_auth = np.percentile(auth_dists, 95)
        q_other = np.percentile(other_dists, 5)
        # If overlap, still take the midpoint — caller can tighten with --confidence-scale.
        return float((q_auth + q_other) / 2.0)
    mean = auth_dists.mean()
    std = auth_dists.std() if auth_dists.size > 1 else 0.0
    return float(mean + 2.0 * std)


def train_model(
    data_dir: Path,
    n_components: int,
    model_path: Path,
    multi_user: bool = False,
    use_dnn: bool = True,
    use_alignment: bool = True,
) -> None:
    """
    Entraîne le modèle Eigenfaces avec support multi-utilisateurs.
    
    Args:
        data_dir: Dossier contenant les données
        n_components: Nombre de composantes PCA
        model_path: Chemin de sauvegarde du modèle
        multi_user: Si True, utilise load_multi_user_faces (structure user_1/, user_2/, etc.)
        use_dnn: Utiliser le détecteur DNN si disponible
        use_alignment: Activer l'alignement de visage
    """
    detector = build_face_detector(use_dnn=use_dnn)
    
    if multi_user:
        images, labels, label_to_name = load_multi_user_faces(
            data_dir, detector, image_size=IMAGE_SIZE, use_alignment=use_alignment
        )
    else:
        images, labels = load_labeled_faces(
            data_dir, detector, image_size=IMAGE_SIZE, use_alignment=use_alignment
        )
        label_to_name = {1: "authorized", 0: "others"}

    if len(images) < 10:
        raise RuntimeError("Pas assez d'images. Visez >= 60 autorisées et >= 30 négatives.")

    X = flatten_images(images).astype(np.float32)
    y = np.array(labels, dtype=int)

    n_components = min(n_components, len(images) - 1, X.shape[1])
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized")
    embeddings = pca.fit_transform(X)

    # Calculer les centroïdes pour chaque utilisateur autorisé
    user_ids = sorted(set(y[y > 0]))  # Labels > 0 sont les utilisateurs autorisés
    user_centroids = {}
    user_distances = {}
    all_auth_dists = []
    all_other_dists = []

    for user_id in user_ids:
        user_embeddings = embeddings[y == user_id]
        if len(user_embeddings) == 0:
            continue
        
        centroid = user_embeddings.mean(axis=0)
        user_centroids[user_id] = centroid
        
        # Distances intra-utilisateur
        dists = np.linalg.norm(user_embeddings - centroid, axis=1)
        user_distances[user_id] = dists
        all_auth_dists.extend(dists.tolist())
        
        # Distances des autres utilisateurs à ce centroïde
        if len(user_ids) > 1:
            for other_user_id in user_ids:
                if other_user_id != user_id:
                    other_embeddings = embeddings[y == other_user_id]
                    if len(other_embeddings) > 0:
                        other_dists = np.linalg.norm(other_embeddings - centroid, axis=1)
                        all_other_dists.extend(other_dists.tolist())

    # Distances des "others" (label 0) vers le centroïde le plus proche
    if 0 in y:
        other_embeddings = embeddings[y == 0]
        if len(other_embeddings) > 0 and len(user_centroids) > 0:
            # Pour chaque "other", calculer la distance au centroïde le plus proche
            for other_emb in other_embeddings:
                min_dist = min(
                    np.linalg.norm(other_emb - centroid) for centroid in user_centroids.values()
                )
                all_other_dists.append(min_dist)

    all_auth_dists = np.array(all_auth_dists)
    all_other_dists = np.array(all_other_dists)

    # Calculer un seuil global ou par utilisateur
    if len(user_centroids) == 1:
        # Mode simple: un seul utilisateur
        threshold = compute_threshold(all_auth_dists, all_other_dists)
        thresholds = {list(user_centroids.keys())[0]: threshold}
    else:
        # Mode multi-utilisateurs: seuil par utilisateur
        thresholds = {}
        for user_id in user_ids:
            user_dists = user_distances[user_id]
            # Distances des autres vers ce centroïde
            other_to_this = []
            for other_emb in embeddings[y != user_id]:
                other_to_this.append(np.linalg.norm(other_emb - user_centroids[user_id]))
            other_to_this = np.array(other_to_this)
            thresholds[user_id] = compute_threshold(user_dists, other_to_this)
        
        # Seuil global pour la compatibilité
        threshold = compute_threshold(all_auth_dists, all_other_dists)

    model = {
        "pca": pca,
        "user_centroids": user_centroids,  # Dict {user_id: centroid}
        "authorized_centroid": user_centroids.get(1, list(user_centroids.values())[0] if user_centroids else None),  # Compatibilité
        "thresholds": thresholds,  # Dict {user_id: threshold}
        "threshold": threshold,  # Seuil global (compatibilité)
        "image_size": IMAGE_SIZE,
        "label_names": label_to_name,
        "user_distances": {k: v.tolist() for k, v in user_distances.items()},
        "multi_user": multi_user,
        "use_alignment": use_alignment,
    }

    ensure_dir(model_path.parent)
    joblib.dump(model, model_path)

    # Résumé
    summary = {
        "samples": int(len(images)),
        "n_users": len(user_centroids),
        "users": {str(uid): label_to_name.get(uid, f"user_{uid}") for uid in user_ids},
        "other_samples": int(np.sum(y == 0)),
        "threshold": float(threshold),
        "thresholds_per_user": {str(uid): float(th) for uid, th in thresholds.items()},
    }
    
    if len(all_auth_dists) > 0:
        summary["auth_dist_p50"] = float(np.percentile(all_auth_dists, 50))
        summary["auth_dist_p95"] = float(np.percentile(all_auth_dists, 95))
    
    if len(all_other_dists) > 0:
        summary["other_dist_p5"] = float(np.percentile(all_other_dists, 5))
        summary["other_dist_p50"] = float(np.percentile(all_other_dists, 50))
    
    summary["n_components"] = int(pca.n_components_)
    
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Eigenfaces PCA model for face verification.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root folder containing authorized/ and others/ or user_*/")
    parser.add_argument("--n-components", type=int, default=80, help="Max number of PCA components.")
    parser.add_argument("--model-path", type=Path, default=Path("models/eigenfaces.joblib"), help="Path to save the trained model.")
    parser.add_argument("--multi-user", action="store_true", help="Enable multi-user mode (expects user_1/, user_2/, etc.)")
    parser.add_argument("--use-dnn", action="store_true", default=True, help="Use DNN face detector if available")
    parser.add_argument("--use-alignment", action="store_true", default=True, help="Enable face alignment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        args.data_dir,
        args.n_components,
        args.model_path,
        multi_user=args.multi_user,
        use_dnn=args.use_dnn,
        use_alignment=args.use_alignment,
    )
