# Reconnaissance de visage par Eigenfaces (PCA) - Version Améliorée

Prototype de contrôle d'accès binaire avec support multi-utilisateurs. Si le visage correspond à un utilisateur autorisé, le système affiche `ACCES AUTORISE`, sinon `ACCES REFUSE`.

## 🚀 Nouvelles fonctionnalités

- ✅ **Détecteur DNN OpenCV** : Détection de visage plus robuste avec fallback automatique vers Haar cascade
- ✅ **Alignement de visage** : Recalage automatique basé sur les landmarks pour réduire la variabilité angulaire
- ✅ **Vote majoritaire amélioré** : Lissage temporel sur fenêtre glissante (20 frames par défaut) pour réduire le bruit
- ✅ **Support multi-utilisateurs** : Gestion de plusieurs utilisateurs autorisés avec centroïdes séparés
- ✅ **Prétraitement amélioré** : CLAHE (Contrast Limited Adaptive Histogram Equalization) pour robustesse à l'éclairage

## Prérequis

- Python 3.9+
- Webcam fonctionnelle
- `pip install -r requirements.txt`

## Structure

### Mode simple (un utilisateur)
- `data/authorized/` : visages de l'utilisateur autorisé
- `data/others/` : visages des autres personnes (négatifs)

### Mode multi-utilisateurs
- `data/user_1/` : visages du premier utilisateur autorisé
- `data/user_2/` : visages du deuxième utilisateur autorisé
- `data/user_3/` : etc.
- `data/others/` : visages des autres personnes (négatifs)

- `models/eigenfaces.joblib` : modèle entraîné

## Utilisation

### 1) Collecte des visages

#### Mode simple
```bash
python collect_faces.py --label authorized --count 80
python collect_faces.py --label others --count 40
```

#### Mode multi-utilisateurs
```bash
python collect_faces.py --label user_1 --count 80
python collect_faces.py --label user_2 --count 80
python collect_faces.py --label others --count 40
```

**Options disponibles :**
- `--use-dnn` : Utiliser le détecteur DNN (par défaut activé, fallback Haar si indisponible)
- `--use-alignment` : Activer l'alignement de visage (par défaut activé)
- `--delay` : Délai entre captures (défaut: 0.25s)

### 2) Entraînement Eigenfaces (PCA)

#### Mode simple
```bash
python train.py --data-dir data --n-components 80
```

#### Mode multi-utilisateurs
```bash
python train.py --data-dir data --n-components 80 --multi-user
```

**Options disponibles :**
- `--n-components` : Nombre de composantes PCA (défaut: 80)
- `--multi-user` : Activer le mode multi-utilisateurs
- `--use-dnn` : Utiliser le détecteur DNN pour le prétraitement
- `--use-alignment` : Activer l'alignement de visage

Le script affiche un résumé avec :
- Seuil global et seuils par utilisateur (mode multi-user)
- Statistiques des distances (percentiles)
- Nombre d'échantillons par classe

### 3) Démonstration en direct

```bash
python live_demo.py --model-path models/eigenfaces.joblib --camera-index 0
```

**Options disponibles :**
- `--confidence-scale` : Facteur de multiplication du seuil (<1 plus permissif, >1 plus strict, défaut: 1.0)
- `--buffer-size` : Taille du buffer pour vote majoritaire (défaut: 20 frames)
- `--min-frames` : Nombre minimum de frames avant décision (défaut: 8)
- `--use-dnn` : Utiliser le détecteur DNN
- `--use-alignment` : Activer l'alignement de visage

**États affichés :**
- `VEUILLEZ VOUS APPROCHER DE LA CAMERA` : Aucun visage détecté
- `ANALYSE EN COURS...` : Visage détecté, analyse en cours (pendant les premières frames)
- `ACCES AUTORISE` ou `ACCES AUTORISE - USER_1` : Visage reconnu (mode multi-user)
- `ACCES REFUSE` : Visage non reconnu

**Touches :**
- `q` ou `Esc` : Quitter

## Notes techniques

### Détection de visage
- **Détecteur DNN OpenCV** : Plus robuste aux angles et éclairages variés
- **Fallback Haar cascade** : Si les modèles DNN ne sont pas disponibles
- Détection automatique du plus grand visage dans le cadre

### Prétraitement
- Conversion en niveaux de gris
- **Alignement automatique** : Rotation basée sur la position des yeux pour normaliser l'orientation
- Redimensionnement à 200x200 pixels
- **CLAHE** : Égalisation adaptative d'histogramme pour robustesse à l'éclairage

### Reconnaissance
- **PCA (Principal Component Analysis)** avec `whiten=True` pour décorréler les composantes
- Projection dans un espace de faible dimension (80 composantes par défaut)
- Distance euclidienne au centroïde de l'utilisateur autorisé
- **Seuil adaptatif** : Calculé entre le 95e percentile des distances autorisées et le 5e percentile des distances négatives

### Décision temporelle
- **Vote majoritaire** sur une fenêtre glissante de 20 frames
- Décision après au moins 8 frames consécutives avec visage détecté
- Réduction du bruit et des faux positifs/négatifs

### Mode multi-utilisateurs
- Chaque utilisateur a son propre centroïde dans l'espace PCA
- Seuil calculé individuellement pour chaque utilisateur
- Identification de l'utilisateur le plus proche avec affichage du nom

## Améliorations de performance

1. **Détecteur DNN** : Meilleure précision de détection, moins de faux négatifs
2. **Alignement** : Réduction de la variabilité due aux angles de vue
3. **Vote majoritaire** : Stabilité temporelle, réduction du bruit
4. **CLAHE** : Robustesse aux variations d'éclairage
5. **Multi-utilisateurs** : Support natif pour plusieurs personnes autorisées

## Dépannage

### Le système ne reconnaît pas mon visage
- Vérifiez que vous avez capturé suffisamment d'images (≥80 recommandé)
- Variez les angles, distances et éclairages lors de la capture
- Ajustez `--confidence-scale` à 0.8 ou 0.9 pour être plus permissif
- Vérifiez l'éclairage lors de la démo (évitez les reflets, ombres fortes)

### Trop de faux positifs
- Augmentez `--confidence-scale` à 1.2 ou 1.3
- Ajoutez plus d'images négatives dans `data/others/`
- Augmentez `--buffer-size` pour un vote plus conservateur

### Détection instable
- Augmentez `--min-frames` pour exiger plus de frames avant décision
- Vérifiez que la webcam fonctionne correctement
- Assurez-vous d'avoir un bon éclairage frontal

## Structure des fichiers

```
.
├── face_utils.py          # Utilitaires : détection, prétraitement, alignement
├── collect_faces.py       # Script de capture depuis webcam
├── train.py               # Entraînement du modèle Eigenfaces
├── live_demo.py           # Démonstration en temps réel
├── requirements.txt       # Dépendances Python
├── README.md              # Ce fichier
├── data/
│   ├── authorized/       # Visages autorisés (mode simple)
│   ├── user_1/           # Utilisateur 1 (mode multi-user)
│   ├── user_2/           # Utilisateur 2 (mode multi-user)
│   └── others/           # Visages négatifs
└── models/
    └── eigenfaces.joblib # Modèle entraîné
```

## Licence

Ce projet est un prototype éducatif pour la reconnaissance faciale par Eigenfaces (PCA).
