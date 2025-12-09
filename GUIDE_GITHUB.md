# Guide pour publier le projet sur GitHub

## Étapes pour mettre le projet sur GitHub

### Étape 1 : Initialiser Git (si pas déjà fait)

```bash
git init
```

### Étape 2 : Ajouter tous les fichiers

```bash
git add .
```

**Note** : Le fichier `.gitignore` exclut automatiquement :
- Les fichiers Python compilés (`__pycache__/`)
- Les environnements virtuels (`venv/`, `env/`)
- Les fichiers IDE (`.vscode/`, `.idea/`)
- Les modèles DNN volumineux

**Optionnel** : Si vous ne voulez PAS versionner les données d'entraînement (images), décommentez ces lignes dans `.gitignore` :
```
# models/*.joblib
# data/**/*.png
```

### Étape 3 : Faire le premier commit

```bash
git commit -m "Initial commit: Système de reconnaissance faciale par Eigenfaces (PCA)"
```

### Étape 4 : Créer le dépôt sur GitHub

1. **Aller sur GitHub** : https://github.com
2. **Cliquer sur le bouton "+"** en haut à droite → "New repository"
3. **Remplir les informations** :
   - **Repository name** : `reconnaissance-faciale-eigenfaces` (ou le nom que vous préférez)
   - **Description** : "Système de contrôle d'accès par reconnaissance faciale utilisant l'algorithme Eigenfaces (PCA)"
   - **Visibilité** : Public ou Private (selon votre choix)
   - **NE PAS** cocher "Initialize with README" (on a déjà un README)
   - **NE PAS** ajouter .gitignore ou license (on a déjà .gitignore)
4. **Cliquer sur "Create repository"**

### Étape 5 : Connecter le dépôt local à GitHub

GitHub vous donnera des commandes, mais voici les commandes à exécuter :

```bash
# Remplacer USERNAME par votre nom d'utilisateur GitHub
# Remplacer REPO_NAME par le nom du dépôt que vous avez créé

git remote add origin https://github.com/USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

**Exemple** :
```bash
git remote add origin https://github.com/votre-username/reconnaissance-faciale-eigenfaces.git
git branch -M main
git push -u origin main
```

### Étape 6 : Vérifier

Allez sur votre dépôt GitHub, vous devriez voir tous vos fichiers !

---

## Commandes Git utiles pour la suite

### Ajouter des modifications

```bash
git add .
git commit -m "Description des modifications"
git push
```

### Voir l'état du dépôt

```bash
git status
```

### Voir l'historique des commits

```bash
git log
```

### Créer une nouvelle branche (pour des fonctionnalités)

```bash
git checkout -b nouvelle-fonctionnalite
# Faire des modifications...
git add .
git commit -m "Ajout nouvelle fonctionnalité"
git push -u origin nouvelle-fonctionnalite
```

---

## Recommandations pour GitHub

### 1. Améliorer le README.md

Assurez-vous que votre `README.md` contient :
- Description du projet
- Instructions d'installation
- Exemples d'utilisation
- Captures d'écran (optionnel mais recommandé)

### 2. Ajouter une licence

Si vous voulez partager votre code, créez un fichier `LICENSE` :
- MIT License (populaire et permissive)
- Apache 2.0
- GPL v3

### 3. Ajouter des topics/tags

Sur GitHub, ajoutez des topics pour faciliter la découverte :
- `face-recognition`
- `eigenfaces`
- `pca`
- `opencv`
- `python`
- `computer-vision`

### 4. Créer des releases

Quand vous avez une version stable :
1. Aller dans "Releases" → "Create a new release"
2. Tag : `v1.0.0`
3. Titre : "Version 1.0.0 - Système de reconnaissance faciale"
4. Description : Résumé des fonctionnalités

---

## Résumé des commandes (copier-coller rapide)

```bash
# 1. Initialiser Git
git init

# 2. Ajouter les fichiers
git add .

# 3. Premier commit
git commit -m "Initial commit: Système de reconnaissance faciale par Eigenfaces (PCA)"

# 4. Connecter à GitHub (remplacer USERNAME et REPO_NAME)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# 5. Renommer la branche en main
git branch -M main

# 6. Pousser vers GitHub
git push -u origin main
```

---

## Problèmes courants

### Erreur : "remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/USERNAME/REPO_NAME.git
```

### Erreur : "failed to push some refs"

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Oublier de mettre à jour .gitignore

Si vous avez déjà commité des fichiers que vous ne voulez pas :
```bash
git rm --cached fichier_a_supprimer
git commit -m "Remove fichier_a_supprimer"
git push
```

---

**Bon courage avec votre projet GitHub ! 🚀**

