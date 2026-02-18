# Utilisation avec Docker - Guide Complet

**Ce projet doit être exécuté dans Docker.**

## Vue d'ensemble

- `Dockerfile` - Image Docker avec toutes les dépendances
- `docker-compose.yml` - Configuration pour Linux/Windows
- `docker-compose.mac.yml` - Configuration optimisée pour Mac
- `docker-entrypoint.sh` - Script d'initialisation automatique

## Sur Mac

### Prérequis

1. **Installer Docker Desktop** : https://www.docker.com/products/docker-desktop
2. **Installer XQuartz** (pour l'affichage graphique) :
   ```bash
   brew install --cask xquartz
   ```

### Configuration XQuartz

1. **Lancer XQuartz**
2. **XQuartz → Préférences → Sécurité**
   - ✅ Cochez "Allow connections from network clients"
3. **Redémarrer XQuartz**

### Utilisation

```bash
# 1. Construire l'image
docker-compose -f docker-compose.mac.yml build

# 2. Configurer DISPLAY
export DISPLAY=:0
xhost +localhost

# 3. Lancer
docker-compose -f docker-compose.mac.yml up
```

### Permissions caméra (macOS)

La première fois, autorisez l'accès à la caméra :
- **Système** → **Préférences Système** → **Confidentialité** → **Caméra**
- Cochez **Docker** ou **Terminal**

## Sur Linux

```bash
# 1. Construire
docker-compose build

# 2. Configurer DISPLAY
export DISPLAY=:0
xhost +local:docker

# 3. Lancer
docker-compose up
```

## Sur Windows/WSL2

```bash
# 1. Construire
docker-compose build

# 2. Lancer VcXsrv sur Windows avec "Disable access control"
# 3. Dans WSL, configurer DISPLAY
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

# 4. Lancer
docker-compose up
```

## Commandes utiles

```bash
# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down

# Rebuild complet
docker-compose build --no-cache

# Accéder au shell du conteneur
docker-compose exec cv3d /bin/bash

# Lancer une commande spécifique
docker-compose run --rm cv3d python run_hand_tracking.py --flip
```

## Fonctionnalités automatiques

Le script `docker-entrypoint.sh` configure automatiquement :
- ✅ Détection de la caméra (`/dev/video0`, `/dev/video1`, etc.)
- ✅ Configuration DISPLAY pour X11
- ✅ Paramètres optimaux selon l'environnement
