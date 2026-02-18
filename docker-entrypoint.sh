#!/bin/bash
# Script d'entrée Docker qui configure automatiquement l'environnement

set -e

echo "=== Configuration automatique de l'environnement ==="

# Vérifier et configurer DISPLAY si nécessaire
if [ -z "$DISPLAY" ]; then
    # Essayer de détecter automatiquement l'IP Windows depuis WSL
    if [ -f /etc/resolv.conf ]; then
        WINDOWS_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
        if [ ! -z "$WINDOWS_IP" ]; then
            export DISPLAY="${WINDOWS_IP}:0.0"
            echo "DISPLAY configuré automatiquement: $DISPLAY"
        fi
    fi
fi

# Vérifier si /dev/video0 existe
CAMERA_INDEX="auto"
if [ -e /dev/video0 ]; then
    echo "✅ Caméra détectée: /dev/video0"
    CAMERA_INDEX=0
elif [ -e /dev/video1 ]; then
    echo "✅ Caméra détectée: /dev/video1"
    CAMERA_INDEX=1
elif [ -e /dev/video2 ]; then
    echo "✅ Caméra détectée: /dev/video2"
    CAMERA_INDEX=2
else
    echo "❌ ERREUR: Aucune caméra détectée dans /dev/video*"
    echo "   → Pour activer la caméra, configurez usbipd-win :"
    echo "     1. Installez: winget install dorssel.usbipd-win"
    echo "     2. Partagez: usbipd bind --busid 1-1 (PowerShell admin)"
    echo "     3. Attachez: ./attach-camera.sh 1-1 (dans WSL)"
    echo ""
    echo "   → Utilisation de --auto-camera en dernier recours..."
    CAMERA_INDEX="auto"
fi

# Préparer les arguments de la commande
ARGS="$@"

# Si aucun argument n'est fourni, utiliser les valeurs par défaut
if [ -z "$ARGS" ]; then
    if [ "$CAMERA_INDEX" = "auto" ]; then
        ARGS="run_hand_cube.py --flip --auto-camera"
    else
        ARGS="run_hand_cube.py --flip --camera-index $CAMERA_INDEX"
    fi
else
    # Si l'utilisateur a fourni des arguments mais pas --camera-index, l'ajouter automatiquement
    if [[ "$ARGS" != *"--camera-index"* ]] && [[ "$ARGS" != *"--auto-camera"* ]]; then
        if [ "$CAMERA_INDEX" = "auto" ]; then
            ARGS="$ARGS --auto-camera"
        else
            ARGS="$ARGS --camera-index $CAMERA_INDEX"
        fi
    fi
    # Si --camera-index est présent mais sans valeur, utiliser la détection
    if [[ "$ARGS" == *"--camera-index"* ]] && [[ "$ARGS" != *"--camera-index"*" "*[0-9]* ]]; then
        if [ "$CAMERA_INDEX" = "auto" ]; then
            ARGS=$(echo "$ARGS" | sed 's/--camera-index[^ ]*/--auto-camera/')
        else
            ARGS=$(echo "$ARGS" | sed "s/--camera-index[^ ]*/--camera-index $CAMERA_INDEX/")
        fi
    fi
fi

echo ""
echo "🚀 Exécution: python $ARGS"
echo ""

# Exécuter la commande
exec python $ARGS
