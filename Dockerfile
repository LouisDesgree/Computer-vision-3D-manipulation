FROM python:3.11-slim

# Installer les dépendances système nécessaires pour OpenCV et MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglx0 \
    libglu1-mesa \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    v4l-utils \
    wget \
    fonts-dejavu \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer le répertoire pour les modèles MediaPipe
RUN mkdir -p models && \
    chmod -R 755 /app

# Copier et rendre exécutable le script d'entrée
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Variables d'environnement
ENV DISPLAY=:0
ENV PYTHONUNBUFFERED=1
ENV QT_X11_NO_MITSHM=1
ENV LIBGL_ALWAYS_INDIRECT=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MESA_GLSL_VERSION_OVERRIDE=330
ENV QT_QPA_FONTDIR=/usr/share/fonts
ENV FONTCONFIG_PATH=/etc/fonts

# Utiliser le script d'entrée qui configure tout automatiquement
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["run_hand_cube.py", "--flip"]
