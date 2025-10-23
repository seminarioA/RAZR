# ===========================================================
# Dockerfile para RAZR - Versión CPU Only
# ===========================================================

FROM python:3.10-slim-bullseye

# Instalación de dependencias del sistema
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev ffmpeg alsa-utils \
    libasound2-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Comando por defecto (puede modificarse con docker run)
CMD ["python3", "razr.py", "--mostrar-fps-procesado", "--mostrar-grafico"]