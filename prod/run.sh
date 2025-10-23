#!/bin/bash
# ===========================================================
# Lanzador experimental RAZR en contenedor Docker (CPU Only)
# ===========================================================

# Limitar hardware: 2 núcleos de CPU y 3 GB de RAM
CPU_LIMIT="0-3"
MEMORY_LIMIT="4g"

# Nombre de la imagen y del contenedor
IMAGE_NAME="razr_cpu"
CONTAINER_NAME="razr_experimento"

# Construir imagen
docker build -t $IMAGE_NAME .

# Ejecutar contenedor
docker run -it --rm \
  --cpuset-cpus="$CPU_LIMIT" \
  --memory="$MEMORY_LIMIT" \
  --device=/dev/video0:/dev/video0 \
  --device /dev/snd \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  --workdir /app \
  --name $CONTAINER_NAME \
  $IMAGE_NAME