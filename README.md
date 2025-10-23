RAZR - Detector de Somnolencia

Sistema de visión computacional para la detección autónoma de
somnolencia en tiempo real mediante el Eye Aspect Ratio (EAR) y
MediaPipe Face Mesh.
Este documento describe los pasos para configurar el entorno, instalar
dependencias y ejecutar el script razr.py.

<h2>1. Requisitos previos</h2>

Antes de comenzar, asegúrate de tener instaladas las siguientes
herramientas:

-   Python 3.10+
-   pip (gestor de paquetes de Python)
-   virtualenv (opcional, pero recomendado)
-   Cámara web (para la captura en tiempo real)
-   Archivo de audio alarma.wav (para el aviso acústico)

<h2>2. Crear entorno virtual</h2>

Se recomienda aislar el proyecto en un entorno virtual para evitar
conflictos de dependencias.

En Windows (PowerShell o CMD)

    python -m venv venv

En Linux / macOS

    python3 -m venv venv

<h2>3. Activar el entorno virtual</h2>

En Windows

    venv\Scripts\activate

En Linux / macOS

    source venv/bin/activate

Nota: Verás que el prompt cambia para indicar que estás dentro del
entorno virtual, por ejemplo:

    (venv) user@machine:~/razr$

<h2>4. Instalar dependencias con pip</h2>

Una vez activado el entorno virtual, instala los paquetes necesarios:

    pip install mediapipe==0.10.21             opencv-python==4.9.0.80             imutils==0.5.4             simpleaudio==1.0.4             matplotlib==3.10.1             scipy==1.15.2             numpy==1.26.4

O si prefieres, crea un archivo requirements.txt con el siguiente
contenido:

    mediapipe==0.10.21
    opencv-python==4.9.0.80
    imutils==0.5.4
    simpleaudio==1.0.4
    matplotlib==3.10.1
    scipy==1.15.2
    numpy==1.26.4

Y ejecuta:

    pip install -r requirements.txt

<h2>5. Ejecutar el script razr.py</h2>

Ejecuta el script principal del sistema desde la terminal:

    python razr.py --ruta-alarma alarma.wav --mostrar-grafico --audio-activo

Argumentos opcionales

  ------------------------------------------------------------------------
  Parámetro           Descripción            Valor por defecto
  ------------------- ---------------------- -----------------------------
  --ruta-alarma       Ruta del archivo .wav  alarma.wav
                      para la alerta sonora  

  --indice-camara     Índice del dispositivo 0
                      de cámara              

  --umbral-ear        Umbral de EAR para     0.16
                      activar la alarma      

  --tiempo-umbral     Tiempo mínimo (seg)    1.0
                      que debe mantenerse    
                      por debajo del EAR     

  --duracion-alarma   Duración de la alarma  5
                      en segundos            

  --mostrar-grafico   Muestra el gráfico del False
                      EAR en tiempo real     

  --audio-activo      Activa el sonido de    False
                      alarma                 

  --salida-dir        Carpeta donde se       results
                      guardan las métricas e 
                      imágenes               
  ------------------------------------------------------------------------

<h2>6. Controles en ejecución</h2>

  Tecla   Acción
  ------- --------------------------------------------------------
  f       Guarda el rostro con los landmarks oculares (Figura 1)
  g       Guarda el histograma EAR (Figura 2, .png y .jpg)
  q       Finaliza la ejecución y cierra ventanas

<h2>7. Resultados generados</h2>

Durante la ejecución, se crean los siguientes archivos en la carpeta
results/:

  --------------------------------------------------------------------------
  Archivo                           Descripción
  --------------------------------- ----------------------------------------
  fig1_rostro_EAR.png               Captura del rostro con puntos oculares
                                    (Fig. 1 IEEE Q1)

  fig2_histograma_EAR.png / .jpg    Gráfico temporal del EAR (Fig. 2 IEEE
                                    Q1)

  metrics_run_YYYYMMDD_HHMMSS.csv   Registro de métricas por frame (EAR,
                                    latencia, estado de alarma)
  --------------------------------------------------------------------------

<h2>8. Desactivar el entorno virtual</h2>

Cuando termines de trabajar, puedes salir del entorno virtual con:

    deactivate

<h2>9. Verificación</h2>

Para comprobar que las dependencias se instalaron correctamente:

    python -m pip list

Y asegúrate de ver las versiones esperadas de cada paquete (por ejemplo,
mediapipe 0.10.21, opencv-python 4.9.0.80, etc.).
