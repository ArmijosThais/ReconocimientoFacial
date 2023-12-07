import os
import face_recognition
from PIL import Image

def obtener_rostro_fotos(carpeta_principal, carpeta_destino):
    # Recorre todas las subcarpetas en la carpeta principal
    for nombre_subcarpeta in os.listdir(carpeta_principal):
        ruta_subcarpeta = os.path.join(carpeta_principal, nombre_subcarpeta)

        # Comprueba si es una carpeta
        if os.path.isdir(ruta_subcarpeta):
            # Crea la carpeta de destino si no existe
            carpeta_estudiante = os.path.join(carpeta_destino, nombre_subcarpeta)
            os.makedirs(carpeta_estudiante, exist_ok=True)

            # Recorre todos los archivos en la subcarpeta
            for nombre_archivo in os.listdir(ruta_subcarpeta):
                if nombre_archivo.endswith(".png"):
                    ruta_png = os.path.join(ruta_subcarpeta, nombre_archivo)

                    # Carga la imagen utilizando face_recognition
                    imagen = face_recognition.load_image_file(ruta_png)

                    # Detecta los rostros en la imagen
                    face_locations = face_recognition.face_locations(imagen)

                    # Extrae y guarda los rostros encontrados
                    if len(face_locations) == 1:
                        for (top, right, bottom, left) in face_locations:
                            rostro = imagen[top:bottom, left:right]
                            imagen_rostro = Image.fromarray(rostro)
                            ruta_destino = os.path.join(carpeta_estudiante, nombre_archivo)
                            imagen_rostro.save(ruta_destino)

# Ruta de la carpeta principal que contiene las carpetas de los estudiantes
carpeta_principal = 'C:/ReconomiciemtoIA/FotosSinFondo'

# Ruta de la carpeta donde se guardarán las imágenes de los rostros
carpeta_destino = 'C:/ReconomiciemtoIA/FotosSoloRostros'

# Quita el fondo de las imágenes en la carpeta principal y guarda los rostros en la carpeta de destino
obtener_rostro_fotos(carpeta_principal, carpeta_destino)
