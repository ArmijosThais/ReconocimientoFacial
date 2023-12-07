import cv2
import os
import numpy as np

# Directorio raíz que contiene las subcarpetas de las personas
root_dir = 'C:/ReconomiciemtoIAProyecto/FotosSinFondo'

# Directorio de destino para el dataset
dataset_dir = 'C:/ReconomiciemtoIAProyecto/dataset'

# Lista de transformaciones de aumento de datos a aplicar
transformations = [
    ('rotation', 10, 60),
    ('horizontal_flip', 0.5, 60),
    ('scale', 0.2, 60),
    ('crop', 0.1, 60),
    ('brightness', 0.1, 60),
    ('shift', 0.2, 60)
]

def apply_rotation(image, angle):
    rows, cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, matrix, (cols, rows))
    return rotated

def apply_horizontal_flip(image):
    flipped = cv2.flip(image, 1)
    return flipped

def apply_scale(image, factor):
    rows, cols = image.shape[:2]
    scaled = cv2.resize(image, (int(cols*(1+factor)), int(rows*(1+factor))))
    return scaled

def apply_crop(image, ratio):
    rows, cols = image.shape[:2]
    crop_ratio = min(ratio, 0.5)  # Limitar el recorte máximo a la mitad de la imagen
    crop_rows = int(rows * crop_ratio)
    crop_cols = int(cols * crop_ratio)
    cropped = image[crop_rows:rows-crop_rows, crop_cols:cols-crop_cols]
    return cropped

def apply_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * factor
    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened

def apply_shift(image, shift):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, shift], [0, 1, shift]])
    shifted = cv2.warpAffine(image, M, (cols, rows))
    return shifted

# Crear el directorio de destino del dataset
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print("Nueva carpeta: dataset")
    
# Iterar sobre las subcarpetas de las personas
for person_dir in os.listdir(root_dir):
    person_path = os.path.join(root_dir, person_dir)

    # Crear una subcarpeta en el directorio del dataset para la persona actual
    person_dataset_dir = os.path.join(dataset_dir, person_dir)
    if not os.path.exists(person_dataset_dir):
        os.makedirs(person_dataset_dir)
    
    # Obtener todas las imágenes originales en la subcarpeta actual
    original_images = os.listdir(person_path)
    
    # Verificar si la persona ya tiene al menos 300 imágenes en el dataset
    if len(original_images) >= 400:
        selected_images = original_images[:400]  # Seleccionar las primeras 300 imágenes
    else:
        selected_images = original_images  # Utilizar todas las imágenes existentes
        
    # Iterar sobre las imágenes seleccionadas
    for image_file in selected_images:
        image_path = os.path.join(person_path, image_file)
        image = cv2.imread(image_path)
        
        # Guardar la imagen original en el directorio del dataset
        output_path = os.path.join(person_dataset_dir, image_file)
        cv2.imwrite(output_path, image)
        
    # Generar imágenes adicionales mediante transformaciones de aumento de datos
    remaining_images = 300 - len(selected_images)
    
    # Verificar si es necesario generar imágenes adicionales
    if remaining_images > 0:
        for transform, param, min_images in transformations:
            if remaining_images >= min_images:
                for i in range(min_images):
                    image_file = np.random.choice(selected_images)  # Seleccionar una imagen aleatoria
                    image_path = os.path.join(person_path, image_file)
                    image = cv2.imread(image_path)
                    
                    transformed_image = None
                    
                    if transform == 'rotation':
                        angle = np.random.choice([-10, 10])  # Rotación hacia la izquierda o la derecha
                        transformed_image = apply_rotation(image, angle)
                    elif transform == 'horizontal_flip':
                        transformed_image = apply_horizontal_flip(image)
                    elif transform == 'scale':
                        transformed_image = apply_scale(image, 0.2)
                    elif transform == 'crop':
                        transformed_image = apply_crop(image, 0.1)
                    elif transform == 'brightness':
                        transformed_image = apply_brightness(image, 0.6)
                    elif transform == 'shift':
                        transformed_image = apply_shift(image, 10)
                    
                    if transformed_image is not None:
                        transformed_output_path = os.path.join(person_dataset_dir, f'{transform}_{i+1}_{image_file}')
                        
                        # Verificar si la imagen generada ya existe en el dataset
                        if not os.path.exists(transformed_output_path):
                            cv2.imwrite(transformed_output_path, transformed_image)
                            
                    remaining_images -= 1
                    if remaining_images == 0:
                        break
                    
            if remaining_images == 0:
                break