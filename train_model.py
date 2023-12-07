from collections import Counter
import random
import face_recognition
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
import os

# Los datos de entrenamiento y prueba serán todas las codificaciones faciales de todas las imágenes conocidas, y las etiquetas son sus nombres
codificaciones = []
nombres = []

# Directorio de imágenes de rostros recortados
image_dir = 'C:/ReconomiciemtoIAProyecto/dataset'

# Recorrer cada persona en el directorio de imágenes
for persona in os.listdir(image_dir):
    persona_dir = os.path.join(image_dir, persona)
    if os.path.isdir(persona_dir):
        # Recopilar todas las imágenes para la persona actual
        images = []
        for person_img in os.listdir(persona_dir):
            contador = 0
            img_path = os.path.join(persona_dir, person_img)
            images.append(img_path)

        # Mezclar aleatoriamente las imágenes para la persona actual
        random.shuffle(images)

        # Procesar las imágenes de entrenamiento
        for img_path in images:
            if contador < 150:
                face = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(face)

                # Si la imagen contiene exactamente una cara
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
                    # Agregar la codificación facial de la imagen actual con la etiqueta correspondiente (nombre) a los datos de entrenamiento
                    codificaciones.append(face_enc)
                    nombres.append(persona)
                    contador+=1
                else:
                    print(persona + "/" + os.path.basename(img_path) + " se omitió y no se puede usar para el modelo")

# Imprimir el número de repeticiones de cada nombre
"""repeticiones = Counter(nombres)
for nombre, cantidad in repeticiones.items():
    print("Nombre:", nombre, "Repeticiones:", cantidad)"""


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(codificaciones, nombres, test_size=0.2, random_state=42)

# Crear y entrenar el clasificador SVM
clf = svm.SVC(gamma='scale', probability=True)
clf.fit(X_train, y_train)

# Imprimir la exactitud del modelo de entrenamiento
exactitud_entrenamiento = clf.score(X_train, y_train)
print("Exactitud del Modelo de Entrenamiento:", exactitud_entrenamiento)

# Imprimir la exactitud del modelo de prueba
exactitud_prueba = clf.score(X_test, y_test)
print("Exactitud del Modelo de Prueba:", exactitud_prueba)

# Guardar el modelo entrenado en un archivo
model_filename = 'trained_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(clf, file)

print("Modelo entrenado y guardado como", model_filename)
