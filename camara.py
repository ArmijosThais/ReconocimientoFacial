import cv2
import face_recognition
import numpy as np
import pickle

# Cargar el modelo entrenado desde el archivo
model_filename = 'trained_model.pkl'
with open(model_filename, 'rb') as file:
    clf = pickle.load(file)

# Iniciar la captura de video
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar cada fotograma del flujo de video
    ret, frame = video_capture.read()

    # Encontrar todas las caras en el fotograma
    face_locations = face_recognition.face_locations(frame)

    # Si se encuentran caras, reconocerlas
    if len(face_locations) > 0:
        # Convertir el fotograma de formato BGR (OpenCV) a formato RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Obtener las codificaciones faciales para cada cara en el fotograma
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Predecir los nombres y las probabilidades de pertenencia para cada cara utilizando el clasificador entrenado
        face_names = []
        face_probabilities = []
        for face_encoding in face_encodings:
            face_prediction = clf.predict([face_encoding])
            probabilities = clf.predict_proba([face_encoding])[0]
            predicted_probability = probabilities[np.where(clf.classes_ == face_prediction)]*100
            face_names.append(face_prediction[0])
            face_probabilities.append(predicted_probability[0])

        # Dibujar rectángulos y etiquetas o "Desconocido" en cada cara del fotograma
        for (top, right, bottom, left), name, probability in zip(face_locations, face_names, face_probabilities):
            if probability < 35:
                name = 'Desconocido'
            else:
                name = f"{name}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name,(left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'No Rostros', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar el fotograma resultante
    cv2.imshow('Reconocimiento de Rostros en Tiempo Real', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
video_capture.release()
cv2.destroyAllWindows()
