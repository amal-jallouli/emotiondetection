import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle en utilisant le chemin relatif, car il se trouve dans le même dossier que le script
model = load_model('ProjetML.keras')

# Initialisez la caméra
cap = cv2.VideoCapture(0)  # 0 pour utiliser la webcam par défaut

# Vérifiez si la caméra est ouverte correctement
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

# Charger le classificateur en cascade pour la détection de visage
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Utilisation du chemin relatif

while True:
    # Capture une image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture d'image.")
        break

    # Convertir l'image en niveaux de gris pour la détection de visages
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Si une face est détectée
    for (x, y, w, h) in faces:
        # Découper la région de la face détectée et redimensionner
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))

        # Prétraiter l'image avant de la donner au modèle
        face_resized = np.array(face_resized)  # Convertir en tableau NumPy
        face_resized = face_resized.astype('float32') / 255.0  # Normaliser
        face_resized = np.expand_dims(face_resized, axis=0)  # Ajouter la dimension batch
        face_resized = np.expand_dims(face_resized, axis=-1)  # Ajouter la dimension de canal (1 pour image en niveaux de gris)

        # Prédire l'émotion
        emotion = model.predict(face_resized)
        emotion_label = np.argmax(emotion)  # Obtenir l'indice de l'émotion prédite

        # Dictionnaire des émotions
        emotion_dict = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Sadness', 6: 'Surprise'}

        # Afficher l'émotion prédite sur l'image
        cv2.putText(frame, emotion_dict[emotion_label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Dessiner un rectangle autour de la face détectée
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afficher le flux vidéo avec l'émotion détectée
    cv2.imshow('Emotion Detection - Press Q to Quit', frame)

    # Quitter la boucle si la touche 'Q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
