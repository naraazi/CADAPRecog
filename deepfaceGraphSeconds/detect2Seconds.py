import cv2
import time
import pandas as pd
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
    '/home/lorenzo/Python/TOYlab/deepFace/deep_haar/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
attributes = ("emotion",)

frames = []  # -- lista para armazenar os quadros
emotions_data = []  # -- lista para armazenar os dados das emoções
start_time = time.time()  # -- início da gravação

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, attributes, enforce_detection=False)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                result[0]['dominant_emotion'],
                (50, 50),
                font,
                3,
                (0, 0, 255),
                2,
                cv2.LINE_4)
    cv2.imshow('Webcam', frame)

    # -- calcule o tempo decorrido em segundos
    elapsed_time = time.time() - start_time

    # -- armazena o quadro atual
    frames.append(frame)

    # -- extrai os dados das emoções e adiciona à lista de emoções com o tempo decorrido em segundos
    emotions = result[0]['emotion']
    emotions_data.append({'time': elapsed_time, **emotions})

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -- cria um DataFrame pandas com os dados das emoções e o tempo decorrido em segundos e salva em um CSV
df = pd.DataFrame(emotions_data)
df.to_csv('/home/lorenzo/Python/TOYlab/deepFace/deepfaceGraphSeconds/emotions_data_seconds.csv', index=False)

# -- calc do tempo decorrido
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo de gravação: {elapsed_time:.2f} segundos")
