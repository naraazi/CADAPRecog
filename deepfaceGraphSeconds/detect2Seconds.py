import cv2
import time
import pandas as pd
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
    'C:/Users/loren/OneDrive/Documentos/DesenvolvimentoIA/CADAPRecog/deep_haar/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
attributes = ("emotion",)

frames = []
emotions_data = []
start_time = time.time()

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

    elapsed_time = time.time() - start_time

    frames.append(frame)

    emotions = result[0]['emotion']
    emotions_data.append({'time': elapsed_time, **emotions})

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(emotions_data)
df.to_csv('C:/Users/loren/OneDrive/Documentos/DesenvolvimentoIA/CADAPRecog/deepfaceGraphSeconds/emotions_data_seconds'
          '.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo de gravação: {elapsed_time:.2f} segundos")
