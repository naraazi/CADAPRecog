# -- wheel: \loren\appdata\local\pip\cache\wheels\97\4a\3f\f6f222440f03d888f026ac848b6b4ea57183ebfcd8f3cb8904
# -- weights: C:\Users\loren\.deepface\weights\facial_expression_model_weights.h5

import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
    'C:/Users/loren/OneDrive/Documentos/DesenvolvimentoIA/CADAPRecog/deep_haar/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
attributes = ("emotion",)

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, attributes, enforce_detection=False)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.putText(frame,
                    text=result[0]['dominant_emotion'],
                    org=(x, y), thickness=3,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0, 0, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
