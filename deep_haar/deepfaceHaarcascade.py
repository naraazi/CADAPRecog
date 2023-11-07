# -- DEPENDENCIA (pesos): .deepface/weights/*pesos
import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
    '/home/lorenzo/Python/TOYlab/deepFace/deep_haar/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
attributes = ("emotion",)

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, attributes, enforce_detection=False)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # -- faz um menorzinho mais bonito
        cv2.putText(frame,
                    text=result[0]['dominant_emotion'],
                    org=(x, y), thickness=3,  # -- thickness aqui
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0, 0, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # -- deixa a fonte gigantona e fora do retangulo
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(frame,
    #             result[0]['dominant_emotion'],
    #             (50, 50),
    #             font,
    #             3,
    #             (0, 0, 255),
    #             2,
    #             cv2.LINE_4)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
