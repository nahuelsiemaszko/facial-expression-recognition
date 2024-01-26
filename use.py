import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

video = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (64, 128, 0),
    'Fear': (128, 0, 0),
    'Happy': (64, 250, 250),
    'Neutral': (128, 128, 128),
    'Sad': (255, 165, 0),
    'Surprise': (0, 165, 255)
}

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    for x, y, w, h in faces:
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        accuracy = result[0][label] * 100
        print(f"{labels_dict[label]}: {accuracy:.2f}%")

        color = emotion_colors[labels_dict[label]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(frame,
                    f"{labels_dict[label]}: {accuracy:.2f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
