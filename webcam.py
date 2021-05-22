from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# loading model
model = load_model('gender_detection.model')

# Attach webcam
webcam = cv2.VideoCapture(0)
classes = ['Male', 'Female']
while webcam.isOpened():
    status, frame = webcam.read()

    # face detection
    face, confidence = cv.detect_face(frame)

    for idx, f in enumerate(face):
        # corner points
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop detected face
        face_crop = np.copy(frame[startY:endY, startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis = 0)

        # apply model
        conf = model.predict(face_crop)[0]

        # get label
        idx = np.argmax(conf)
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # add label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # output display
    cv2.imshow("gender detection", frame)

    # exit
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
webcam.release()
cv2.destroyAllWindows()