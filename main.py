import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
#eyes_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
mask_model = load_model('wearmask_model_ver1.h5')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('THERE WAS AN ERROR WHILE OPENING THE VIDEO CAPTURE DEVICE!')
    exit(0)

while True:
    ret, frame = cap.read()  # video captured frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting the image seen to greyscale format

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                                minNeighbors=5,
                                                minSize=(60, 60),
                                                flags=cv2.CASCADE_SCALE_IMAGE)  # recognizing the face

    #defining lists
    list_faces = []     #all the detected faces
    list_predict = []   #all the predictions

    for (x, y, w, h) in faces:

        #preprocessing steps are same that are followed when training the model
        face_fr = frame[y:y + h, x:x + w]
        face_fr = cv2.cvtColor(face_fr, cv2.COLOR_BGR2RGB)
        face_fr = cv2.resize(face_fr, (224, 224))
        face_fr = img_to_array(face_fr)
        face_fr = np.expand_dims(face_fr, axis=0)
        face_fr = preprocess_input(face_fr)
        list_faces.append(face_fr)
        if len(list_faces) > 0:
            list_predict = mask_model.predict(list_faces)
        for predicted in list_predict:
            (mask, nomask) = predicted
        #drawing rectangle around the face and a tag
        boxtext = "Mask On" if mask > nomask else "Mask Off"
        color = (0, 255, 0) if boxtext == "Mask" else (255, 255, 0)
        boxtext = "{}: {:.2f}%".format(boxtext, max(mask, nomask) * 100)
        cv2.putText(frame, boxtext, (x, y - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        #eyes = eyes_cascade.detectMultiScale(roi_gray)   # recognizing the area of the eyes
        #for (ex,ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,255,0), 2)


    cv2.imshow('Face Capture', frame)  # the result of the capture displayed
    if cv2.waitKey(20) & 0xff == ord('x'):  #with waitkey the capture will become visible, with ord 'x' the X key is our exit button
        break  # to break the while loop

cap.release()
cv2.destroyAllWindows()
