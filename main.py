import os
import numpy as np
from PIL import Image
import cv2
import time
import sys

path = 'user_data'
name = ''

if not os.path.exists("user_data"):
    os.mkdir('user_data')

def face_generator():
    global name
    cam = cv2.VideoCapture(0)
    cam.set(3,640)
    cam.set(4,480)
    dectector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_id=input("Enter ID of user: ")
    name=input("Enter name: ")
    sample=int(input("Enter how many samples you wish to take: "))

    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

    print("Taking sample images of user... Please look at the camera.")

    count=0
    while True:
        ret,img=cam.read()
        converted_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=dectector.detectMultiScale(converted_image,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            count+=1

            cv2.imwrite("user_data/face."+str(face_id)+"."+str(count)+".jpg",converted_image[y:y+h,x:x+w])
            cv2.imshow("image",img)

        k=cv2.waitKey(1) & 0xff
        if k==27 or count>=sample:
            break

    print("Image samples taken successfully!")
    cam.release()
    cv2.destroyAllWindows()

    # After face images are taken, start the face authentication process
    permission = input("Do you wish to train your image data for face authentication [y|n] : ")
    permission_task(permission, 1)

def permission_task(val, task):
    if val.lower() == 'y':
        if task == 1:
            training_data()
        elif task == 2:
            detection()
    else:
        print("Thank you for using this application!")
        sys.exit()

def training_data():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    dectector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def images_and_labels(path):
        images_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in images_paths:
            gray_image = Image.open(image_path).convert('L')
            img_arr = np.array(gray_image, 'uint8')

            id = int(os.path.split(image_path)[-1].split(".")[1])

            faces = dectector.detectMultiScale(img_arr)

            for (x, y, w, h) in faces:
                face_samples.append(img_arr[y:y + h, x:x + w])
                ids.append(id)
        return face_samples, ids

    print("Training data... Please wait...")
    faces, ids = images_and_labels(path)

    recognizer.train(faces, np.array(ids))
    recognizer.write('trained_data.yml')

    print("Data trained successfully!")

    # After training data, proceed to detect if you're a spoofer or not
    detection()

def detection():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_data.yml')
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 5
    names = ['', name]
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    min_w = 0.1 * cam.get(3)
    min_h = 0.1 * cam.get(4)
    no = 0
    start_time = time.time()
    authenticated = False

    while time.time() - start_time < 10:
        if cam is None or not cam.isOpened():
            print('Warning: unable to open video source: ')

        ret, img = cam.read()
        if ret == False:
            print("Unable to detect image")
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(min_w), int(min_h)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])
            if accuracy < 100:
                id = names[id]
                accuracy = " {0}%".format(round(100 - accuracy))
                no += 1
                if int(accuracy[:-1]) > 55:
                    authenticated = True
            else:
                id = "Unknown"
                accuracy = " {0}%".format(round(100 - accuracy))
                no += 1
            cv2.putText(img, "Press Esc to close this window", (5, 25), font, 1, (255, 0, 255), 2)
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 0, 255), 2)
            cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        
        cv2.imshow('Camera', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    if authenticated:
        print("Authenticated")
    else:
        print("Hey Spoofer")

# Start the process by taking face images
print("\t\t\t ##### Welcome to Face Authentication System #####")
face_generator()
