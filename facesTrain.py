import os
import cv2 as cv
import numpy as np


haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
dir = './Faces/train'

people = os.listdir(dir)
# for i in people :
#     print(i)

features = []
labels = []

for i in people :
    path_people = os.path.join(dir , i)
    label = people.index(i)
    
    for j in os.listdir(path_people):
        path_img = os.path.join(path_people , j)
        img = cv.imread(path_img)
        gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
        face_res = haarcascade.detectMultiScale(gray , 1.1 , 4)

        for (x,y,w,h) in face_res:
            onlyFace = gray[y:y+h , x:x+w]
            features.append(onlyFace)
            labels.append(label)
    


features = np.array(features , dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features , labels)

face_recognizer.save('face_trained.yml')

