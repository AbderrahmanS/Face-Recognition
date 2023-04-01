import cv2 as cv

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

haarCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('./Faces/val/jerry_seinfeld/2.jpg')
# cv.imshow('jerry' , img)
gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

# Detect the face in the image
face_res = haarCascade.detectMultiScale(gray , 1.1 , 4)

for (x,y,w,h) in face_res :
    onlyFace = gray[y:y+h , x:x+w]
    # cv.rectangle(img , (x,y) , (x+w,y+h),(0,255,0))

    label , confidence = face_recognizer.predict(onlyFace)
    print('Label is : ' , people[label] , ' with a confidence of ', confidence)




# cv.waitKey(0)