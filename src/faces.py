import numpy as np
import cv2
import pickle
from datetime import date , datetime

import os
from PIL import Image

import time


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
i = 0
face_id =0
number = 0

def recognitionface():
    while(True):
        # Capture frame-by-frame
        global number , i , face_id
        ret, frame = cap.read()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        if number == 0 :
            for (x, y, w, h) in faces:
                #print(x,y,w,h)
                roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
                roi_color = frame[y:y+h, x:x+w]

        	   # recognize? deep learned model predict keras tensorflow pytorch scikit lear
                id_, conf = recognizer.predict(roi_gray)
                print('Confidence ', conf)
                if conf <= 45:
                #print(5: #id_)
                #print(labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    number += 1
                    print(name)
                    # TODO: open door here
                    # print("Found")
                    # time.sleep(5)
                    break

            #
                else :
                    color = (255, 255, 255)
                    stroke = 2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,"Unknow", (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    i += 1
                    print(i)
                    if i >= 30:
                        face_id += 1 
                        today = date.today()
                        now = datetime.now()
                        d2 = today.strftime("%B %d, %Y") 
                        cv2.imwrite("picturestranger/imagestranger/" +str(face_id)+'.'+str(d2)+".jpg",roi_color)

                        i = 0

            #img_item = "10.png"
            #cv2.imwrite(img_item, roi_color)

                color = (255, 0, 0) #BGR 0-255 
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            	#subitems = smile_cascade.detectMultiScale(roi_gray)
            	#for (ex,ey,ew,eh) in subitems:
            	#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        elif number >= 80 :
            number = 0           
        elif number >= 1 :
            number += 1

        print(number)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            # number += 1 
            # if number == 200:q
            #     number = 0
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def datagatherface():

    #Setup camera .
    cam = cv2.VideoCapture(0)
    cam.set(3,640) #set video width
    cam.set(4,480) #set video heght
    #Use 
    face_detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') #'haarcascade_frontakface_default.xml')
    face_id =  input('\n enter user id end press <return> ==> ')

    print ("\n [INFO] Initializing face capture .Look the camera and wait ...")
    #Initialize individual sampling face count 
       
    count = 0
    while (True):
        ret,img =cam.read()
        gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print(gray)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        #size = 4
        #mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
        #faces = face_detector.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            count += 1
            #save the captured image into the datasets folder 
            cv2.imwrite("images/xuantruong/" +str(face_id)+'.' +str(count)+".jpg",gray[y:y+h,x:x+w])
        cv2.imshow('image',img)
        k = cv2.waitKey(100) & 0xff #Press 'ESC' for exiting video
        print ("count" ,count)
        if k==27:
            break
        elif count >=300: #Take 300 face sample and stop video
            break

    # Do a bit of cleanup
    print ("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
def trainrecognizer():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                #print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                #print(label_ids)
                #y_labels.append(label) # some number
                #x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
                pil_image = Image.open(path).convert("L") # grayscale
                size = (640, 480)
                final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(final_image, "uint8")

                #print(image_array)
                # faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                # for (x,y,w,h) in faces:
                    # roi = image_array[y:y+h, x:x+w]

                x_train.append(image_array)
                y_labels.append(id_)


    print(y_labels)
    print(x_train)

    with open("pickles/face-labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("recognizers/face-trainner.yml") 

def main():
    print("Choice function you want to excute .")
    print("Choice number 1 : Data gathering face ")
    print("Choice number 2 : Train the recognizer ")
    print("Choice number 3 : Recognizer face")
    Choice = int(input())
    if Choice == 1 :
        datagatherface()
    elif Choice == 2 :
        trainrecognizer()
    elif Choice == 3 :
        recognitionface()
    else :
        print("You choice number again !")

if __name__ == '__main__':
    main()