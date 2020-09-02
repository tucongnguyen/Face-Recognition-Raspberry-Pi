import numpy as np
import cv2
import pickle

import speech_recognition

import pyttsx3

from datetime import date , datetime

import time

#Khoi tao nen 1 ham nghe 
robot_ear =  speech_recognition.Recognizer()
#khoi tao 1 ham noi
robot_mouth = pyttsx3.init()
#Khoi tao 1 ham nghi
robot_brain = ""

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

def vitualassistant():
	robot_mouth.say("Well come to us house ,Can i helpt you ? ")
	robot_mouth.runAndWait()

	while True:
		
		#su dung with nghia la khi nghe xong roi se tat di
		with speech_recognition.Microphone() as mic:
			print("Robot: I'm listening")
			audio = robot_ear.listen(mic)

		#Try catch Error
		try:
			you = robot_ear.recognize_google(audio)
		except Exception as e:
			you = "" 
		if you == "":
			robot_brain = "I can't hear you , try again"
		elif "hello" in you:
			robot_brain = "Hello ! Well come to paradise "
		elif "today"  in you:
			today = date.today()
			d2 = today.strftime("%B %d, %Y")
			robot_brain =d2
		elif "time" in you :
			now = datetime.now()
			robot_brain = now.strftime("%H hours %M minutes %S seconds")
		elif "bye" in you :
			robot_brain = "Bye"
			print ("Robot :" + robot_brain)
			robot_mouth.say(robot_brain)
			robot_mouth.runAndWait()
			break
		else:
			robot_brain = "I'm fine thank you and you"
			
		print ("Robot :" + robot_brain)


		robot_mouth.say(robot_brain)
		robot_mouth.runAndWait()


def guest():
	robot_mouth.say("Hello , This is Cong Tu's house . ")
	robot_mouth.runAndWait()


cap = cv2.VideoCapture(0)
i = 0
face_id = 1
bolean = False
while(True):
    # Capture frame-by-frame
  
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(roi_gray)
        if conf>=47 and conf <= 54:
        #print(5: #id_)
        #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            i = 0
            print('Confidence ', conf)
            vitualassistant()
            bolean = True
        #
        else :
            color = (255, 255, 255)
            stroke = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            noname ="Unknow"
            cv2.putText(frame,noname, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            if noname =="Unknow":
            	i += 1
            	print(i)
            	if i == 30:
            		i = 0
            		face_id += 1 
            		today = date.today()
					d2 = today.strftime("%B %d, %Y") 
            		cv2.imwrite("images/imagestranger/" +str(face_id)+'.'+str(d2)".jpg",roi_color)
					
			#save the captured image into the datasets folder
		# if i == 20:
		 	 
		# 	cv2.imwrite("images/imagestranger/" +str(face_id)+".jpg",roi_color)
		# 	i == 0

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
    # Display the resulting frame
    cv2.imshow('frame',frame)
    print(bolean)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the captureqq
cap.release()
cv2.destroyAllWindows()


