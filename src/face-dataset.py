import cv2 
import os


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
		cv2.imwrite("images/congtu/" +str(face_id)+'.' +str(count)+".jpg",gray[y:y+h,x:x+w])
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