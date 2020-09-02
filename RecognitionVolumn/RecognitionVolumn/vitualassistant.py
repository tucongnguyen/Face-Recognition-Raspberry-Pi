import speech_recognition

import pyttsx3

from datetime import date , datetime
#Khoi tao nen 1 ham nghe 
robot_ear =  speech_recognition.Recognizer()
#khoi tao 1 ham noi
robot_mouth = pyttsx3.init()
#Khoi tao 1 ham nghi
robot_brain = ""

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
	finally:
		pass


	print( "You " + you)

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
		break
	else:
		robot_brain = "I'm fine thank you and you"
		
	print ("Robot :" + robot_brain)


	robot_mouth.say(robot_brain)
	robot_mouth.runAndWait()