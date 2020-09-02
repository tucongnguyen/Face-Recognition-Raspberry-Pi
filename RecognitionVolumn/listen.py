import speech_recognition
#Khoi tao nen 1 ham nghe 
robot_ear =  speech_recognition.Recognizer()

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