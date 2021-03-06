# Import the required module for text 
# to speech conversion
#from speech_to_text import recog
from gtts import gTTS
 
# This module is imported so that we can 
# play the converted audio
import os
 
# The text that you want to convert to audio

mytext = raw_input("Input the text that you want to: ")
 
# Language in which you want to convert
language = 'en'
 
myobj = gTTS(text=mytext, lang=language, slow=False)
 
# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("welcome.mp3")
 
# Playing the converted file
os.system("play welcome.mp3")
