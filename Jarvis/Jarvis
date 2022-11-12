import whisper
from gtts import gTTS
import pyaudio
import wave
import keyboard
import os
import playsound


chunk = 1024 

# 16 bits per sample
sample_format = pyaudio.paInt16 
chanels = 1
 
# Record at 44400 samples per second
smpl_rt = 44400 
seconds = 4
filename = "path_of_file.wav"
 
# Create an interface to PortAudio
pa = pyaudio.PyAudio() 

#load whisper model
model = whisper.load_model("base")
 

while True:
    frames = [] 
    if(keyboard.read_key() == "r"):
        print('Recording...')
        stream = pa.open(format=sample_format, channels=chanels,
                 rate=smpl_rt, input=True,
                 frames_per_buffer=chunk)
        while(keyboard.is_pressed("r")):
            data = stream.read(chunk)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        print('processing...')      
        sf = wave.open("C:\\Users\\MaxKa\AI Playground\\Jarvis\\x.wav", 'wb')
        sf.setnchannels(chanels)
        sf.setsampwidth(pa.get_sample_size(sample_format))
        sf.setframerate(smpl_rt)
        sf.writeframes(b''.join(frames))
        sf.close()
        print("Transcribe...")
        result = model.transcribe("C:\\Users\\MaxKa\AI Playground\\Jarvis\\x.wav")
        print(result["text"])
        myoutput = gTTS(text=result["text"], lang='en', slow=False)
        myoutput.save("C:\\Users\\MaxKa\AI Playground\\Jarvis\\output.mp3")
        playsound.playsound('C:\\Users\\MaxKa\AI Playground\\Jarvis\\output.mp3')

