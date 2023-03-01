from gtts import gTTS
import pyaudio
import wave
import keyboard
import os
import playsound
import openai


chunk = 1024 

# 16 bits per sample
sample_format = pyaudio.paInt16 
chanels = 1
 
# Record at 44400 samples per second
smpl_rt = 44400 


def speak(s):
    myoutput = gTTS(text=s, lang='en', slow=False)
    myoutput.save("C:\\Users\\MaxKa\AI Playground\\Jarvis\\output.mp3")
    playsound.playsound('C:\\Users\\MaxKa\AI Playground\\Jarvis\\output.mp3')
    os.remove('C:\\Users\\MaxKa\AI Playground\\Jarvis\\output.mp3')    
pass

if __name__ == "__main__":


    # Create an interface to PortAudio
    pa = pyaudio.PyAudio() 

    while True:
        frames = [] 
        if(keyboard.read_key() == "r"):

            #recording 
            print('Recording...')
            stream = pa.open(format=sample_format, channels=chanels,
                    rate=smpl_rt, input=True,
                    frames_per_buffer=chunk)
            while(keyboard.is_pressed("r")):
                data = stream.read(chunk)
                frames.append(data)
            stream.stop_stream()
            stream.close()

            #transcripe to wav file

            print('processing...')      
            sf = wave.open("C:\\Users\\MaxKa\AI Playground\\Jarvis\\x.wav", 'wb')
            sf.setnchannels(chanels)
            sf.setsampwidth(pa.get_sample_size(sample_format))
            sf.setframerate(smpl_rt)
            sf.writeframes(b''.join(frames))
            sf.close()
            
            print("Transcribe...")
            file = open("C:\\Users\\MaxKa\AI Playground\\Jarvis\\x.wav", "rb")
            transcription = openai.Audio.transcribe("whisper-1", file)

            print(transcription)

            #call chatGPT api function
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": transcription}])

            speak(completion)


       

