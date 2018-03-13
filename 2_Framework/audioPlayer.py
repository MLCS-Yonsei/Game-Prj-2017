from pydub import AudioSegment
from pydub.playback import play

import random

import time
import os
dir = os.path.dirname(os.path.abspath(__file__))

class audioPlayer():
    def __init__(self, result):
        self.method = getattr(self, result['flag'], lambda: "nothing")
        
        self.data = result['data']
        self.target_ip = result['target_ip']

        self.method()

    def playFile(self, file_path):
        
        sound = AudioSegment.from_wav(dir + file_path)
        play(sound)
        time.sleep(sound.duration_seconds)

    def overtake(self):
        status = eval(self.data)['status']

        if status:
            # 추월함
            audio_files = ['/audio/overtake-01.wav','/audio/overtake-02.wav','/audio/overtake-03.wav']
            
        else:
            # 추월당함
            audio_files = ['/audio/overtake-04.wav','/audio/overtake-05.wav','/audio/overtake-06.wav']

        audio_file = random.choice(audio_files)
        self.playFile(audio_file)
