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
        
        sound = AudioSegment.from_mp3(file_path)
        play(sound)
        time.sleep(sound.duration_seconds)

    def overtake(self):
        status = eval(self.data)['status']

        if status:
            # 추월함
            audio_files = [dir + '/audio/choowal.mp3']
            
        else:
            # 추월당함
            audio_files = [dir + '/audio/choowaldang.mp3']

        audio_file = random.choice(audio_files)
        self.playFile(audio_file)
