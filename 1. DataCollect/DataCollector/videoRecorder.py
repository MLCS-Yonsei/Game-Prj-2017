
import subprocess 
import multiprocessing as mp
from threading import Thread
from multiprocessing import Pool
from queue import Empty

import time
import datetime
import os
import signal

class RecordWorker(mp.Process):
    def __init__(self,que,i,resolution,record_time):
        super(Recorder,self).__init__()
        self.queue = que
        self.i = i
        self.resolution = resolution
        if isinstance(record_time, datetime.datetime):
            self.record_time = record_time.strftime('%Y%m%d%H%M%S')
        else:
            self.record_time = record_time

    def run(self):
        i = self.i
        print("Getting Video Source from /dev/video" + str(i))
        video_source = "/dev/video" + str(i)
        
        print("Saving in video/" + self.record_time + "_" + str(i))
        cmd = ["ffmpeg", "-y", "-loglevel", "panic", "-f", "v4l2", "-thread_queue_size", "512", "-i", video_source, "-f", "alsa", "-thread_queue_size", "512", "-i", "hw:"+ str(i+2) +",0", "-vcodec", "libx264", "-s", self.resolution, "-r", "30", "-aspect", "16:9", "-acodec", "libmp3lame", "-b:a", "128k", "-channels", "2", "-ar", "48000", "video/" + self.record_time + "_" + str(i) + ".mp4"]
        self.subprocess = subprocess.Popen(cmd)

        while True:
            a = self.subprocess.poll()
            if a is None:
                time.sleep(1)
                try:
                    if self.queue.get(0) == "exit":
                        print("Stopping Video Recording from /dev/video" + str(i))
                        self.subprocess.terminate()
                        # self.subprocess.wait()
                        break
                    else:
                        pass
                except Empty:
                    pass
                # print("run")
            # else:
            #     print("exiting")
        

class VideoRecorder():

    def __init__(self, camera_number, resolution, record_time):
        self.jobs = []
        self.queues = []
        self.camera_number = camera_number
        self.resolution = resolution
        self.record_time = record_time

    def start(self):
        for i in range(self.camera_number):
            q = mp.Queue()
            job = RecordWorker(q, i, self.resolution, self.record_time)
            self.queues.append(q)
            self.jobs.append(job)
            job.start()

    def stop(self):
        for q in self.queues:
            q.put("exit")


if __name__ == "__main__":
    current_time = datetime.datetime.now()

    camera_number = 2
    resolution = "1280x720"

    vr = VideoRecorder(camera_number, resolution, current_time)
    vr.start()
    # wait for a while and then kill jobs
    time.sleep(6)
    vr.stop()

    time.sleep(2)