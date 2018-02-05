# import the necessary packages
from __future__ import print_function
from imutils.video import FPS
import argparse
import imutils
import cv2
import pyaudio
import wave
import subprocess
import os
import threading
from time import sleep
import numpy as np
import tkinter as tk

p = pyaudio.PyAudio()
print("Looking for Audio Sources..")
for i in range(p.get_device_count()):
    if p.get_device_info_by_index(i).get('maxInputChannels') > 0:
        print("Found : ", p.get_device_info_by_index(i)["index"], p.get_device_info_by_index(i)["name"])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-mdi", "--maximum-device-index", type=int, default=4,
	help="Maximum device index to check for the video source")
args = vars(ap.parse_args())

class VideoGrabber():

    def __init__(self):
        self.grabbed = False

        self.device_list = []
        self.device_dic = {}

        for i in range(args["maximum_device_index"]):
            print("Looking for Video Source.. index : " + str(i))
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print("Found : " + str(i))
                self.device_list.append(i)
            cap.release()

    def grab_video(self):
        for device_idx in self.device_list:
            # created a *threaded* video stream, allow the camera sensor to warmup,
            # and start the FPS counter
            print("[INFO] sampling THREADED frames from webcam from Device : " + str(device_idx))
            self.device_dic["vs" + str(device_idx)] = cv2.VideoCapture(device_idx)
            self.device_dic["fps" + str(device_idx)] = None

            # initiating video writer
            # for mac Use mp4v to .mov (refer https://gist.github.com/takuma7/44f9ecb028ff00e2132e -> codec list)
            self.device_dic["fourcc" + str(device_idx)] = cv2.VideoWriter_fourcc(*'avc1')
            self.device_dic["writer" + str(device_idx)] = None
            (self.device_dic["h" + str(device_idx)],self.device_dic["w" + str(device_idx)]) = (None, None)
            self.device_dic["zeros" + str(device_idx)] = None

            # initiating audio writer
            #self.device_dic["audio_thread" + str(device_idx)] = AudioRecorder()

        # Loop that grabs video
        while not self.stopped:
            for device_idx in self.device_list:
                # Initiate FPS counter
                if self.device_dic["fps" + str(device_idx)] == None:
                    self.device_dic["fps" + str(device_idx)] = FPS().start()

                # grab the frame from the threaded video stream 
                (self.grabbed, self.device_dic["frame" + str(device_idx)]) = self.device_dic["vs" + str(device_idx)].read()

                # update the FPS counter
                self.device_dic["fps" + str(device_idx)].update()

                # and resize it
                # to have a maximum width of * pixels
                self.device_dic["frame" + str(device_idx)] = imutils.resize(self.device_dic["frame" + str(device_idx)], width=720)
                
                if self.device_dic["writer" + str(device_idx)] is None:
                    # store the image dimensions, initialzie the video writer,
                    # and construct the zeros array
                    print(self.device_dic["vs" + str(device_idx)].get(cv2.CAP_PROP_FPS))
                    (self.device_dic["h" + str(device_idx)],self.device_dic["w" + str(device_idx)]) = self.device_dic["frame" + str(device_idx)].shape[:2]
                    self.device_dic["writer" + str(device_idx)] = cv2.VideoWriter("./video/device" + str(device_idx) + ".mov", self.device_dic["fourcc" + str(device_idx)], 24, (self.device_dic["w" + str(device_idx)],self.device_dic["h" + str(device_idx)]), True)
                    zeros = np.zeros((self.device_dic["h" + str(device_idx)],self.device_dic["w" + str(device_idx)]), dtype="uint8")

                self.device_dic["writer" + str(device_idx)].write(self.device_dic["frame" + str(device_idx)])

        # Stopping Video Grabber
        if self.stopped:
            for device_idx in self.device_list:
                # Do some cleanup and Stop FPS counter
                self.device_dic["writer" + str(device_idx)].release()
                self.device_dic["fps" + str(device_idx)].stop()
                self.device_dic["vs" + str(device_idx)].release()
                
                print("[INFO] elasped time: {:.2f}".format(self.device_dic["fps" + str(device_idx)].elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(self.device_dic["fps" + str(device_idx)].fps()))
                
                #cv2.destroyAllWindows()

    def stop(self):
        print("Stopping Video Grabber..")

        self.stopped = True
        self.video_thread.join()

        # Convert the video file with calculated FPS
        for device_idx in self.device_list:
            cmd = "ffmpeg -y -r " + str(self.device_dic["fps" + str(device_idx)].fps()) + " -i " + "./video/device" + str(device_idx) + ".mov" + " -r 30 ./video/device" + str(device_idx) + ".mp4"
            subprocess.call(cmd, shell=True)

    def start(self):
        self.video_thread = threading.Thread(target=self.grab_video)
        self.video_thread.daemon = True
        self.stopped = False

        self.video_thread.start()
        #return self

# vg = VideoGrabber()
# def key(event):
#     print(event.char)
#     if event.char == "s":
#         vg.start()
#     elif event.char == "k":
#         vg.stop()


# #Set up GUI
# window = tk.Tk()  #Makes main window
# window.wm_title("Motion Device")
# window.config(background="#FFFFFF")

# window.bind_all('<Key>', key)

# #Graphics window
# imageFrame = tk.Frame(window)
# imageFrame.grid(row=0, column=0, padx=10, pady=2)

# #Capture video frames
# lmain0 = tk.Label(imageFrame)
# lmain0.pack()

# #Capture video frames
# lmain1 = tk.Label(imageFrame)
# lmain1.pack()

# #Capture video frames
# lmain2 = tk.Label(imageFrame)
# lmain2.pack()

# #Slider window (slider controls stage position)
# sliderFrame = tk.Frame(window, width=600, height=100)
# sliderFrame.grid(row = 600, column=0, padx=10, pady=2)

# window.mainloop()  #Starts GUI


# loop over some frames...this time using the threaded stream
# while self.device_dic["fps" + str(0)]._numFrames < args["num_frames"]:
    
    

    

