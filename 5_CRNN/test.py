import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

img_rows,img_cols,img_depth=120,160,3
X_tr=[]  

listing = os.listdir('./dataset/boxing')[:1]

for vid in listing:
    vid = './dataset/boxing/'+vid
    frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
  

    for k in range(15):
        ret, frame = cap.read()
        # frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray)
        # plt.show()
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    input=np.array(frames)

    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)

    X_tr.append(ipt)

print(X_tr[0].shape)
# plt.imshow(gray)
#     plt.show()
#     frames.append(gray)

