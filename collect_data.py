import os
import sys
from glob import glob
from multiprocessing import Pool, cpu_count
from itertools import repeat

import cv2
import numpy as np
from face_recognition import face_locations


#------------------------------------------------------------------------------
#   Draw face localtions on an image
#------------------------------------------------------------------------------
def draw_face_locations(img, face_locs):
    corner1 = []
    corner2 = []
    len_locs = len(face_locs)
    img_new = np.array(img)
    if len_locs:
        corner1 = np.zeros([len_locs, 2], dtype=int)
        corner2 = np.zeros([len_locs, 2], dtype=int)
        cl_blue = (255, 0, 0)
        i = 0
        for (x,y,w,h) in face_locs:
            c1 = (h, x)
            c2 = (y, w)
            corner1[i,:] = c1
            corner2[i,:] = c2
            cv2.rectangle(img_new, c1, c2, cl_blue, 3)
            i += 1
    return img_new, corner1, corner2


#------------------------------------------------------------------------------
#	Detect blurness
#------------------------------------------------------------------------------
def detect_blur(img, thres=None):
    n_channels = len(img.shape)
    var = 0
    for c in range(n_channels):
        channel = img[...,c]
        var += cv2.Laplacian(channel, cv2.CV_64F).var()

    var = var/n_channels
    if thres==None:
        return var
    else:
        return True if var>=thres else False, var


#------------------------------------------------------------------------------
#	Input student ID first
#------------------------------------------------------------------------------
input_name = input(">> Name: ")

# Check the existance
folder = os.path.join("raw_dataset", input_name)
if not os.path.exists(folder):
    os.makedirs(folder)
    save_id = 0
else:
    key = input(">> Name has already existed. Append? [y/n] ")
    save_id = len(glob(os.path.join(folder, "*.*")))
    if key=="n" or key=="N":
        sys.exit("Exit the application.")


#------------------------------------------------------------------------------
#	Calibrate the blurness ===> Need to be parallelized
#------------------------------------------------------------------------------
print("Wait for calibrating the blurness...")

cap = cv2.VideoCapture(0)
frames = []
for i in range(50):
    _, frame = cap.read()
    frames.append(frame)

pools = Pool(processes=cpu_count())
args = zip(frames, repeat(None))
blurness_vars = pools.starmap(detect_blur, args)

blurness_thres = sum(blurness_vars) / len(blurness_vars) * 0.90
print("Finish calibrating the blurness: %f" % blurness_thres)
print("--------------------------------------------------------\n")


#------------------------------------------------------------------------------
#	Collect samples of image
#------------------------------------------------------------------------------
frame_idx = 0
while(True):
    # Read frame
    if frame_idx<10:
        frame_idx += 1
        continue

    ret, frame = cap.read()
    print("[FrameID %d] " % frame_idx)
    frame_idx += 1

    # Detect blur
    blurness, var = detect_blur(frame, thres=blurness_thres)

    # Detect face
    if blurness:
        face_locs = face_locations(frame)
        img, _, _ = draw_face_locations(frame, face_locs)
    else:
        img = frame




    # Save the frame in handy mode
    # if blurness and len(face_locs)==1:
    # 	key = input(">> Do you want to save the frame? [y/n] ")
    # 	if key=="y" or key=="Y":
    # 		filename = os.path.join(folder, "%s_%d.png" % (input_name, save_id))
    # 		cv2.imwrite(filename, frame)
    # 		save_id += 1
    # print("--------------------------------------------------------\n")

    # Save the frame in automatic mode
    if blurness and len(face_locs)==1:
        filename = os.path.join(folder, "%s_%d.png" % (input_name, save_id))
        cv2.imwrite(filename, frame)
        save_id += 1
    print("Number of saved frames: %d" % (save_id))
    print("--------------------------------------------------------\n")
    # Show the frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()