import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from face_detection import RetinaFace
from threading import Thread
from playsound import playsound
import time
from pygame import mixer
import pygame
from multiprocessing import Process


w = 640
h = 480

detector = RetinaFace()
thresh = 0.6

def detect_and_predict_mask(frame, maskNet):
    # membuat dan menentukan dimensi deteksi wajah
    (h, w) = frame.shape[:2] #dari indeks 0 hingga indeks 1, mengambil 2 element pertama
    #kita cuma mengambil 2 indeks, yaitu indeks height dan width
    detections = detector(frame)

    # lokaliaisi wajah dan preds yg didapatkan
    faces = []
    locs = []
    preds = []

    for face in detections:
        # confidence dalam pendeteksian
        box, landmarks, confidence = face #memasukkan semua nilai box, ldmrks, dan cnfdnc dari variabel face

        # menentukan confidence, jika confidence lebih besar dari args
        if confidence >= thresh:
            # memberikan bounding box
            # box = int(max(0, box[0])), int(max(0, box[1])), int(max(0, box[2])), int(max(0, box[3]))
            (startX, startY, endX, endY) = box.astype("int") #bounding box yg diconvert ke bentuk int

            # memastikan bounding box berada di wajah
            (startX, startY) = (max(0, startX), max(0, startY)) #outputnya bisa jadi mines, jadi diambil nilai 0
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY)) #karena 256-1

            # convert deteksi face
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face) #convert PIL image to numpy array
            face = preprocess_input(face) #memberikan identity number ke setiap batch image, dari 3d ke 4d

            # mengintegrasikan bounding box ke face detector
            faces.append(face)
            locs.append((startX, startY, endX, endY))
        else:
            print("rejected", confidence)
    # membuat prediksi jika ada wajah yang ditemukan, minimal 1
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32") #mengubah nilai warna pixel int ke float
        preds = maskNet.predict(faces, batch_size=32) #jumlah komputasi yang dapat dilakukan dalam 1x proses

    return (locs, preds)


# menload model dari hdd
print("Sedang membuka model..,")
maskNet = load_model("mask_detector.h5")

print("Memulai video streaming ...")

class WebcamStream :
    def __init__(self, stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream 
        self.stopped = True 

        # reference to the thread for reading next available frame from input stream 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing 
        
    # method for starting the thread for grabbing next available frame in input stream 
    def start(self):
        self.stopped = False
        self.t.start() 

    # method for reading next frame 
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()

    # method for returning latest read frame 
    def read(self):
        return self.frame

    # method called to stop reading frames 
    def stop(self):
        self.stopped = True 

webcam_stream = WebcamStream(stream_id=0) 
webcam_stream.start()

def warning_sound():
    playsound('masker_warning.mp3')


# processing frames in input stream
num_frames_processed = 0 
start = time.time()
count = 0
output_path = "detected_object"
while True :
    if webcam_stream.stopped is True :
        break
    else :
        frame = webcam_stream.read()
        frame = cv2.resize(frame,(w,h))
        
        # menentukan face memakai masker atau tidak
        (locs, preds) = detect_and_predict_mask(frame, maskNet)

        # jika ada pendeteksi face, loop
        for (box, pred) in zip(locs, preds):
            # unpack prediction yang telah dibuat
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # isUseMask = mask > 0.56 & withoutMask < 0.75
            # memberikan warna dan label kepada pendeteksian

            if mask > withoutMask :
                label = "Mask"
            else :
                label = "No Mask"
                t = Thread(target=warning_sound)
                t.start()
                t.join(timeout=1)
                # name = "frame%d.jpg"%count
                # cv2.imwrite(os.path.join(output_path, name), frame) 

                count = count + 1

            if label == "Mask" :
                color = (0, 255, 0)
            else :
                color = (0, 0, 255)


            # menuliskan probability ketika wajah terdeteksi
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # output
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2) 

    # adding a delay for simulating time taken for processing a frame 
    delay = 0.3 # delay value in seconds. so, delay=1 is equivalent to 1 second 
    time.sleep(delay) 
    num_frames_processed += 1 

    cv2.imshow('frame' , frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()
webcam_stream.stop() # stop the webcam stream 

# printing time elapsed and fps 
elapsed = end-start
fps = num_frames_processed/elapsed 
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows 
cv2.destroyAllWindows()
