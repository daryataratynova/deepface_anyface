from deepface.DeepFace import extract_faces
import argparse
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages
import os
import csv
import glob
import cv2


def detect(
    img_w,
    img_h,
    confidence,
    models
):
    data_folders = ['fairface/train/*.*', 'fairface/val/*.*']

    for path in data_folders:
        for model in models:
            for image_path in glob.glob(path):
                image_read = cv2.imread(image_path)
                dim = (img_w, img_h)
                resized = cv2.resize(image_read, dim, interpolation = cv2.INTER_AREA)
    
                try: #try to get a face
                    pred  = extract_faces(resized, \
                        conf_thres = confidence, detector_backend =  model, \
                        align = False, enforce_detection= True)
                    found = True
                except: #if we cannot get a face we set status to Fale
                        found = False

                if (found == False): #if model could not find a face then we save path
                    if  'train' in path:
                            file  = "fairface/"+ model + "/train_undetected.csv"
                            file_exists = os.path.isfile(file)
                    else:
                            file  = "fairface/" + model + "/val_undetected.csv"
                            file_exists = os.path.isfile(file)

                    with open (file, 'a') as csvfile:
                        headers = ['file']
                        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow({'file': image_path})