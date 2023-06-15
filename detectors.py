from deepface.DeepFace import extract_faces
import argparse
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages
import os
import csv
import glob
import cv2
from models.experimental import attempt_load
import torch
from utils.general import non_max_suppression_face

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def deepface_detect(
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
                    if pred[0]["confidence"] > confidence:
                        found = True
                    else:
                        found = False
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
def detect(
    img_w,
    img_h,
    confidence,
    model_name,
    weights
):
    iou_thres = 0.
    data_folders = ['fairface/train/*.*', 'fairface/val/*.*']

    for path in data_folders:

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(weights, device)

            for image_path in glob.glob(path):
                image_read = cv2.imread(image_path)
                dim = (img_w, img_h)
                resized = cv2.resize(image_read, dim, interpolation = cv2.INTER_AREA)

                resized = resized.transpose(2, 0, 1).copy()

                resized = torch.from_numpy(resized)
                resized = resized.float()  # uint8 to fp16/32
                resized /= 255.0  # 0 - 255 to 0.0 - 1.0
                if resized.ndimension() == 3:
                    resized = resized.unsqueeze(0)

                pred = model(resized)[0]
                pred = non_max_suppression_face(pred, confidence, iou_thres)

                if len(pred[0]) == 0: 
                    found = False 
                else: 
                    found = True
                print(pred, found)

                if (found == False): #if model could not find a face then we save path
                    if  'train' in path:
                            file  = "fairface/"+ model_name + "/train_undetected.csv"
                            file_exists = os.path.isfile(file)
                    else:
                            file  = "fairface/" + model_name + "/val_undetected.csv"
                            file_exists = os.path.isfile(file)

                    with open (file, 'a') as csvfile:
                        headers = ['file']
                        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow({'file': image_path})
