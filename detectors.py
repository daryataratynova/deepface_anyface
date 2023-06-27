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
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.datasets import letterbox, LoadImages
import numpy as np
import copy

def load_model(weights, device):
    print('i am trying to load a model')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.02 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def deepface_detect(
    img_w,
    img_h,
    confidence,
    models
):
    data_folders = [ 'fairface/val/*.*', 'fairface/train/*.*']

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
    img_size,
    confidence,
    model_name,
    weights
):
    print('i am in detect 1')
    iou_thres = 0.5
    print('i am going to read')
    data_folders = ['fairface/train/', 'fairface/val/']
    print('i read folder')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('i have set device')
    model = load_model(weights, device)
    print('i loaded model')
    imgsz=(img_size, img_size)
    print('i am in detector')
    for folder_path in data_folders:
        dataset = LoadImages(folder_path, img_size=imgsz)
        print('i am in folder')
        for path, im, im0s, vid_cap in dataset:
            if len(im.shape) == 4:
                orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
            else:
                orgimg = im.transpose(1, 2, 0)
            
            orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
            img0 = copy.deepcopy(orgimg)
            h0, w0 = orgimg.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

            imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

            img = letterbox(img0, new_shape=imgsz)[0]
            # Convert from w,h,c to c,w,h
            img = img.transpose(2, 0, 1).copy()

            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img)[0]

            
            # Apply NMS
            pred = non_max_suppression_face(pred, confidence, iou_thres)
            print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
        
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                    for j in range(det.size()[0]):
                        xyxy = det[j, :4].view(-1).tolist()
                        conf = det[j, 4].cpu().numpy()
                        landmarks = det[j, 5:15].view(-1).tolist()
                        class_num = det[j, 15].cpu().numpy()
                        
                        im0 = show_results(im0, xyxy, conf, landmarks, class_num)
            if len(pred[0]) == 0: 
                found = False 
            else: 
                found = True
            print(path)
            if (found == False): #if model could not find a face then we save path
                if  'train' in path:
                        file  = "fairface/"+ model_name+ '/' + str(confidence) + '_train_undetected' + str(img_size) + '.csv'
                        file_exists = os.path.isfile(file)
                else:
                        file  = "fairface/" + model_name + '/'+ str(confidence) + '_val_undetected'+ str(img_size) +'.csv'
                        file_exists = os.path.isfile(file)
                        print('i am writing in val')
                with open (file, 'a') as csvfile:
                    headers = ['file']
                    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow({'file': path})
