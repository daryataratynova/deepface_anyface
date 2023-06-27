import argparse
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages
from detectors import detect, deepface_detect
import os
import csv


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=640, help='inference size (pixesls)')
    parser.add_argument('--h', type=int, default=480, help='inference size (pixesls)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence threshold, 0 if not required')
    parser.add_argument('--models', type = str, nargs = '+')
    parser.add_argument('--model_name', type =str, default = 'yolov5l')
    parser.add_argument('--deepface', type = int, default = 1) # 0 false, 1 true
    parser.add_argument('--weights', type = str) # 0 false, 1 true
    parser.add_argument('--img_size', type = int, default= 1240 )
    opt = parser.parse_args()

    if opt.deepface == 0:
        print("hi")
        detect(opt.img_size, opt.conf_thres, opt.model_name, opt.weights)
        print("hi2")
    elif opt.deepface == 1:
        deepface_detect(opt.w, opt.h, opt.conf_thres, opt.models)

