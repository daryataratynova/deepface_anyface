import argparse
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages
from deepface_detector import detect
import os
import csv


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=640, help='inference size (pixesls)')
    parser.add_argument('--h', type=int, default=480, help='inference size (pixesls)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence threshold, 0 if not required')
    parser.add_argument('--models', type = str, nargs = '+')
    parser.add_argument('--deepface', type = int, default = 1) # 0 false, 1 true
    opt = parser.parse_args()

    if opt.deepface == 0:
        print("will be later")
    elif opt.deepface == 1:
        detect(opt.w, opt.h, opt.conf_thres, opt.models)

