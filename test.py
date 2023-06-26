import cv2
import torch
# from deepface.DeepFace import extract_faces
# pred  = extract_faces("fairface/train/62625.jpg", \
#                         conf_thres = 0.5, detector_backend =  "ssd", \
#                         align = False, enforce_detection= T)
# print(pred[0]["confidence"])

image_read = cv2.imread("fairface/train/12639.jpg")
dim = (1280, 1280)
resized = cv2.resize(image_read, dim, interpolation = cv2.INTER_AREA)

resized = resized.transpose(2, 0, 1).copy()

resized = torch.from_numpy(resized)
resized = resized.float()  # uint8 to fp16/32
resized /= 255.0  # 0 - 255 to 0.0 - 1.0
if resized.ndimension() == 3:
    resized = resized.unsqueeze(0)

print(resized)