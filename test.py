
from deepface.DeepFace import extract_faces
pred  = extract_faces("fairface/train/69470.jpg", \
                        conf_thres = 0.005, detector_backend =  "retinaface", \
                        align = False, enforce_detection= False)
print(pred[0]['confidence'])
