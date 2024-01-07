import cv2
import yaml
import sys
sys.path.append("./tddfa_v2/")

from tddfa_v2.FaceBoxes import FaceBoxes
from tddfa_v2.TDDFA import TDDFA
from tddfa_v2.utils.functions import draw_landmarks
from tddfa_v2.utils.render import render
from tddfa_v2.utils.depth import depth

import matplotlib.pyplot as plt

# load config
cfg = yaml.load(open('./tddfa_v2/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    from tddfa_v2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from tddfa_v2.TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    tddfa = TDDFA(gpu_mode=False, **cfg)
    face_boxes = FaceBoxes()

# given an image path
img_fp = './tddfa_v2/examples/inputs/emma.jpg'
img = cv2.imread(img_fp)
plt.imshow(img[..., ::-1])

# face detection
boxes = face_boxes(img)
print(f'Detect {len(boxes)} faces')
print(boxes)

param_lst, roi_box_lst = tddfa(img, boxes)
dense_flag = False
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
draw_landmarks(img, ver_lst, dense_flag=dense_flag, show_flag=True)

while(True):
    False