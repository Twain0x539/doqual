import cv2
import yaml
import sys
import os
import numpy as np
from torch import nn
from skimage import transform as trans

sys.path.append("./tddfa_v2/")



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

from tddfa_v2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from tddfa_v2.TDDFA_ONNX import TDDFA_ONNX

# given an image path
def normalize_face(img, kps):  # performs face alignment

    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    src[:, 0] += 8.0

    kps = np.array(kps)[:, :2]
    dst = kps.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    m = tform.params[0:2, :]
    warped_img = cv2.warpAffine(img, m, (112, 112), borderValue=0.0)

    return np.array(warped_img)

# def estimate_doc_format(img, kps):
#     src = np.array([
#         [30.2946, 51.6963],
#         [65.5318, 51.5014],
#         [48.0252, 71.7366],
#         [33.5493, 92.3655],
#         [62.7299, 92.2041]], dtype=np.float32)
#     src[:, 0] += 8.0
#
#     kps = np.array(kps)[:, :2]
#     dst = kps.astype(np.float32)
#
#     tform = trans.SimilarityTransform()
#     tform.estimate(dst, src)
#     m = tform.params[0:2, :]
#     warped_img = cv2.warpAffine(img, m, (112, 112), borderValue=0.0)
#
#     return np.array(warped_img)
def estimate_doc_format(img, kps):
    return img

def get_bgr_unif(img):
    return 100


# Select closest to the center bbox
def sel_cc_bbox(im_shape, bboxes):
    cy, cx = im_shape[0] / 2, im_shape[1] / 2

    clid = 0
    cldist = 99999
    for i, bbox in enumerate(bboxes):
        bcx, bcy = bbox[2] - bbox[0] / 2, bbox[3] - bbox[1] / 2
        dist = np.sqrt(bcx - cx) + np.sqrt(bcy - cy)

        if dist < cldist:
            cldist = dist
            clid = i

    return clid


class ImageProcessor(nn.module):

    def __init__(self, config_path='./tddfa_v2/configs/mb1_120x120.yml'):
        self.cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
        self.face_boxes = FaceBoxes_ONNX()
        self.tddfa = TDDFA_ONNX(**self.cfg)
        self.qaa_model = None


    def detect_faces(self, img):
        return self.face_boxes(img)

    def align_landmarks(self, img, boxes):
        param_lst, roi_box_lst = self.tddfa(img, boxes)
        return self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

    def get_face_quality(self, img, ver_lst):
        normalized_face = normalize_face(img, ver_lst)
        quality = self.qaa_model(normalized_face)
        return quality

    def get_bgr_unif(self, img, ver_lst):
        doc_format_img = estimate_doc_format(img, ver_lst)
        bgr_unif_score = get_bgr_unif(doc_format_img)
        return bgr_unif_score