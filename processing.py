import cv2
import yaml
import sys
import os
import numpy as np
from torch import nn
from qaa.models import qe_pipeline
from skimage import transform as trans

sys.path.append("./tddfa_v2/")



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

from tddfa_v2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from tddfa_v2.TDDFA_ONNX import TDDFA_ONNX

# given an image path
def normalize_face(img, kps, src):  # performs face alignment

    if src is None:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0


    _kps = np.array((np.mean(kps[36:41][:2]), np.mean(kps[42:47][:2])))
    kps = np.vstack((_kps, kps[33, 48, 54][:2]))

    dst = kps.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    m = tform.params[0:2, :]
    warped_img = cv2.warpAffine(img, m, (112, 112), borderValue=0.0)

    return np.array(warped_img)


def get_bgr_unif(img):
    # Use graph cut segmentation
    return 100


# Select closest to the center bbox
def sel_cc_bbox(im_shape, bboxes):
    cy, cx = im_shape[0] / 2, im_shape[1] / 2
    clid = 0
    cldist = 999999
    for i, bbox in enumerate(bboxes):
        bcy, bcx = bbox[2] - bbox[0] / 2, bbox[3] - bbox[1] / 2
        dist = np.sqrt(np.power((bcx - cx),2) + np.power((bcy - cy),2))

        if dist < cldist:
            cldist = dist
            clid = i

    return clid


class ImageProcessor(nn.Module):

    def __init__(self, config_path='./tddfa_v2/configs/mb1_120x120.yml'):
        super().__init__()
        self.cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
        self.face_boxes = FaceBoxes_ONNX()
        self.tddfa = TDDFA_ONNX(**self.cfg)
        self.qaa_model = None

    def process(self, img, use_tracking=False, boxes=None, tgt_id=None, ver_lst=None):

        if use_tracking:
            tgt_box = boxes[tgt_id]
            tgt_box = self.calc_tracking(img, prev_box=tgt_box, prev_ver_lst=ver_lst)
            boxes[tgt_id] = tgt_box
        else:
            boxes = self.detect_faces(img)
            tgt_id = sel_cc_bbox(img.shape, boxes)
            tgt_box = boxes[tgt_id]
        ver_lst = self.align_landmarks(img, [tgt_box,])
        face_qual = self.get_face_quality(img, ver_lst)
        bgr_unif = self.get_bgr_unif(img, ver_lst)
        return boxes, tgt_id, ver_lst, face_qual, bgr_unif

    def detect_faces(self, img):
        return self.face_boxes(img)

    def align_landmarks(self, img, boxes):
        param_lst, roi_box_lst = self.tddfa(img, boxes)
        return self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

    def get_face_quality(self, img, ver_lst):
        # normalized_face = normalize_face(img, ver_lst)
        # quality = self.qaa_model(normalized_face)
        # return quality
        return 0

    def get_bgr_unif(self, img, ver_lst):
        # doc_format_img = normalize_face(img, ver_lst, dst_lst)
        # bgr_unif_score = get_bgr_unif(doc_format_img)
        # return bgr_unif_score
        return 0

    def calc_tracking(self, img, prev_box, prev_ver_lst):
        # Use optical flow
        pass
