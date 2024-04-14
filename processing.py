import cv2
import yaml
import sys
import os
import numpy as np
from torch import nn
from qaa.models import qe_pipeline
from skimage import transform as trans
from recognition.models.iresnet import *
import torch
from torchvision import transforms

sys.path.append("./tddfa_v2/")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

from tddfa_v2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from tddfa_v2.TDDFA_ONNX import TDDFA_ONNX

def normalize_face(img, ver_lst, src=None, output_size=(112,112)):  # performs face alignment

    if src is None:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0

    _kps = np.array((np.mean(ver_lst[36:41,:2], axis=0), np.mean(ver_lst[42:47,:2], axis=0)))

    kps = np.vstack((_kps, ver_lst[[33, 48, 54],:2]))

    dst = kps.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    m = tform.params[0:2, :]
    warped_img = cv2.warpAffine(img, m, output_size, borderValue=0.0)
    return np.array(warped_img)


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
        self.qaa_model = iresnet18()
        self.qaa_model.load_state_dict(torch.load("./recognition/weights/ms1mv3_arcface_r18_fp16.pth", map_location=device))
        self.qaa_model.eval()
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     ])

    def process(self, img):

        boxes = self.detect_faces(img)

        tgt_id = None
        ver_lst = None
        face_qual = None
        bgr_unif = None
        print(boxes)
        if len(boxes) >= 1:
            tgt_id = sel_cc_bbox(img.shape, boxes)
            tgt_box = boxes[tgt_id]
            ver_lst = self.align_landmarks(img, [tgt_box,])
            face_qual = self.get_face_quality(img, ver_lst)
            bgr_unif, tr_img = self.get_bgr_unif(img, ver_lst, tgt_box)
        print("Face Qual and BGRU computed")
        return boxes, tgt_id, ver_lst, face_qual, bgr_unif

    def detect_faces(self, img):
        return self.face_boxes(img)

    def align_landmarks(self, img, boxes):
        param_lst, roi_box_lst = self.tddfa(img, boxes)
        return self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

    def get_face_quality(self, img, ver_lst):
        ver_lst = ver_lst[0]
        ver_lst = np.swapaxes(ver_lst, 0, 1)
        normalized_face = normalize_face(img, ver_lst)
        print("Face was normalized")
        normalized_face = self.transform(normalized_face).unsqueeze(0)
        with torch.no_grad():
            quality = self.qaa_model(normalized_face)
            quality = torch.norm(quality).detach().cpu().item()

            quality = quality / 0.27
            quality = np.clip(quality, a_min=0, a_max= 100).astype("uint8")
        print("Face quality estimated")
        return quality

    def get_bgr_unif(self, img, initial_face_points, bbox):
        # img = cv2.resize(img, (300, 300))
        # mask = np.zeros(img.shape[:2], np.uint8)
        # for y,x in initial_face_points:
        #     mask[y][x] = 1 # Initial foreground
        # H, W = img.shape[:2]
        # sure_background = [(0,0), (0, W-1), (H-1, 0), (H-1, W-1)]

        # print(bbox)
        # bgdModel = np.zeros((1, 65), np.float64)
        # fgdModel = np.zeros((1, 65), np.float64)
        # rect = (bbox[1], bbox[0], bbox[3], bbox[2])
        #
        # print("Trying GrabCut")
        # mask, _, _ = cv2.grabCut(img, None, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # print("GrabCut successful")
        # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # tr_img = img * mask2[:, :, np.newaxis]
        # return 100, tr_img
        return 100, None