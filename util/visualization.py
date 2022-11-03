import os
import sys

import cv2
import matplotlib
import json
import numpy as np
from skimage import io,data
from typing import Iterable
import torch
from torch import device, nn
import torchvision.transforms as TT 
import datasets.transforms as T
import math
from PIL import Image
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch.nn.functional as F


transform = TT.Compose([
    TT.Resize(800),
    TT.ToTensor(),
    TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


@torch.no_grad()
class output_visualizer:
    def __init__(self, model, postprocessors, img_paths, ann_path, img_ids, dataset_file, threshold):
        self.model = model
        self.threshold = threshold
        self.img_paths = img_paths #path = '../deformable_detr_mainstream/input/caltech/set00/V007/images/I00559.jpg'
        self.ann_path = ann_path
        self.img_ids = img_ids
        self.pil_img = None
        self.ann = None
        self.attn = None
        self.tensor_img = None
        self.device = torch.device('cuda')
        self.dataset = dataset_file


    def compute_on_image(self, im):
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)
        img = img.to(self.device)

        convert_tensor = TT.ToTensor()

        size = im.size

        model = self.model
        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        #probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu().detach().numpy()
        #import pdb;pdb.set_trace()
        probas = outputs['pred_logits'].sigmoid()[0, :, :].cpu()
        keep = probas.max(-1).values > self.threshold
        # convert boxes from [0; 1] to image scales
        
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), size)
        return outputs, probas, bboxes_scaled, keep

    
    def standard_visualization(self):

        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        colors = COLORS * 100

        label_list = ['background','person','people','person-fa','person?']
        CLASSES = [
        'object', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush']

        for img_id, img_path in zip(self.img_ids, self.img_paths):

            self.pil_img = Image.open(img_path)

            from pycocotools.coco import COCO
            coco = COCO(self.ann_path)
            ann_id = coco.getAnnIds(imgIds=img_id)
            targets = coco.loadAnns(ann_id)

            #outputs, attention = self.compute_on_image(self.pil_img)
            outputs, probas, bboxes_scaled, keep = self.compute_on_image(self.pil_img)

            dpi = 80
            scale = 2

            width, height = self.pil_img.width, self.pil_img.height

            figsize = scale * width / float(dpi), scale * height / float(dpi)

            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.imshow(self.pil_img)
            ax.axis('off')
            ax.axis('tight')

            # heatmap = np.uint8(255 * attention)
            # ax.imshow(heatmap, alpha=0.35)

            # Configure axis
            ax = plt.gca()

            dpi = 80
            scale = 2

            width, height = self.pil_img.width, self.pil_img.height

            figsize = scale * width / float(dpi), scale * height / float(dpi)

            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.imshow(self.pil_img)
            ax.axis('off')
            ax.axis('tight')
            ax = plt.gca()
            
            prob = probas[keep]
            boxes = bboxes_scaled

            for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3))
                cl = p.argmax()
                text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))
            '''
            if targets is not None:
                
                for i in range(len(targets)):
                    groundtruth = targets[i]['bbox']

                    #groundtruth[0], groundtruth[2] = groundtruth[0]/640, groundtruth[2]
                    
                    rect = plt.Rectangle((groundtruth[0], groundtruth[1]), groundtruth[2], groundtruth[3],
                                            color='blue',
                                            fill=False, linewidth=3)

                    # Add the patch to the Axes
                    ax.add_patch(rect)
            '''

            if self.dataset == "caltech":
                plt.savefig('output/image/test_caltech_9.jpg')
            else:
                plt.savefig('output/image_voc/val_voc_159_' + img_id + '.jpg')
        
        print("Visualization ends!")