import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
from util.visualization import output_visualizer
from main import get_args_parser

def main(args):

    device = torch.device(args.device)
    model, _, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Start visualization")
    dir = 'data/voc/VOCdevkit/val2017'

    numbers_of_img_to_plot = 5

    img_ids = []
    img_paths = []

    img_file_names = os.listdir(dir)
    random.seed(43)
    random_list = random.sample(range(0, len(img_file_names)), numbers_of_img_to_plot)

    for i in range(len(random_list)):

        filename = img_file_names[random_list[i]]
        img_id, _ = filename.split('.')

        img_ids.append(img_id)

        img_path = os.path.join(dir, filename)

        img_paths.append(img_path)

    ann_path = 'data/coco/annotations/instances_val2017.json'
    data = args.dataset_file

    visual = output_visualizer(model, postprocessors, img_paths, ann_path, img_ids, data, threshold=0.3)

    visual.standard_visualization()




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)