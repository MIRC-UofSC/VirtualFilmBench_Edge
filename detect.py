"""
This module is the most recent version of the Yolov7 Detect script for Edge Detection.

Version: 1.2.0 (No-cross-frame-detection)
Authors: Liang Z. and Deja S.
Created: ???
Edited: 21-03-2025
"""

import argparse
import itertools
import math
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import yaml
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from xml.etree import ElementTree as ET
import torchvision.transforms as T

unloader = T.ToPILImage()
import numpy as np
from operator import itemgetter
import os
from utils.torch_utils import ModelEMA, select_device, intersect_dicts
from models.yolo import Model
import torch.nn as nn
from datetime import date
from models.common import Conv

# Deja's Imports
# ------------------------------
from PIL import Image
import torchvision.models


# Text Orientation functions ==========================================================================================
def white_or_black_text(img):
    """
    This function determines if there is white text or black text present.
    """
    pixel_num = img.size
    num_white_pixels = cv2.countNonZero(img)
    white_pixel_per = (num_white_pixels / pixel_num) * 100

    if white_pixel_per < 60:
        return True
    else:
        return False


def crop_img(x_centre, y_centre, width, height, img):
    img_h, img_w, _ = img.shape

    # Convert center coordinates to top-left corner
    x1 = int(x_centre - width / 2)
    y1 = int(y_centre - height / 2)
    x2 = int(x_centre + width / 2)
    y2 = int(y_centre + height / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return img[y1:y2, x1:x2]


# ====================================================================================================================


# Weighted Voting ====================================================================================================
# Old version
# def calculate_weights(num_weights, char_weights, label, stack_values):
#     weighted_votes = {}
#
#     for i, c in enumerate(label):
#         if c in num_weights:
#             weight = num_weights[c]
#         elif c in char_weights:
#             weight = char_weights[c]
#         else:
#             continue
#
#         # Adjust weight using stack values 1s and 2s
#         num_ones = stack_values[i].count(1)
#         num_twos = stack_values[i].count(2)
#
#         # Boost weight if more 2s, reduce if more 1s
#         adjusted_weight = weight + (num_twos * 2) - (num_ones * 1)
#
#         weighted_votes[c] = weighted_votes.get(c, 0) + adjusted_weight
#
#     final_vote = max(weighted_votes, key=weighted_votes.get) if weighted_votes else None
#
#     if '6' in weighted_votes and '9' in weighted_votes:
#         # If 6 won, switch to 9
#         if final_vote == '6':
#             final_vote = '9'
#         # If 9 won, switch to 6
#         elif final_vote == '9':
#             final_vote = '6'
#
#     return final_vote

def weighted_voting_classifier(label, num_weights, char_weights, stack_values, class_threshold=0.6):
    # Initialise vote counters for each class
    class_votes = {1: 0.0, 2: 0.0}
    total_weight = 0.0

    # Calculate weighted votes for each character
    for i, c in enumerate(label):
        # Get base weight for the character
        if c in num_weights:
            base_weight = num_weights[c]
        elif c in char_weights:
            base_weight = char_weights[c]
        else:
            continue

        # Get stack information for this position
        stack_counts = {
            1: stack_values[i].count(1),
            2: stack_values[i].count(2)
        }

        # Calculate stack-adjusted weight
        stack_modifier = (stack_counts[2] * 2.0) - (stack_counts[1] * 1.0)
        # Ensure weight doesn't go below 0.1
        adjusted_weight = max(0.1, base_weight + stack_modifier)

        # Determine which class gets the vote based on stack majority
        if stack_counts[2] > stack_counts[1]:
            class_votes[2] += adjusted_weight
        elif stack_counts[1] > stack_counts[2]:
            class_votes[1] += adjusted_weight
        else:
            # If tied, split the vote
            class_votes[1] += adjusted_weight * 0.5
            class_votes[2] += adjusted_weight * 0.5

        total_weight += adjusted_weight

    # Calculate confidence scores
    if total_weight > 0:
        confidence_scores = {
            cls: votes / total_weight
            for cls, votes in class_votes.items()
        }

        # Determine if label should be reversed
        should_reverse = confidence_scores[2] > class_threshold

        # Special handling for 6/9 case
        if '6' in label or '9' in label:
            if should_reverse:
                # Additional threshold for 6/9 cases to avoid unnecessary flips
                should_reverse = confidence_scores[2] > 0.7
    else:
        should_reverse = False
        confidence_scores = {1: 0.5, 2: 0.5}

    return should_reverse, max(confidence_scores.values())


# ====================================================================================================================

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Deja's Directories ===============================================================================================
    # save_img_path = Path(save_dir / 'backwards_imgs')
    exp_img_path = Path(save_dir / 'exp_img')

    # Create directory for exp images
    # if not os.path.exists(exp_img_path):
    #     os.mkdir(exp_img_path)
    # ==================================================================================================================

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Change: (03-07-2024) Change from half = device.type != 'cpu', half was not initialising to True for 'cpu'.
    # half = device != 'cpu' or device != ''  # half precision only supported on CUDA
    half = False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    #if trace:
    #    model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Orientation / Rotation Model Configurations ======================================================================
    orientation_model = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")

    # Freeze pre-trained layers
    for param in orientation_model.parameters():
        param.requires_grad = False

    # Modify last layer
    orientation_model.fc = torch.nn.Linear(orientation_model.fc.in_features, 2)
    orientation_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    orientation_model.maxpool = torch.nn.Identity()

    # Load pre-trained weights
    if device == torch.device('cpu'):
        orientation_model.load_state_dict(
            torch.load("runs/train/yolo7-14N/orientation_weights/best_model_23-10-2024.pth",
                       map_location=torch.device('cpu')))
    else:
        orientation_model.load_state_dict(
            torch.load("runs/train/yolo7-14N/orientation_weights/best_model_23-10-2024.pth"))
    orientation_model.to(device)
    orientation_model.eval()

    # Image transformations
    transform = T.Compose([
        T.Resize((64, 64)),
        T.Pad((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Orientation Class Labels
    o_class_labels = {"0": 1, "1": 2}
    # ==================================================================================================================

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    current_date = date.today()
    da = str(current_date).strip().split('-')
    td = ''
    td = da[-2] + da[-1]
    t0 = time.time()
    cnt = 0

    # we make root element of xml
    # OLD - 62 - For experiments yolov7-12 >=
    # namestrain = ['AGFA', 'INC.', 'J', 'a', 'ansco', 'b', 'be', 'belgium', 'c', 'circle', 'color', 'd', 'dupont', 'e',
    #               'eastman', 'eight', 'exchange', 'f', 'film', 'five', 'four', 'fuji', 'g', 'gevaert', 'h', 'ilford',
    #               'k', 'kodak', 'l', 'm', 'n', 'nine', 'nitrate', 'not', 'number', 'o', 'of', 'one', 'p',
    #               'panchromatic', 'pathe', 'plus', 'property', 'r', 's', 's-afety', 'safety', 'seven', 'six', 'sold',
    #               'square', 't', 'three', 'to', 'triangle', 'two', 'u', 'v', 'x', 'y', 'z', 'zero', 'B', 'D', 'H',
    #               'KODAK', 'N', 'agfa', 'ferrania', 'news', 'pancro', 'stock']

    # namestrain = ['KODAK', 'B', 'D', 'H', 'N', 'agfa', 'ferrania','news','pancro','stock', 'AGFA', 'INC.', 'J',
    # 'a', 'ansco', 'b', 'be', 'belgium', 'c', 'circle', 'color', 'd', 'dupont', 'e', 'eastman', 'eight', 'exchange',
    # 'f', 'film', 'five', 'four', 'fuji', 'g', 'gevaert', 'h', 'ilford', 'k', 'kodak', 'l', 'm', 'n', 'nine',
    # 'nitrate', 'not', 'number', 'o', 'of', 'one', 'p', 'panchromatic', 'pathe', 'plus', 'property', 'r', 's',
    # 's-afety', 'safety', 'seven', 'six', 'sold', 'square', 't', 'three', 'to', 'triangle', 'two', 'u', 'v', 'x',
    # 'y', 'z', 'zero']

    # Old: Classes 72 - For experiments yolov7-13 and yolo7-14
    # namestrain = ['A', 'AGFA', 'ANSCO', 'B', 'BE', 'BELGIUM', 'C', 'CINEMA', 'COLOR', 'D', 'DUPONT', 'E', 'EASTMAN',
    #               'EXCHANGE', 'F', 'FERRANIA', 'FILM', 'FRANCE', 'FUJI', 'G', 'GEVAERT', 'GOERZ', 'H', 'I', 'ILFORD',
    #               'INC.', 'J', 'K', 'KODAK', 'L', 'M', 'N', 'NEWS', 'NITRATE', 'NOT', 'O', 'OF', 'P', 'PANCHRO',
    #               'PANCHROMATIC', 'PATHE', 'PROPERTY', 'R', 'S', 'S-AFETY', 'SAFE-TY', 'SAFETY', 'SOLD', 'STOCK', 'T',
    #               'TENAX', 'TO', 'U', 'V', 'X', 'Y', 'Z', 'circle', 'eight', 'five', 'four', 'nine', 'one', 'plus',
    #               'seven', 'six', 'square', 'three', 'trademark', 'triangle', 'two', 'zero']

    # New: Classes 79 - For experiments yolov7-14N <=
    namestrain = ['A', 'AGFA', 'ANSCO', 'B', 'BE', 'BELGIUM', 'BIX', 'C', 'CINEMA', 'COLOR', 'D', 'DUP-', 'DUPONT', 'E',
                  'EASTMAN', 'EXCHANGE', 'F', 'FERRANIA', 'FILM', 'FRANCE', 'FUJI', 'G', 'GEVAERT', 'GOERZ', 'H', 'I',
                  'ILFORD', 'INC.', 'J', 'K', 'KODAK', 'L', 'M', 'N', 'NEWS', 'NITRATE', 'NOT', 'O', 'OF', 'P',
                  'PANCHROMATIC', 'PANCRO', 'PATHE', 'PROPERTY', 'R', 'S', 'S-AFETY', 'SAF-ETY', 'SAFE-TY', 'SAFETY',
                  'SOLD', 'SPEED', 'STOCK', 'SUPERPAN', 'T', 'TENAX', 'TO', 'U', 'ULTRA', 'V', 'VINCENNES', 'X', 'Y',
                  'Z', 'circle', 'eight', 'explosion', 'five', 'four', 'nine', 'one', 'plus', 'seven', 'six', 'square',
                  'three', 'triangle', 'two', 'zero']

    # The number of labels
    num_classes = 79

    shapes = ['circle', 'plus', 'square', 'triangle']

    # Original Detection XML Document
    VB_info = ET.Element("Virtual_Bench")

    # Outlier XML Document
    GB_info = ET.Element("Virtual_Bench")

    # ET.SubElement(VB_info, "CreatorID").text = 'edge_inf_{}_2024.pt'.format(td)
    # ET.SubElement(VB_info, "input_id").text = os.path.basename(source)
    # ET.SubElement(VB_info, "user_id").text = 'd00001'
    # ET.SubElement(VB_info, "vb_compute_id").text = 'test_0001'
    # ET.SubElement(VB_info, "input_title").text = os.path.basename(source)
    # ET.SubElement(VB_info, "CreatorContext").text = 'Univ. of South Carolina'
    # ET.SubElement(VB_info, "Class_Dictionary").text = ' '.join(namestrain)
    # nu = ' '.join(namestrain)
    # ET.SubElement(VB_info, "vb_input_id").text = '002516d'

    for path, img, im0s, vid_cap in dataset:

        # Copying the image for backwards detection
        # Path -> Image path
        # img -> Padded and reformatted image
        # img0s -> The original image read from the video frame

        # Mask the middle of the video to only perform detection on the edges
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        nb, _, height, width = img.shape
        masks = np.ones((height, width))

        if opt.mask_size == "normal":
            masks[int(height * 0.25):int(height * 0.85), 0:width] = 0
        elif opt.mask_size == "wide":
            masks[0:height, int(width * 0.1):int(width * 0.90)] = 0
        else:
            print("ERROR: Mask size is invalid!")
            exit()

        # masks[0:height, int(width * 0.2):int(width * 0.85)] = 0
        # masks[0:height, int(width * 0.1):int(width * 0.85)] = 0
        # masks[int(height * 0.25):int(height * 0.85), 0:width] = 0
        # masks[0:height, int(width * 0.1):int(width * 0.90)] = 0
        m = torch.tensor(masks, dtype=torch.float32)
        m = m.unsqueeze(0)
        m = m.unsqueeze(0)
        m = m.to(device)
        imgs2 = img * m
        # imgs2 = imgs2.half()

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(imgs2, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        (h, w) = im0s.shape[:2]
        # annotation = create_tree(source, path, h, w)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)  #-1
                # if frame in dataset:
                dataset.frame += 1

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {namestrain[int(c)]}{'s' * (n > 1)}, "  # add to string '1 film, 1 four, 1 three'

                save_txt = True
                # Write results
                blist = []

                # Garbage Collection
                rejected_detections = []

                # Left and Right concatenation candidates
                l_potential_blist = {}
                r_potential_blist = {}
                # l_blist = []
                # r_blist = []

                det2 = []

                # Non-cat, left, and right concat label
                nnn = ''
                l_nnn = ''
                r_nnn = ''

                # Non-cat, Left and Right Labels
                lab = []
                l_lab = []
                r_lab = []

                k = l_k = r_k = num_classes

                nums = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
                nu = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

                # Scott's suggestion
                nu_weights = [1, 1, 2, 4, 4, 2, 4, 9, 1, 4]

                # For Concatenation experiment
                characters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                              "S", "T", "U", "V", "W", "X", "Y", "Z"]

                # Scott's suggestion
                # full symmetry (flip and rotate): 1
                # half symmetry: 4
                # zero symmetry: 9
                # note the weights for Z,2,5,6,9 are lower than their symmetries because Z, 2, and 5 are symmetric to
                # each other as are 6 and 9.
                characters_weights = [4, 4, 4, 4, 4, 9, 9, 1, 1, 9, 4, 9, 4, 4, 1, 9, 9, 9, 4, 4, 4, 4, 4, 1, 4, 2]

                # Mapping numbers and characters to their respective weights
                num_weights_dict = dict(zip(nu, nu_weights))
                char_weights_dict = dict(zip(characters, characters_weights))

                # det = sorted(det, key=itemgetter(0))

                # For Normal
                det_n = []
                for d in det:
                    if (d[0] < width / 2 and d[2] > width / 2) or (d[0] > width / 2 and d[2] < width / 2):
                        continue
                    else:
                        det_n.append(d)

                det_n = sorted(det_n, key=lambda x: (x[1], x[0]))
                det = torch.stack(det_n)

                flag = False

                # Non-cat, Left and Right stack
                stk = []
                l_stk = []
                r_stk = []

                # Concatenation width threshold
                width_thr = 25

                # Left and Right Orientation Stack
                left_orientation_stack = []
                right_orientation_stack = []

                # Liang's way to concatenate integers =================================================================
                # for j in range(len(det)):
                #     line = torch.zeros((6,))
                #     box = det[j, :4]
                #     conf = det[j, -2]
                #
                #     clsnum = int(det[j, :][-1])
                #     cls = namestrain[int(det[j, :][-1])]  # 'film'
                #     # cls = names[int(det[j, :][-1])]  # 'film'
                #
                #     if cls not in nums:
                #         line[:4] = box
                #         line[-2] = conf
                #         line[-1] = clsnum
                #         det2.append(line)
                #         stk.append(cls)
                #         nnn = ''
                #     else:
                #         if stk and stk[-1] in nums and ((blist[-1][0] > width / 2 and box[0] > width / 2) or (
                #                 blist[-1][0] < width / 2 and box[0] < width / 2)):
                #             b = blist.pop()
                #             b1 = b[:4]
                #             b2 = box
                #             bb = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
                #             linec = torch.zeros((6,))
                #             linec[0] = bb[0].item()
                #             linec[1] = bb[1].item()
                #             linec[2] = bb[2].item()
                #             linec[3] = bb[3].item()
                #             linec[-2] = conf
                #             #lab.append(ll)
                #             linec[-1] = k - 1
                #             #k+=1
                #             nnn += nu[nums.index(cls)]
                #             lab.pop()
                #             lab.append(nnn)
                #
                #             blist.append(linec)
                #             stk.append('zero')
                #             #blist=[]
                #         else:
                #             linec = torch.zeros((6,))
                #             linec[0] = box[0].item()
                #             linec[1] = box[1].item()
                #             linec[2] = box[2].item()
                #             linec[3] = box[3].item()
                #             linec[-2] = conf
                #             idd = nums.index(cls)
                #             nnn = nu[idd]
                #             linec[-1] = k
                #             k += 1
                #             lab.append(nnn)
                #             blist.append(linec)
                #             stk.append(cls)
                #             # nnn = ''
                #             # lab=[]

                # New Left + Right Edge Concatenation of characters and integers ======================================
                for j in range(len(det)):
                    line = torch.zeros((6,))
                    box = det[j, :4]
                    conf = det[j, -2]

                    clsnum = int(det[j, :][-1])
                    cls = namestrain[int(det[j, :][-1])]  # 'film'

                    # Check Bounding Box Area for outliers
                    bb_area = abs(np.ceil(box[2].item()) - np.ceil(box[0].item())) * abs(
                        np.ceil(box[1].item()) - np.ceil(box[3].item()))
                    if bb_area < 2500.0:

                        temp_box = torch.zeros((4,))
                        temp_box[0] = box[0].item()
                        temp_box[1] = box[1].item()
                        temp_box[2] = box[2].item()
                        temp_box[3] = box[3].item()

                        rejected_detections.append({
                            "bbox": temp_box,
                            "frame": frame,
                            "class_name": cls,
                            "class_num": clsnum,
                            "conf": conf.item()
                        })
                        continue

                    # For labels that do not need to be concatenated
                    if cls not in nums and cls not in characters:
                        line[:4] = box
                        line[-2] = conf
                        line[-1] = clsnum
                        det2.append(line)
                        stk.append(cls)
                        nnn = ''

                        # Check area of shape
                        # if cls in shapes:
                        #     area = abs(np.ceil(box[2].item()) - np.ceil(box[0].item())) * abs(np.ceil(box[1].item()) - np.ceil(box[3].item()))
                        #     print(f"Shape Area : {cls} : {area}")
                        # exit()

                    else:
                        # Detect the orientation of the characters =====================================================
                        # Preparing image
                        boundB = [np.ceil(box[0].item()), np.ceil(box[1].item()), np.ceil(box[2].item()),
                                  np.ceil(box[3].item())]
                        input_crop = crop_img(boundB[0], boundB[1], boundB[3], boundB[2], im0)
                        input_crop = cv2.cvtColor(input_crop, cv2.COLOR_BGR2RGB)
                        input_crop = Image.fromarray(input_crop)
                        ori_img = transform(input_crop)
                        ori_img = ori_img.unsqueeze(0)
                        ori_img = ori_img.to(device)

                        # Predict the orientation class
                        with torch.no_grad():
                            outputs = orientation_model(ori_img)
                        # New
                        probs = torch.sigmoid(outputs)
                        preds = (probs[0] >= 0.60).int()

                        # _, preds = torch.max(outputs, 1)
                        pred_ori_label = o_class_labels[f"{preds[0]}"]
                        # ==============================================================================================

                        if l_stk and (l_stk[-1] in nums or l_stk[-1] in characters) and (box[0] < width / 2):
                            b2 = box
                            curr_bbox_width = int(b2[2] - b2[0])
                            found = True

                            # Iterates through previous bbox widths to see if current bbox is similar; if so, they are
                            # concatenated together.
                            for key in l_potential_blist:
                                found = True
                                o_l_stack = l_potential_blist[key]["o_left_stack"]
                                prev_box = b = l_potential_blist[key]["c_box"].pop()
                                b1 = b[:4]

                                prev_bbox_width = int(b1[2] - b1[0])
                                diff = abs(curr_bbox_width - prev_bbox_width)

                                # If the bounding box width is similar to the previous bounding box by a given threshold
                                # the box will be added to the previous box and update the concatenation
                                # candidate object
                                if diff <= width_thr:
                                    l_nnn = l_potential_blist[key]["c_nnn"]
                                    bb = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
                                    linec = torch.zeros((6,))
                                    linec[0] = bb[0].item()
                                    linec[1] = bb[1].item()
                                    linec[2] = bb[2].item()
                                    linec[3] = bb[3].item()
                                    linec[-2] = conf
                                    linec[-1] = l_k - 1

                                    if cls in nums:
                                        l_nnn += nu[nums.index(cls)]
                                    else:
                                        l_nnn += characters[characters.index(cls)]

                                    # Create a new list with existing orientations plus new one
                                    new_o_l_stack = o_l_stack.copy() if o_l_stack else []
                                    new_o_l_stack.append(pred_ori_label)

                                    l_potential_blist[key]["l_lab"].pop()
                                    l_potential_blist[key]["l_lab"].append(l_nnn)
                                    l_potential_blist[key]["c_box"].append(linec)
                                    l_potential_blist[key]["c_nnn"] = l_nnn
                                    l_potential_blist[key]["o_left_stack"] = new_o_l_stack
                                    break
                                else:
                                    l_potential_blist[key]["c_box"].append(prev_box)
                                    l_potential_blist[key]["o_left_stack"].append(o_l_stack)
                                    found = False
                                    continue

                            # If no similar box is found, create a new concatenation candidate object.
                            if not found:
                                linec = torch.zeros((6,))
                                linec[0] = box[0].item()
                                linec[1] = box[1].item()
                                linec[2] = box[2].item()
                                linec[3] = box[3].item()
                                linec[-2] = conf
                                linec[-1] = l_k - 1
                                if cls in nums:
                                    idd = nums.index(cls)
                                    l_nnn = nu[idd]
                                else:
                                    idd = characters.index(cls)
                                    l_nnn = characters[idd]
                                l_stk.append(cls)
                                l_potential_blist[curr_bbox_width] = {"c_box": [], "l_lab": [], "c_nnn": l_nnn,
                                                                      "o_left_stack": []}
                                l_potential_blist[curr_bbox_width]["c_box"].append(linec)
                                l_potential_blist[curr_bbox_width]["l_lab"].append(l_nnn)
                                l_potential_blist[curr_bbox_width]["o_left_stack"].append(pred_ori_label)

                        # For right edge of video
                        elif r_stk and (r_stk[-1] in nums or r_stk[-1] in characters) and (box[0] > width / 2):
                            b2 = box
                            curr_bbox_width = int(b2[2] - b2[0])
                            found = True

                            # Iterates through previous bbox widths to see if current bbox is similar; if so, they are
                            # concatenated together.
                            for key in r_potential_blist:
                                found = True
                                o_r_stack = r_potential_blist[key]["o_right_stack"]
                                prev_box = b = r_potential_blist[key]["c_box"].pop()
                                b1 = b[:4]

                                prev_bbox_width = int(b1[2] - b1[0])
                                diff = abs(curr_bbox_width - prev_bbox_width)

                                # If the bounding box width is similar to the previous bounding box by a given threshold
                                # the box will be added to the previous box and update the concatenation
                                # candidate object
                                if diff <= width_thr:
                                    r_nnn = r_potential_blist[key]["c_nnn"]
                                    bb = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
                                    linec = torch.zeros((6,))
                                    linec[0] = bb[0].item()
                                    linec[1] = bb[1].item()
                                    linec[2] = bb[2].item()
                                    linec[3] = bb[3].item()
                                    linec[-2] = conf
                                    linec[-1] = r_k - 1

                                    if cls in nums:
                                        r_nnn += nu[nums.index(cls)]
                                    else:
                                        r_nnn += characters[characters.index(cls)]

                                    # Create a new list with existing orientations plus new one
                                    new_o_r_stack = o_r_stack.copy() if o_r_stack else []
                                    new_o_r_stack.append(pred_ori_label)

                                    r_potential_blist[key]["r_lab"].pop()
                                    r_potential_blist[key]["r_lab"].append(r_nnn)
                                    r_potential_blist[key]["c_box"].append(linec)
                                    r_potential_blist[key]["c_nnn"] = r_nnn
                                    # r_potential_blist[key]["o_right_stack"].append(o_r_stack)
                                    r_potential_blist[key]["o_right_stack"] = new_o_r_stack
                                    # r_stk.append('zero')
                                    break
                                else:
                                    r_potential_blist[key]["c_box"].append(prev_box)
                                    r_potential_blist[key]["o_right_stack"].append(pred_ori_label)
                                    found = False
                                    continue

                            # If no similar box is found, create a new concatenation candidate object.
                            if not found:
                                linec = torch.zeros((6,))
                                linec[0] = box[0].item()
                                linec[1] = box[1].item()
                                linec[2] = box[2].item()
                                linec[3] = box[3].item()
                                linec[-2] = conf
                                linec[-1] = r_k - 1
                                if cls in nums:
                                    idd = nums.index(cls)
                                    r_nnn = nu[idd]
                                else:
                                    idd = characters.index(cls)
                                    r_nnn = characters[idd]
                                r_stk.append(cls)
                                r_potential_blist[curr_bbox_width] = {"c_box": [], "r_lab": [], "c_nnn": r_nnn,
                                                                      "o_right_stack": []}
                                r_potential_blist[curr_bbox_width]["c_box"].append(linec)
                                r_potential_blist[curr_bbox_width]["r_lab"].append(r_nnn)
                                r_potential_blist[curr_bbox_width]["o_right_stack"].append(pred_ori_label)

                        else:
                            linec = torch.zeros((6,))
                            linec[0] = box[0].item()
                            linec[1] = box[1].item()
                            linec[2] = box[2].item()
                            linec[3] = box[3].item()
                            linec[-2] = conf

                            bb_width = int(torch.tensor(linec[2] - linec[0]).numpy().item())

                            # For the left side
                            if box[0] < width / 2:

                                if cls in nums:
                                    idd = nums.index(cls)
                                    l_nnn = nu[idd]
                                else:
                                    idd = characters.index(cls)
                                    l_nnn = characters[idd]

                                linec[-1] = l_k
                                l_k += 1
                                l_stk.append(cls)
                                l_potential_blist[bb_width] = {"c_box": [], "l_lab": [], "c_nnn": l_nnn,
                                                               "o_left_stack": []}
                                l_potential_blist[bb_width]["c_box"].append(linec)
                                l_potential_blist[bb_width]["l_lab"].append(l_nnn)
                                l_potential_blist[bb_width]["o_left_stack"].append(pred_ori_label)

                            # For the right side
                            elif box[0] > width / 2:

                                if cls in nums:
                                    idd = nums.index(cls)
                                    r_nnn = nu[idd]
                                else:
                                    idd = characters.index(cls)
                                    r_nnn = characters[idd]

                                linec[-1] = r_k
                                r_k += 1
                                r_stk.append(cls)
                                r_potential_blist[bb_width] = {"c_box": [], "r_lab": [], "c_nnn": r_nnn,
                                                               "o_right_stack": []}
                                r_potential_blist[bb_width]["c_box"].append(linec)
                                r_potential_blist[bb_width]["r_lab"].append(r_nnn)
                                r_potential_blist[bb_width]["o_right_stack"].append(pred_ori_label)
                # ======================================================================================================
                # plot_one_box(bb, mosaic, label=ll, color=color, line_thickness=tl)

                # Processing the concatenations
                # This is where we can determine which result we should use: Normal or Flipped
                # Left
                for key in l_potential_blist:
                    while len(l_potential_blist[key]["c_box"]):
                        bl = l_potential_blist[key]["c_box"].pop()
                        linec = bl
                        det2.append(linec)

                        temp_l_label = l_potential_blist[key]["l_lab"][0]

                        filter_1 = [x for x in l_potential_blist[key]["o_left_stack"] if x in [1]]
                        filter_2 = [x for x in l_potential_blist[key]["o_left_stack"] if x in [2]]

                        filter_1_sum = len(filter_1)
                        filter_2_sum = len(filter_2)

                        # Filters through classification labels to determine label direction
                        if filter_2_sum > filter_1_sum:
                            temp_l_label = temp_l_label[::-1]

                        # Checks if the last character is a letter
                        if len(temp_l_label) >= 3:
                            if temp_l_label[-1].upper() in characters:
                                temp_l_label = temp_l_label[::-1]

                        l_lab.append(temp_l_label)

                # Right
                for key in r_potential_blist:
                    while len(r_potential_blist[key]["c_box"]):
                        bl = r_potential_blist[key]["c_box"].pop()
                        linec = bl
                        det2.append(linec)

                        temp_r_label = r_potential_blist[key]["r_lab"][0]

                        filter_1 = [x for x in r_potential_blist[key]["o_right_stack"] if x in [1]]
                        filter_2 = [x for x in r_potential_blist[key]["o_right_stack"] if x in [2]]

                        filter_1_sum = len(filter_1)
                        filter_2_sum = len(filter_2)

                        # Filters through classification labels to determine label direction
                        if filter_2_sum > filter_1_sum:
                            temp_r_label = temp_r_label[::-1]

                        # Checks if the last character is a letter
                        if len(temp_r_label) >= 3:
                            if temp_r_label[-1].upper() in characters:
                                temp_r_label = temp_r_label[::-1]

                        r_lab.append(temp_r_label)

                # Non-concatenation labels
                while len(blist) != 0:
                    bl = blist.pop()
                    linec = bl
                    det2.append(linec)

                for *xyxy, conf, cls in reversed(det2):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    box = torch.tensor(xyxy).numpy()
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                    if cls < num_classes:
                        label = namestrain[int(cls)]
                        num = int(cls)
                    else:
                        # For left side
                        if x1 < width / 2:
                            label = l_lab[int(cls - num_classes)]
                        # For right side
                        elif x1 > width / 2:
                            label = r_lab[int(cls - num_classes)]
                        num = int(cls - num_classes)

                    # Save image for experiment ========================================================================
                    # For Long videos
                    # if (frame % 100) == 0:
                    #     normal_img = im0
                    #     cv2.rectangle(normal_img, (x1, y1), (x2, y2), color=(90, 200, 0), thickness=2)
                    #     cv2.putText(normal_img, f"{label}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 200, 0), 2,
                    #                 cv2.LINE_AA)
                    #
                    #     normal_save_name = f"{frame}_{label}_normal.png"
                    #     cv2.imwrite(os.path.join(exp_img_path, normal_save_name), normal_img)

                    # For Short videos
                    # normal_img = im0
                    # cv2.rectangle(normal_img, (x1, y1), (x2, y2), color=(90, 200, 0), thickness=2)
                    # cv2.putText(normal_img, f"{label}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 200, 0), 2,
                    #             cv2.LINE_AA)
                    #
                    # normal_save_name = f"{frame}_{label}_normal.png"
                    # cv2.imwrite(os.path.join(exp_img_path, normal_save_name), normal_img)
                    # =================================================================================================

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    event = ET.SubElement(VB_info, "event")
                    ET.SubElement(VB_info, "CreatorID").text = 'edge_inf_{}_2024.pt'.format(td)
                    ET.SubElement(VB_info, "CreatorContext").text = 'Univ. of South Carolina'
                    # ET.SubElement(event, "event_creator").text = 'machine'
                    # ET.SubElement(event, "event_frame_id").text = str(cnt) #'machine id'
                    # ET.SubElement(event, "film_change").text = 'None'
                    # ET.SubElement(event, "sqeuence_change").text = 'None'
                    # ET.SubElement(event, "damage").text = 'None'
                    ET.SubElement(event, "location_absolute_in").text = str(frame)
                    ET.SubElement(event, "location_absolute_out").text = str(frame)
                    bbox = ET.SubElement(event, "location_pixels")
                    bbox.text = ','.join(str(x) for x in xywh)
                    # bbox.text = ','.join(str(x) for x in box)
                    # xmlcls = ET.SubElement(event, "event_confidence")
                    # xmlcls.text = str(conf.item())
                    ET.SubElement(event, "event_confidence").text = str(conf.item())
                    # event_join = ET.SubElement(event, "join")
                    # event_join.text = 'splices'
                    if cls < num_classes:
                        # Liang's Original Code
                        # if cls == 9 or cls == 41 or cls == 54:
                        if namestrain[int(cls)] in shapes:
                            ET.SubElement(event, "EdgeMark").text = 'Symbol'
                            ET.SubElement(event, "EdgeMarkSymbol").text = str(label)
                        else:
                            ET.SubElement(event, "EdgeMark").text = 'Word'
                            ET.SubElement(event, "EdgeMarkWord").text = str(label)
                    else:
                        ET.SubElement(event, "EdgeMark").text = 'String'
                        ET.SubElement(event, "EdgeMarkString").text = str(label)
                    # ET.SubElement(event, "classification").text = str(label) #str(cls.item())
                    # ET.SubElement(event_join, "coupling").text = 'None'

                    if save_img or view_img:  # Add bbox to image
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[num], line_thickness=3)

                # Garbage XML Events ===================================================================================
                for obj in rejected_detections:
                    gb_xywh = (xyxy2xywh(torch.tensor(obj["bbox"]).view(1, 4)) / gn).view(-1).tolist()
                    gb_event = ET.SubElement(GB_info, "event")

                    ET.SubElement(GB_info, "CreatorID").text = 'edge_inf_{}_2024.pt'.format(td)
                    ET.SubElement(GB_info, "CreatorContext").text = "Univ. of South Carolina"
                    ET.SubElement(gb_event, "location_absolute_in").text = str(obj["frame"])
                    ET.SubElement(gb_event, "location_absolute_out").text = str(obj["frame"])

                    gb_bbox = ET.SubElement(gb_event, "location_pixels")
                    gb_bbox.text = ",".join(str(x) for x in gb_xywh)

                    ET.SubElement(gb_event, "event_confidence").text = str(obj["conf"])

                    if namestrain[int(obj["class_num"])] in shapes:
                        ET.SubElement(gb_event, "EdgeMark").text = "Symbol"
                        ET.SubElement(gb_event, "EdgeMarkSymbol").text = str(namestrain[int(obj["class_num"])])
                    else:
                        ET.SubElement(gb_event, "EdgeMark").text = "Word"
                        ET.SubElement(gb_event, "EdgeMarkWord").text = str(namestrain[int(obj["class_num"])])

                # ======================================================================================================

            # Print time (inference + NMS)

            print(f'Normal: {s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # cv2.imwrite('/home/lz/Downloads/yolov7/t/im{}.jpg'.format(cnt), im0)
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        cnt += 1

    # save splice info xml
    tree = ET.ElementTree(VB_info)

    # write the tree into an XML file
    name = str(os.path.basename(source))
    tree.write(str(save_dir / f'{name}.xml'), encoding='utf-8', xml_declaration=True)

    # Save outlier XML
    gb_tree = ET.ElementTree(GB_info)
    gb_tree.write(str(save_dir / f"{name}_rejected.xml"), encoding='utf-8', xml_declaration=True)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--mask_size', default="wide", type=str, help="Video frame mask: 'normal' or 'wide'")
    parser.add_argument('--debug', action='store_true', help="Produce debug XML for character concatenation.")
    opt = parser.parse_args()
    # print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    # print(opt['img_size'])

    # Deja's Weights
    # opt.weights = "runs/train/yolo7-13_train/weights/best.pt"

    # Liang's Weights
    # opt.weights = "Liang_runs/train/yolov7-12/weights/best.pt"

    opt.conf = 0.65
    opt.img_size = 1280

    # (Liang) Sources
    # opt.source= "/Users/gregwilsbacher/Documents/deep_learning_workspace/video_samples/vb_samp146.mov"
    # opt.source= "/work/zhouj/projects/MIRC/AEO-light/video/vb_samp109_p1_crush.mov"

    # (Deja S.) My sources
    # opt.source = "edge.mov"
    # opt.source = "../Dataset/vb_samp109_p1_crush.mov"
    # opt.source = "../Dataset/vb_samp242-h264.mov"
    # opt.source = "../Dataset/vb_demo5.mov"
    # opt.source = "../Dataset/vb_samp279.mov"
    # opt.source = "../Dataset/vb_samp126-trim.mov"
    # opt.source = "../Dataset/vb_samp127.mov"
    # opt.source = "../Dataset/vb_samp125.mov"
    # opt.source = "../Dataset/vb_samp296_midPart.mov"
    # opt.source = "../Dataset/f2024118B_partial.mov"
    # opt.source = "../Dataset/f20120329A_partial.mov"
    # opt.source = "../Dataset/f20250121A_Pos.mov"
    # opt.source = "../Dataset/f20250121A_Neg.mov"
    # opt.source = "../Dataset/vb_samp39_rescan.mov"

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
