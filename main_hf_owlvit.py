# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import urllib
from datetime import datetime, time
import csv
import numpy as np
import os
from numba import cuda as ncuda


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torch import cuda
import json
from timm import create_model
import torch
import torchvision
import torchvision.transforms as T
import glob
import urllib
import time
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import ViTModel,ViTFeatureExtractor, ViTForImageClassification

def get_less_used_gpu(gpus=None, debug=False):
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'

    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        cuda.empty_cache()

    if debug:
        print('After:')
        get_less_used_gpu(debug=True)

# url, filename = (
#     "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
# urllib.request.urlretrieve(url, filename)
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
prob_lim = float(sys.argv[1])
path = sys.argv[2]
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
from transformers import AutoFeatureExtractor, ConditionalDetrForObjectDetection
import torch
from PIL import Image
import requests
import pandas as pd

files = glob.glob(path)
if os.path.exists("s:/files_hf_vit.csv"):
    files_list = open("s:/files_hf_vit.csv", 'a', newline='')
    writer_files_list = csv.writer(files_list)
    df_file_list = pd.read_csv("s:/files_hf_vit.csv", header=0)
else:
    files_list = open("s:/files_hf_vit.csv", 'w', newline='')
    string = ["path", "f"]#, "label", "prob"]
    writer_files_list = csv.writer(files_list)
    writer_files_list.writerow(string)
    files_list.close()
    df_file_list = pd.read_csv("s:/files_hf_vit.csv", header=0)

    f = open("s:/labels_hf_vit.csv", 'w', newline='')
    writer = csv.writer(f)
    string = str(datetime.now()).replace(" ", "|")
    string = np.append(string, " ---start-----")
    string =["path","f","label","prob"]
    writer.writerow(string)
    f.close()
if len(files)<=len(df_file_list):
    print("All files procesed")
    exit(0)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
#device ='cuda'
#device = 'cpu'

#get_less_used_gpu([0],debug=True)
#free_memory([0],debug=True)
from transformers import OwlViTProcessor, OwlViTForObjectDetection

mn="google/owlvit-base-patch16"
mn="google/owlvit-base-patch32"
#mn="google/owlvit-large-patch14"
# large doesnt fit to 3060 gpu "google/owlvit-large-patch14"
texts = [["nude", "nude body", "nude picture","nudes"]]
processor = OwlViTProcessor.from_pretrained(mn)
model = OwlViTForObjectDetection.from_pretrained(mn )#
model.eval()
model.to(device)

f = open("s:/labels_hf_vit.csv", 'a', newline='')
writer = csv.writer(f)
categories = model.config.id2label
for file in files:
    filename = os.fsdecode(file)
    df_file = df_file_list.query('f=="'+os.path.basename(filename)+'"')
    if len(df_file)>0:
        continue

    print(f' Hug Face vit -{filename}')
    #if filename.endswith(".png") or filename.endswith(".jpg"):
    image = PIL.Image.open(filename)
    inputs = processor(text=texts, images=image, return_tensors="pt")

    inputs = inputs.to(device)
    try:
        outputs = model(**inputs)
    except Exception as e:
        #free_memory([0],debug=True)
        #ncuda.select_device(0)
        #ncuda.close()
        #ncuda.select_device(0)
        #processor = OwlViTProcessor.from_pretrained(mn)
       # model = OwlViTForObjectDetection.from_pretrained(mn)  #
        #model.eval()
       #model.to(device)
        #continue
        exit(-1)
    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    target_sizes = target_sizes.to(device)
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Print detected objects and rescaled box coordinates
    score_threshold = 0.1
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= prob_lim:
            prob= round(score.item(), 3)
            prob = "{:.2f}%".format(score * 100)
            string = filename
            string = np.append(string, os.path.basename(filename))
            string = np.append(string, text[label])
            string = np.append(string, prob)
            # f = open("s:/labels.csv", 'a', newline='')

            writer.writerow(string)
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    string = filename
    string = np.append(string, os.path.basename(filename))
    writer_files_list.writerow(string)
    #
    # inputs = feature_extractor(images=image, return_tensors="pt")
    # inputs = inputs.to(device)
    # outputs = model(**inputs)
    # logits = outputs.logits
    # model predicts one of the 21,841 ImageNet-22k classes

f.close()
    #time.sleep(5)
#plt.imshow(img)

