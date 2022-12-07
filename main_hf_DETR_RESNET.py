# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import urllib
from datetime import datetime, time
import csv
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"
import matplotlib.pyplot as plt
import PIL
from PIL import Image
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

from PIL import Image
import requests

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
device ='cuda'
#device = 'cpu'


from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
model.eval()
model.to(device)
files = glob.glob(path)
#files = glob.glob("S:/good_imgs/1/*.jpg")#s:/content/*.jpg

#files.sort(key=os.path.getmtime,reverse=True)

f = open("s:/labels_hf_detr_resnet.csv", 'w', newline='')
writer = csv.writer(f)

string = str(datetime.now()).replace(" ", "|")
string = np.append(string, " ---start-----")
string =["path","f","label","prob"]
writer.writerow(string)
categories = model.config.id2label
for file in files:
    filename = os.fsdecode(file)
    print(f' Hug Face -{filename}')
    #if filename.endswith(".png") or filename.endswith(".jpg"):
    image = PIL.Image.open(filename)

    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9
        if score > prob_lim:
            labels= model.config.id2label[label.item()]
            prob = "{:.2f}%".format(score*100)
            string = filename
            string = np.append(string, os.path.basename(filename))
            string = np.append(string, labels)
            string = np.append(string, prob)
            # f = open("s:/labels.csv", 'a', newline='')
            writer = csv.writer(f)
            writer.writerow(string)
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

f.close()
    #time.sleep(5)
#plt.imshow(img)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
