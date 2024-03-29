# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
from datetime import datetime, time
import csv
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import json
from timm import create_model
import torch
import torchvision
import torchvision.transforms as T
import glob
import time
def check_image(filename):
    matches = ["pen", "ink"]
    top5 = get_description(filename)
    # blablabla
    top5_prob = top5.values[0]
    top5_indices = top5.indices[0]
    string = filename
    for i in range(5):
        labels = imagenet_labels[str(int(top5_indices[i]))]
        pr=float(top5_prob[i]) * 100

        prob = "{:.2f}%".format(float(top5_prob[i]) * 100)
        if pr<60 :
            return 0
        else:
            if any(x in labels for x in matches):
                return 0
            else:
                return pr
        print(labels, prob)
        string = np.append(string, labels)
        string = np.append(string, prob)

    f = open("s:/content/labels.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(string)
    f.close()
def create_convnext_model():
    ordinal='0'
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    #device = torch.device('cuda:{}'.format(ordinal))


    #search nude better of all convnext
    model_name = "convnext_large_in22k"

    #model_name = "convnext_base_in22k"
    #model_name = "convnext_xlarge_in22k"
    # model_name = "vit_base_patch16_224"
    #model_name = "vit_huge_patch14_224_in21k"
    # https://rwightman.github.io/pytorch-image-models/models/noisy-student/
    #model_name = "tf_efficientnet_b0_ns"
    #model_name = "convnext_xlarge_384_in22ft1k"
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device for convnext = ", device)
    #device='cuda'
    # create a ConvNeXt model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
    model = create_model(model_name, pretrained=True).to(device)


    # Define transforms for test
    from timm.data.constants import \
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
    NORMALIZE_STD = IMAGENET_DEFAULT_STD
    SIZE = 256

    # Here we resize smaller edge to 256, no center cropping
    transforms = [
                  T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
                  T.ToTensor(),
                  T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
                  ]

    transforms = T.Compose(transforms)
    return model,transforms,device
def get_description(fname):
    img = PIL.Image.open(fname)
    img_tensor = transforms(img).unsqueeze(0).to(convnext_device)

    # inference
    output = torch.softmax(convnext_model(img_tensor), dim=1)
    top5 = torch.topk(output, k=5)

    return top5


prob_lim = float(sys.argv[1])
path = sys.argv[2]

imagenet_labels = json.load(open('label_to_words.json'))
convnext_model, transforms, convnext_device = create_convnext_model()



files = glob.glob(path)
#files = glob.glob("S:/good_imgs/1/*.jpg")#s:/content/*.jpg

#files.sort(key=os.path.getmtime,reverse=True)

f = open("s:/labels.csv", 'w', newline='')
writer = csv.writer(f)

string = str(datetime.now()).replace(" ", "|")
string = np.append(string, " ---start-----")
string =["path","f","label","prob"]
writer.writerow(string)

for file in files:
    filename = os.fsdecode(file)
    print(f'conv {prob_lim}-{filename}')
    if filename.endswith(".png") or filename.endswith(".jpg"):
        top5 = get_description(filename)
            #blablabla
    rc= check_image(filename)
    if rc>60:
        pass
    top5_prob = top5.values[0]
    top5_indices = top5.indices[0]

    for i in range(5):
        labels = imagenet_labels[str(int(top5_indices[i]))]
        prob = "{:.2f}%".format(float(top5_prob[i])*100)
        #print(labels, prob)
        if float(top5_prob[i]) > prob_lim:
            if labels!="picture frame":
                string = filename
                string = np.append(string,os.path.basename(filename))
                string = np.append(string, labels)
                string = np.append(string,prob)
                #f = open("s:/labels.csv", 'a', newline='')
                writer = csv.writer(f)
                writer.writerow(string)
f.close()
    #time.sleep(5)
#plt.imshow(img)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
