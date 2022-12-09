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
import urllib
import time
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
def check_image(filename,transform):
    matches = ["pen", "ink"]


    # Print top categories per image
    top5_prob ,top5_catid = get_description(filename,transform)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    # blablabla

    string = filename
    for i in range(5):
        labels = categories[top5_catid[i]]
            #imagenet_labels[str(int(top5_indices[i]))]
        pr= float(top5_prob[i].item())*100
            #float(top5_prob[i]) * 100

        prob = "{:.2f}%".format(pr)
            #float(top5_prob[i]) * 100)
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

def create_vit_model():
    from transformers import ViTFeatureExtractor, ViTModel
    from PIL import Image
    import requests
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    md_name = "google/vit-huge-patch14-224-in21k"#'google/vit-base-patch32-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(md_name)
    model = ViTModel.from_pretrained(md_name)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
def create_bert_model():
    from transformers import BeitFeatureExtractor, BeitForImageClassification
    from PIL import Image
    import requests
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    #'microsoft/beit-base-patch16-224-pt22k-ft22k'
    md= "microsoft/beit-large-patch16-224-pt22k-ft22k"
    feature_extractor = BeitFeatureExtractor.from_pretrained(md)
    model = BeitForImageClassification.from_pretrained(md)
    model.eval()
    model.to(device)
    return model,feature_extractor,device

def create_convnext_model():
    ordinal='0'
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    #device = torch.device('cuda:{}'.format(ordinal))
    #model_name = "convnext_xlarge_in22k"

    # !!! Working init model from transformers import AutoFeatureExtractor, ViTForImageClassification
    # from PIL import Image
    # import requests
    # feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-patch16-224')
    # model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224')

    # from transformers import CLIPProcessor, CLIPModel
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model_name = "vit_huge_patch14_224_in21k"
    model_name = "hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k"
    model_name = 'hf_hub:nateraw/resnet50-oxford-iiit-pet'

    #model_name = "hf_hub:timm/eca_nfnet_l0"
    #model_name = "vit_base_patch16_224"
    # https://rwightman.github.io/pytorch-image-models/models/noisy-student/
    #model_name = "tf_efficientnet_b0_ns"
    #model_name = "convnext_xlarge_384_in22ft1k"
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device for convnext = ", device)
    #device='cuda'
    # create a ConvNeXt model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
   # model = create_model(model_name, pretrained=True).to(device)
    #labls= model.pretrained_cfg['labels']
   # model.eval()

    # Create Transform
  #  transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # Get the labels from the model config



    return model,transform,device
def get_description(fname,transform):

    img = PIL.Image.open(fname)
    x = transform(img).unsqueeze(0)
    out = convnext_model(x)

    # Apply softmax to get predicted probabilities for each class
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    # Grab the values and indices of top 5 predicted classes
    values, indices = torch.topk(probabilities, top_k)

    # Prepare a nice dict of top k predictions
    predictions = [
        {"label": labels[i], "score": v.item()}
        for i, v in zip(indices, values)
    ]
    print(predictions)
    #img_tensor = transform(img).unsqueeze(0).to(convnext_device)

    # inference
   # with torch.no_grad():
    #    out = convnext_model(img_tensor)
    # probabilities = torch.nn.functional.softmax(out[0], dim=0)
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    return top5_prob ,top5_catid


    output = torch.softmax(convnext_model(img_tensor), dim=1)
    top5 = torch.topk(output, k=5)

    return top5

# url, filename = (
#     "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
# urllib.request.urlretrieve(url, filename)
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
prob_lim = float(sys.argv[1])
path = sys.argv[2]
# imagenet_labels = json.load(open('label_to_words.json'))
#create_vit_model()

model, feature_extractor,device = create_bert_model()
#convnext_model, transform, convnext_device = create_convnext_model()
#labels = convnext_model.pretrained_cfg['labels']
#top_k = min(len(labels), 5)


files = glob.glob(path)
#files = glob.glob("S:/good_imgs/1/*.jpg")#s:/content/*.jpg

#files.sort(key=os.path.getmtime,reverse=True)

f = open("s:/labels_hf_beit.csv", 'w', newline='')
writer = csv.writer(f)

string = str(datetime.now()).replace(" ", "|")
string = np.append(string, " ---start-----")
string =["path","f","label","prob"]
writer.writerow(string)
categories = model.config.id2label
for file in files:
    filename = os.fsdecode(file)
    print(f' Hug Face beit-{filename}')
    #if filename.endswith(".png") or filename.endswith(".jpg"):
    image = PIL.Image.open(filename)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 21,841 ImageNet-22k classes
    predicted_class_idx = logits.argmax(-1).item()
    top5 = torch.topk(logits, k=15)
    top5_prob = top5.values[0]
    top5_catid = top5.indices[0]
   # print("Predicted class:", model.config.id2label[predicted_class_idx])
    #top5_prob ,top5_catid= get_description(filename,transform)
            #blablabla
    #rc= check_image(filename,transform)
    # if rc>60:
    #     pass
    #top5_prob = top5.values[0]
   # top5_indices = top5.indices[0]
    #continue
    for i in range(len(top5_prob)):
        labels = categories.get(int(top5_catid[i]), "Not_exist")
        #labels = categories[top5_catid[i]]
            #imagenet_labels[str(int(top5_indices[i]))]

        pr= float(top5_prob[i].item())#*100
            #float(top5_prob[i]) * 100

        prob = "{:.2f}%".format(pr)
        if float(pr/100) > prob_lim:
        # labels = imagenet_labels[str(int(top5_indices[i]))]
        # prob = "{:.2f}%".format(float(top5_prob[i])*100)
        # #print(labels, prob)
        # if float(top5_prob[i]) > prob_lim:
        #     if labels!="picture frame":
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
