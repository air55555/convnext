# --------------------------------------------------------
# ImageNet-21K Pretraining for The Masses
# Copyright 2021 Alibaba MIIL (c)
# Licensed under MIT License [see the LICENSE file for details]
# Written by Tal Ridnik
# --------------------------------------------------------
import os
import glob,csv
from datetime import datetime
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import urllib
from argparse import Namespace
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from src_files.semantic.semantics import ImageNet21kSemanticSoftmax
import timm,sys
import numpy as np
prob_lim = float(sys.argv[1])
path = sys.argv[2]
############### Downloading metadata ##############
print("downloading metadata...")
url, filename = (
    "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/fall11/imagenet21k_miil_tree.pth",
    "imagenet21k_miil_tree.pth")
if not os.path.isfile(filename):
    urllib.request.urlretrieve(url, filename)
args = Namespace()
args.tree_path = filename
semantic_softmax_processor = ImageNet21kSemanticSoftmax(args)
print("done")

############### Loading (ViT) model from timm package ##############
print("initilizing model...")
model = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)#.cuda()
model = torch.nn.DataParallel(model, device_ids=list(range(0,4))).cuda()
model.eval()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
print("done")

files = glob.glob(path)
#files = glob.glob("S:/good_imgs/1/*.jpg")#s:/content/*.jpg

#files.sort(key=os.path.getmtime,reverse=True)

f = open("s:/labels21.csv", 'w', newline='')
writer = csv.writer(f)

string = str(datetime.now()).replace(" ", "|")
string = np.append(string, " ---start-----")
string =["path","f","label","prob"]
writer.writerow(string)
f.close()
f = open("s:/labels21.csv", 'a', newline='')
for file in files:

    filename = os.fsdecode(file)
    print(f'img21 {prob_lim} -{filename}')
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = Image.open(filename).convert('RGB')
        tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

        lbls = []
        probs= []
        with torch.no_grad():
            logits = model(tensor)
            semantic_logit_list = semantic_softmax_processor.split_logits_to_semantic_logits(logits)

            # scanning hirarchy_level_list
            for i in range(len(semantic_logit_list)):
                logits_i = semantic_logit_list[i]

                # generate probs
                probabilities = torch.nn.functional.softmax(logits_i[0], dim=0)
                top1_prob, top1_id = torch.topk(probabilities, 1)

                if top1_prob > prob_lim:
                    top_class_number = semantic_softmax_processor.hierarchy_indices_list[i][top1_id[0]]
                    top_class_name = semantic_softmax_processor.tree['class_list'][top_class_number]
                    top_class_description = semantic_softmax_processor.tree['class_description'][top_class_name]
                    lbls.append(top_class_description)
                    probs.append(top1_prob)
        #print(f"labels found {lbls}.{probs}")
        for i in range(0,len(lbls)):
            labels = lbls[i]
            prob = "{:.2f}%".format(float(probs[i]) * 100)
            #print(labels, prob)
            string = filename
            string = np.append(string, os.path.basename(filename))
            string = np.append(string, labels)
            string = np.append(string, prob)

            writer = csv.writer(f)
            writer.writerow(string)
f.close()

############## Visualization ##############
# import matplotlib
# import os
# import numpy as np
#
# if os.name == 'nt':
#     matplotlib.use('TkAgg')
# else:
#     matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# plt.imshow(img)
# plt.axis('off')
# plt.title('Semantic labels found: \n {}'.format(np.array(labels)))
# plt.show()
