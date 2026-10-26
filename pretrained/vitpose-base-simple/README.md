---
library_name: transformers
license: apache-2.0
language:
- en
pipeline_tag: keypoint-detection
---

# Model Card for VitPose

<img src="https://cdn-uploads.huggingface.co/production/uploads/6579e0eaa9e58aec614e9d97/ZuIwMdomy2_6aJ_JTE1Yd.png" alt="x" width="400"/>

ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation and ViTPose+: Vision Transformer Foundation Model for Generic Body Pose Estimation. It obtains 81.1 AP on MS COCO Keypoint test-dev set.

## Model Details

Although no specific domain knowledge is considered in the design, plain vision transformers have shown excellent performance in visual recognition tasks. However, little effort has been made to reveal the potential of such simple structures for
pose estimation tasks. In this paper, we show the surprisingly good capabilities of plain vision transformers for pose estimation from various aspects, namely simplicity in model structure, scalability in model size, flexibility in training paradigm,
and transferability of knowledge between models, through a simple baseline model called ViTPose. Specifically, ViTPose employs plain and non-hierarchical vision
transformers as backbones to extract features for a given person instance and a
lightweight decoder for pose estimation. It can be scaled up from 100M to 1B
parameters by taking the advantages of the scalable model capacity and high
parallelism of transformers, setting a new Pareto front between throughput and performance. Besides, ViTPose is very flexible regarding the attention type, input resolution, pre-training and finetuning strategy, as well as dealing with multiple pose
tasks. We also empirically demonstrate that the knowledge of large ViTPose models
can be easily transferred to small ones via a simple knowledge token. Experimental
results show that our basic ViTPose model outperforms representative methods
on the challenging MS COCO Keypoint Detection benchmark, while the largest
model sets a new state-of-the-art, i.e., 80.9 AP on the MS COCO test-dev set. The
code and models are available at https://github.com/ViTAE-Transformer/ViTPose

### Model Description

This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated.

- **Developed by:** Yufei Xu, Jing Zhang, Qiming Zhang, Dacheng Tao
- **Funded by:** ARC FL-170100117 and IH-180100002.
- **License:** Apache-2.0
- **Ported to 🤗 Transformers by:** Sangbum Choi and Niels Rogge

### Model Sources

- **Original repository:** https://github.com/ViTAE-Transformer/ViTPose
- **Paper:** https://arxiv.org/pdf/2204.12484
- **Demo:** https://huggingface.co/spaces?sort=trending&search=vitpose

## Uses

The ViTPose model, developed by the ViTAE-Transformer team, is primarily designed for pose estimation tasks. Here are some direct uses of the model:

Human Pose Estimation: The model can be used to estimate the poses of humans in images or videos. This involves identifying the locations of key body joints such as the head, shoulders, elbows, wrists, hips, knees, and ankles.

Action Recognition: By analyzing the poses over time, the model can help in recognizing various human actions and activities.

Surveillance: In security and surveillance applications, ViTPose can be used to monitor and analyze human behavior in public spaces or private premises.

Health and Fitness: The model can be utilized in fitness apps to track and analyze exercise poses, providing feedback on form and technique.

Gaming and Animation: ViTPose can be integrated into gaming and animation systems to create more realistic character movements and interactions.


## Bias, Risks, and Limitations

In this paper, we propose a simple yet effective vision transformer baseline for pose estimation,
i.e., ViTPose. Despite no elaborate designs in structure, ViTPose obtains SOTA performance
on the MS COCO dataset. However, the potential of ViTPose is not fully explored with more
advanced technologies, such as complex decoders or FPN structures, which may further improve the
performance. Besides, although the ViTPose demonstrates exciting properties such as simplicity,
scalability, flexibility, and transferability, more research efforts could be made, e.g., exploring the
prompt-based tuning to demonstrate the flexibility of ViTPose further. In addition, we believe
ViTPose can also be applied to other pose estimation datasets, e.g., animal pose estimation [47, 9, 45]
and face keypoint detection [21, 6]. We leave them as the future work.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch
import requests
import numpy as np

from PIL import Image

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

url = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# ------------------------------------------------------------------------
# Stage 1. Detect humans on the image
# ------------------------------------------------------------------------

# You can choose detector by your choice
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

inputs = person_image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
)
result = results[0]  # take first image results

# Human label refers 0 index in COCO dataset
person_boxes = result["boxes"][result["labels"] == 0]
person_boxes = person_boxes.cpu().numpy()

# Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

# ------------------------------------------------------------------------
# Stage 2. Detect keypoints for each person found
# ------------------------------------------------------------------------

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes], threshold=0.3)
image_pose_result = pose_results[0]  # results for first image

for i, person_pose in enumerate(image_pose_result):
    print(f"Person #{i}")
    for keypoint, label, score in zip(
        person_pose["keypoints"], person_pose["labels"], person_pose["scores"]
    ):
        keypoint_name = model.config.id2label[label.item()]
        x, y = keypoint
        print(f" - {keypoint_name}: x={x.item():.2f}, y={y.item():.2f}, score={score.item():.2f}")

```
Output:
```
Person #0
 - Nose: x=428.72, y=170.61, score=0.92
 - L_Eye: x=429.47, y=167.83, score=0.90
 - R_Eye: x=428.73, y=168.16, score=0.79
 - L_Ear: x=433.88, y=167.35, score=0.94
 - R_Ear: x=441.09, y=166.86, score=0.90
 - L_Shoulder: x=440.02, y=177.15, score=0.93
 - R_Shoulder: x=446.28, y=178.39, score=0.74
 - L_Elbow: x=436.88, y=197.90, score=0.92
 - R_Elbow: x=433.35, y=201.22, score=0.54
 - L_Wrist: x=431.45, y=218.66, score=0.88
 - R_Wrist: x=420.09, y=212.80, score=0.96
 - L_Hip: x=444.81, y=224.16, score=0.81
 - R_Hip: x=452.33, y=223.91, score=0.82
 - L_Knee: x=442.24, y=256.03, score=0.83
 - R_Knee: x=451.12, y=255.20, score=0.82
 - L_Ankle: x=443.20, y=288.18, score=0.60
 - R_Ankle: x=456.03, y=285.76, score=0.82
Person #1
 - Nose: x=398.12, y=181.71, score=0.87
 - L_Eye: x=398.45, y=179.73, score=0.82
 - R_Eye: x=396.07, y=179.45, score=0.90
 - R_Ear: x=388.85, y=180.22, score=0.88
 - L_Shoulder: x=397.24, y=194.16, score=0.76
 - R_Shoulder: x=384.60, y=190.74, score=0.64
 - L_Wrist: x=402.25, y=207.03, score=0.33
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Dataset details. We use MS COCO [28], AI Challenger [41], MPII [3], and CrowdPose [22] datasets
for training and evaluation. OCHuman [54] dataset is only involved in the evaluation stage to measure
the models’ performance in dealing with occluded people. The MS COCO dataset contains 118K
images and 150K human instances with at most 17 keypoint annotations each instance for training.
The dataset is under the CC-BY-4.0 license. MPII dataset is under the BSD license and contains
15K images and 22K human instances for training. There are at most 16 human keypoints for each
instance annotated in this dataset. AI Challenger is much bigger and contains over 200K training
images and 350 human instances, with at most 14 keypoints for each instance annotated. OCHuman
contains human instances with heavy occlusion and is just used for val and test set, which includes
4K images and 8K instances.


#### Training Hyperparameters

- **Training regime:** ![image/png](https://cdn-uploads.huggingface.co/production/uploads/6579e0eaa9e58aec614e9d97/Gj6gGcIGO3J5HD2MAB_4C.png)

#### Speeds, Sizes, Times

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6579e0eaa9e58aec614e9d97/rsCmn48SAvhi8xwJhX8h5.png)

## Evaluation

OCHuman val and test set. To evaluate the performance of human pose estimation models on the
human instances with heavy occlusion, we test the ViTPose variants and representative models on
the OCHuman val and test set with ground truth bounding boxes. We do not adopt extra human
detectors since not all human instances are annotated in the OCHuman datasets, where the human
detector will cause a lot of “false positive” bounding boxes and can not reflect the true ability of
pose estimation models. Specifically, the decoder head of ViTPose corresponding to the MS COCO
dataset is used, as the keypoint definitions are the same in MS COCO and OCHuman datasets.

MPII val set. We evaluate the performance of ViTPose and representative models on the MPII val
set with the ground truth bounding boxes. Following the default settings of MPII, we use PCKh
as metric for performance evaluation.

### Results

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6579e0eaa9e58aec614e9d97/FcHVFdUmCuT2m0wzB8QSS.png)


### Model Architecture and Objective

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6579e0eaa9e58aec614e9d97/kf3e1ifJkVtOMbISvmMsM.png)

#### Hardware

The models are trained on 8 A100 GPUs based on the mmpose codebase


## Citation

**BibTeX:**

```bibtex
@article{xu2022vitposesimplevisiontransformer,
  title={ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  year={2022},
  eprint={2204.12484},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2204.12484}
}
```