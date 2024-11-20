# Nvidia AI Specialist Certification

- ***An AI of Detecting of Hand Images & Video***

- **📃Table of Contents**
    1. Outline of Project
    2. General Description & Procedure
    3. Proposal for enhancements to the project
    4. Values and Importance of the Project
    5. Current Limitation
    6. Review
    -ref

# **Outline of Project**

### **Title:  An AI of Detecting of Hand Images & Video**

---

## Motivation & Background

There are many AIs that translate spoken language, but there is a lack of AIs that translate sign language despite the need for them. In this global era, where communication is highly valued, people who cannot speak face significant inconveniences. Writing is one form of communication, but many individuals cannot write. For such individuals, sign language is their primary means of communication. Globally, about 70 million people use sign language, most of whom are deaf or hard of hearing. However, there are few who can communicate freely through sign language. Moreover, sign language is different every country.

Recently, there was a case where someone pretended to be a sign language interpreter without actually knowing sign language. My goal is to prevent such situations and make it easier for people to understand sign language.

For them, I propose detecting  AI of Hands images & Video.

# General Description & Procedure

### I separated hand to detail parts - Right/Left hand, Thumb, Index finger, Middle finger, Ring finger, Little finger, Palm

There are many translation programs and AI of language. Language is very important to communication and culture. Also , it is means to express one’s think, emotion, and sharing culture. So many AI developer are making auto-translating program in current days. But someone use hands to communication. These people don’t use sound, but hands and face. They have little community because people don’t know sign language if they were not incapable of speech.

So sign language detect AI will useful for deaf and hard of hearing.

Hand sign requires detecting difference among fingers. Using different finger means different sign. According to these needs, I separate fingers to thumb, index finger, middle finger, ring finger, little finger and palm.

I got data , video of hand  my self. I took videos in a background-free environment.

Labeling with DarkLabel, and classes are righthand, lefthand, thumb, index, middle, ring, little, palm.

I used yolov5-yolov5n model, and for ref, I used yolov5m model.

## Procedure

---

Environments

<aside>
💡

- Vapmix2 - edit video(640x640)
- DarkLabel
- Yolov5 (use yolov5n.pt - ref. only finger, palm, and Right/Left using yolov5m.pt)
</aside>

### 1. Getting Video

[Taking self Video of Right Hand](https://github.com/user-attachments/assets/1e11f422-2855-4d30-817b-06c23d94d3a6)

Taking self Video of Right Hand

- I took videos my hands with iphone 13pro for getting Data
- Right/Left Hands
- I cut the video by 640x640 with vapmix2, video edit program

### 2. Extract Images from Video - DarkLabel

- Add class in DarkLabel.yaml

```python
# Predefined Classes Sets
hands_classes: ["righthand","lefthand","finger","palm"]
hand_classes: ["righthand", "lefthand", "thumb","index","middle","ring","little","palm"]

# Define Format
format10:    # darknet yolo (predefined format]
  fixed_filetype: 1                 # if specified as true, save setting isn't changeable in GUI
  data_fmt: [classid, ncx, ncy, nw, nh]
  gt_file_ext: "txt"                 # if not specified, default setting is used
  gt_merged: 0                    # if not specified, default setting is used
  delimiter: " "                     # if not spedified, default delimiter(',') is used
  classes_set: "hands_classes"     # if not specified, default setting is used
  name: "hands"           # if not specified, "[fmt%d] $data_fmt" is used as default format name
  
format11:    # darknet yolo (predefined format]
  fixed_filetype: 1                 # if specified as true, save setting isn't changeable in GUI
  data_fmt: [classid, ncx, ncy, nw, nh]
  gt_file_ext: "txt"                 # if not specified, default setting is used
  gt_merged: 0                    # if not specified, default setting is used
  delimiter: " "                     # if not spedified, default delimiter(',') is used
  classes_set: "hand_classes"     # if not specified, default setting is used
  name: "hand"           # if not specified, "[fmt%d] $data_fmt" is used as default format name
```

- Extract as Images

![DarkLabel](%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-11-15_125541.png)

DarkLabel

- set class with mine

![set class](image.png)

set class

### 3. Labeling Images-Right/Left hand, Thumb, Index finger, Middle finger, Ring finger, Little finger, Palm

 - ref. Right/Left hand, finger

- Labeling - using DarkLabel(GT save as…)
    
    ![Labeling saved as txt](%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-11-15_125857.png)
    
    Labeling saved as txt
    

### 4. Data Processing

- Data processing using Numpy & Tensorflow - provided by Yolov5

```python
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.python.eager.context import eager_mode

def _preproc(image, output_height=512, output_width=512, resize_side=512):
    ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h*scale), int(w*scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        return tf.squeeze(cropped_image)

def Create_npy(imagespath, imgsize, ext) :
    images_list = [img_name for img_name in os.listdir(imagespath) if
                os.path.splitext(img_name)[1].lower() == '.'+ext.lower()]
    calib_dataset = np.zeros((len(images_list), imgsize, imgsize, 3), dtype=np.float32)

    for idx, img_name in enumerate(sorted(images_list)):
        img_path = os.path.join(imagespath, img_name)
        try:
            # 파일 크기가 정상적인지 확인
            if os.path.getsize(img_path) == 0:
                print(f"Error: {img_path} is empty.")
                continue

            img = Image.open(img_path)
            img = img.convert("RGB")  # RGBA 이미지 등 다른 형식이 있을 경우 강제로 RGB로 변환
            img_np = np.array(img)

            img_preproc = _preproc(img_np, imgsize, imgsize, imgsize)
            calib_dataset[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
            print(f"Processed image {img_path}")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    np.save('calib_set.npy', calib_dataset)
```

### 5. Learning

- Learning Images with Label - provided by Yolov5

```python
#모델 학습하기
!python train.py  --img 512 --batch 16 --epochs 300 --data /content/drive/MyDrive/yolov5/data.yaml --weights yolov5n.pt --cache
```

- about 2h 30m spent
- ref - just finger, palm and Right/Left
    
    ```python
    !python train.py  --img 512 --batch 16 --epochs 100 --data /content/drive/MyDrive/yolov5/data.yaml --weights yolov5m.pt --cache
    ```
    

### 6. Get Results
[Detect_video](https://github.com/user-attachments/assets/f85a2684-98f5-4275-a405-39e86144b54c)

- val_batchlabels_correlogram

![val_batch2_labels.jpg](val_batch2_labels.jpg)

![val_batch2_labels.jpg](val_batch2_labels%201.jpg)

![labels_correlogram.jpg](87747e32-3fb3-40bc-a612-3757db1d1c85.png)

- results
    
    ![F1_curve.png](F1_curve.png)
    
- Results
    
    ![results.png](results.png)
    

### 7. Detecting Test

- images

```python
!python detect.py --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/yolov5/Train/images
```

- videos

```python
!python detect.py --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/yolov5/Train/videos
```

![KakaoTalk_20241115_215959497_01.jpg](ef80e559-2d59-468d-963a-d943c9ea2957.png)

# Proposal for enhancements to the project

---

As importance of communication in global advances, sign language will be translated easily and in every country, they can communicate with people. If the sign language is different, they can use sign-language translator in their phone.

# Values and Importance of this Project

---

Sign language is different in different country. They also need sign language translator to communicate with people in another county. Moreover, there is no normalization in sign. ISL

(International Sign Language) exist but difficult to learn. So AI detecting hand and sign language will useful for translating words and can be used by many deaf and hard of hearing.

# Current Limitations

---

Except thumb, other fingers can’t distinguish easily, because those don’t have features to distinct them. Relative size among them is only way to separate and perceive the difference. And it needs learning varied skin color. 

![KakaoTalk_20241115_215959497_02.jpg](1c989b7e-73f5-4681-90cb-7cb857726443.png)

# Review

---

It was time to patience, endurance and waiting. First Learn images was over 4000. With my computer- cpu, it takes over 30 hours for 300 epochs. But I know the time was worth, and I’m convinced my project will help many people.

As well as learning and using Yolov5, comprehensive understanding of image recognition, object detection, and classification is necessary for this project.

### ref.

Small practicing project

<aside>
📎

[finger detect-using yolov5m model](https://github.com/user-attachments/assets/8a37fc19-5816-4588-b18e-5695a6d3e8da)

finger detect-using yolov5m model

- Results

![val_batch](val_batch1_pred.jpg)

val_batch

![results](results_(1).png)

results

![F1_curve](F1_curve_(1).png)

F1_curve

![labels_correlogram](de682bcb-97e7-4a61-853b-3da936a06daa.png)

labels_correlogram

</aside>
