# SmileDetection

Smile Detection project codes for course cs124, SJTU.

# Problem Description

Given a picture of a person, could you tell me whether he/she is smiling? Please let your computer to give the answer.


# Getting Started
## Installation
- This code was tested with Python 3.7, windows 10
- Dataset [GENKI-4K](http://mplab.ucsd.edu/wordpress/wp-content/uploads/genki4k.tar) should be downloaded to train the models. 
- **data_faces** are face images gernerated from orignal GENKI-4K (using opencv face detector).
- **xmls** containes two xml files from opencv.
- **img_label.txt** is the face image names and their labels. The images that cannot be detected faces by opencv are discarded.
- Clone this repo:
```
git clone https://github.com/huqiaoping/SmileDetection
cd SmileDetection
```

## Preparing
```
pip3 install numpy
pip3 install opencv-python
pip3 install scikit-learn
pip3 install scikit-image
pip3 install pillow
```

## Task 1: Face Detection with Opencv

Run ```face_detection.py``` to detecting face in example.jpg; Run ```face_detection.py --use_camera True``` to detecting faces from your camera real-time.

## Task 2: Smile Detection Models Training

Run ```train_smile_detection_model.py``` to train smile detection models. 10-fold cross validation is utilized.

## Task 3: Real-time Smile Detection 

Run ```realtime_detect_smiles.py ``` to detect smiles.
