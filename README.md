# Smile Detection with Webcam
 The goal of this project is to develop a system that can accurately detect smiles in real-time using machine learning techniques. The system relies on two main features for classification: Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP). These features are extracted from facial images and used to train a Support Vector Machine (SVM) classifier.

 ## Part 1: Data Preprocessing and Face Cropping
The `data_preprocessing.py` script is responsible for loading the images from the Genki-4k dataset, detecting faces in the images using the Haar Cascade classifier, cropping the images to show only the detected faces, and resizing the cropped images to a uniform size. The processed images are saved in the `cropped_images` folder.

## Part 2: Feature Extraction and Model Training
The `webcam_smile_detection.py` script loads the SVM model, accesses the webcam, and performs real-time smile detection on the video feed. The script detects faces using the Haar Cascade classifier, extracts features from the detected faces, and uses the trained SVM model to predict whether the detected faces are smiling or not. The webcam feed is displayed in real-time, with a rectangle drawn around smiling faces and a "Smiling" label displayed above them.

## Part 3: Smile Detection with Webcam
The `webcam_smile_detection.py` script loads the SVM model, accesses the webcam, and performs real-time smile detection on the video feed. The script detects faces using the Haar Cascade classifier, extracts features from the detected faces, and uses the trained SVM model to predict whether the detected faces are smiling or not. The webcam feed is displayed in real-time, with a rectangle drawn around smiling faces and a "Smiling" label displayed above them.

## Requirements
* Python 3.x
* OpenCV (cv2)
* Joblib
* NumPy
* Scikit-image (skimage)
* Scikit-learn (sklearn)
* ImgAug (imgaug)
* tqdm

## How to Use
1. Install Dependencies: Make sure you have Python 3.x installed. Install the required libraries using the following command:
```bash
pip install opencv-python joblib numpy scikit-image scikit-learn imgaug tqdm
```

2. Clone the Repository: Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/Ali-Pourgheysari/Smile-detection-SVM.git
```
3. Run the Scripts: Execute the scripts in the following order:
```bash 
python data_preprocessing.py #  Preprocess the images, detect faces, and resize the cropped faces.
python train_model.py # Extract features, apply data augmentation, train the SVM model, and save it as svm_model_temp.joblib.
python test_model.py # Load the trained SVM model and use the webcam for real-time smile detection.
```
Note: Make sure to place the appropriate Haar Cascade classifier (`haarcascade_frontalface_default.xml`) in the same directory as the scripts.

## Credits
The project uses the Genki-4k dataset and the OpenCV library for face detection. The SVM classifier is trained on extracted HOG and LBP features. The Smile Detection with Webcam project serves as a basic example of real-time smile detection using machine learning techniques.