import cv2
import numpy as np
import pandas as pd


# load images of Genki-4k 
def load_data():
    images = []
    # load images of dataset
    for i in range(1, 4001):
        # the dataset is in Genki_4k folder. change this to your dataset image path
        img = cv2.imread(f'Genki_4K/files/file{i:04}.jpg')
        images.append(img)
    
    # load labels. the labels are in the below path. change this to your dataset labels path
    labels = pd.read_csv('Genki_4K/labels.txt', sep=" ", header=None)

    return images, labels

def detect_face(images):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cropped_images = []
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        # Crop the image to show only the detected face(s)
        for (x, y, w, h) in faces:
            cropped_img = img[y:y+h, x:x+w]

        cropped_images.append(cropped_img)

    return cropped_images

# main function
def main():
    images, labels = load_data()
    images = detect_face(images[:2])
    cv2.imshow("image", images[0])
    cv2.imshow("image1", images[1])
    cv2.waitKey(0)

if __name__ == '__main__':
    main()