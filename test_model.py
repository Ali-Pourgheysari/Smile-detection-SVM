import numpy as np
import cv2
import joblib
from train_model import hog_lbp


def webcam_smile_detectiion(model, model_name = 'svm_model.joblib'):
    cap=cv2.VideoCapture(0)
    model = joblib.load(model_name)
    # fgbg=cv2.createBackgroundSubtractorMOG2()

    while True:
        # Read frame from video capture
        ret, frame = cap.read()
        if ret:
        # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using a face detector
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop over all detected faces
            for (x, y, w, h) in faces:
                # Extract features from the detected face using a pre-trained deep learning model
                # and reshape the features to match the input shape of the SVM classifier
                size = (250, 250)
                resized_image = cv2.resize(frame[y:y+h, x:x+w], size)
                hog, lbp = hog_lbp(resized_image)

                features = []
                features.extend(hog)
                features.extend(lbp)

                features = np.array(features)
                features = features.reshape(1,-1)
                # Use the SVM classifier to predict whether the detected face is smiling or not
                prediction = model.predict(features)

                # If the SVM predicts a smile, draw a rectangle around the face and display a message
                if int(*prediction) == 1:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Smile Detector', frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




def main():
    model_name = 'svm_model.joblib'
    model = joblib.load(model_name)

    webcam_smile_detectiion(model)


if __name__ == '__main__':
    main()