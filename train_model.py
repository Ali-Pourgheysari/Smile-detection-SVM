import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from sklearn import svm
import joblib
import image_normalize as imn

def extreact_features(images):
    
    # Assuming your images are stored in a list called "images"
    hog_features = []
    lbp_features = []

    for image in images:
        # Convert the image to grayscale
        gray_image = rgb2gray(image)
        
        # Extract HOG features
        hog_feature_vector = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        hog_features.append(hog_feature_vector)
        
        # Extract LBP features
        lbp_feature_vector = local_binary_pattern(gray_image, P=8, R=5).ravel()
        lbp_features.append(lbp_feature_vector)

    # Combine the feature vectors into a single matrix
    features = np.hstack((np.array(hog_features), np.array(lbp_features)))
    return features

def train_and_save(X_train, y_train):
    # Choose an appropriate kernel function
    model = svm.SVC(kernel='rbf')

    # Train the SVM model on the training set
    model.fit(X_train, y_train)

    # Save the trained classifier using joblib
    joblib.dump(model, 'svm_model.joblib')

# main function
def main():
    images, labels = imn.load_data()
    features = extreact_features(images)
    train_and_save(features, labels)
    

if __name__ == '__main__':
    main()