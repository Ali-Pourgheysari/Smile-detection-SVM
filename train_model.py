import joblib
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa

from image_normalize import load_data


def extract_features(images):

    # Assuming your images are stored in a list called "images"
    hog_features = []
    lbp_features = []

    for image in images:
        hog_feature_vector, hist = hog_lbp(image)
        hog_features.append(hog_feature_vector)
        lbp_features.append(hist)

    # Combine the feature vectors into a single matrix
    features = [[*x, *y] for x, y in zip(hog_features, lbp_features)]
    return features

def hog_lbp(image):
    # Convert the image to grayscale
    gray_image = rgb2gray(image)

    # Extract HOG features
    hog_feature_vector = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), channel_axis=-1)

    # Extract LBP features
    lbp_feature_vector = local_binary_pattern(gray_image, 24, 3).ravel()

    # histogram of LBP features
    hist, _ = np.histogram(lbp_feature_vector, 256, (0, 256))
    
    return hog_feature_vector, hist


def train_and_save(X_train, y_train, model_name = 'svm_model.joblib'):
    # Choose an appropriate kernel function
    model = svm.SVC(gamma="auto", kernel='rbf')

    # Train the SVM model on the training set
    model.fit(X_train, y_train)

    print("Train accuracy:", model.score(X_train, y_train))

    # Save the trained classifier using joblib
    joblib.dump(model, model_name)


def test(X_test, y_test, model_name = 'svm_model.joblib'):

    # Load SVM model from disk
    model = joblib.load(model_name)

    # Make predictions on the testing set using the loaded model
    y_pred = model.predict(X_test)

    # Evaluate the performance of the loaded model
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy:", accuracy)

def augmentate_data(x_train, y_train):
    # Define the augmentations to be applied
    seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
    ])

    # Convert the image list to a numpy array
    image_array = np.array(x_train)

    # Apply the augmentations to each image and its corresponding label
    augmented_images, augmented_labels = [], []
    for i in range(len(image_array)):
        augmented_image = seq(image=image_array[i])
        augmented_images.append(augmented_image)
        augmented_labels.append(y_train[i])


    return augmented_images, augmented_labels

def main():
    # load data
    images, labels = load_data(filename='Genki_4K/cropped_images')
    # apply augmentation
    augmentatation_images, augmentation_labels = augmentate_data(images, labels)
    # shuffle images and labels
    combined = list(zip(augmentatation_images + images, augmentation_labels + labels))
    np.random.shuffle(combined)
    shuffled_images, shuffled_labels = zip(*combined)

    # extract features
    features = extract_features(shuffled_images)

    # split data of train and test
    X_train, X_test, y_train, y_test = train_test_split(
        features, shuffled_labels, test_size=0.20)

    # train and test the model
    train_and_save(X_train, y_train, model_name = 'svm_model_temp.joblib')
    test(X_test, y_test, model_name = 'svm_model_temp.joblib')


if __name__ == '__main__':
    main()
