import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from sklearn import svm
import joblib
from image_normalize import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extreact_features(images):

    # Assuming your images are stored in a list called "images"
    hog_features = []
    lbp_features = []

    for image in images:
        # Convert the image to grayscale
        gray_image = rgb2gray(image)

        # Extract HOG features
        hog_feature_vector = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                 cells_per_block=(1, 1), channel_axis=-1)
        hog_features.append(hog_feature_vector)

        # Extract LBP features
        lbp_feature_vector = local_binary_pattern(gray_image, 24, 3).ravel()

        # histogram of LBP features
        hist, _ = np.histogram(lbp_feature_vector, 256, (0, 256))
        lbp_features.append(hist)

    # Combine the feature vectors into a single matrix
    features = [[*x, *y] for x, y in zip(hog_features, lbp_features)]
    return features


def train_and_save(X_train, y_train, model_name):
    # Choose an appropriate kernel function
    model = svm.SVC(gamma="auto", kernel='rbf')

    # Train the SVM model on the training set
    model.fit(X_train, y_train)

    print("Train accuracy:", model.score(X_train, y_train))

    # Save the trained classifier using joblib
    joblib.dump(model, model_name)


def test(X_test, y_test, model_name):

    # Load SVM model from disk
    model = joblib.load(model_name)

    # Make predictions on the testing set using the loaded model
    y_pred = model.predict(X_test)

    # Evaluate the performance of the loaded model
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy:", accuracy)


def main():
    # load data
    images, labels = load_data(filename='Genki_4K/cropped_images')

    # shuffle images and labels
    combined = list(zip(images, labels))
    np.random.shuffle(combined)
    shuffled_images, shuffled_labels = zip(*combined)

    # extract features
    features = extreact_features(shuffled_images)

    # split data of train and test
    X_train, X_test, y_train, y_test = train_test_split(
        features, shuffled_labels, test_size=0.25)

    model_name = 'svm_model.joblib'

    # train and test the model
    train_and_save(X_train, y_train, model_name)
    test(X_test, y_test, model_name)


if __name__ == '__main__':
    main()
