import cv2

# load images of Genki-4k 
def load_data():
    images = []
    # load images of dataset
    for i in range(1, 4001):
        # the dataset is in Genki_4k folder. change this to your dataset image path
        img = cv2.imread(f'Genki_4K/files/file{i:04}.jpg')
        images.append(img)
    
    # load labels. the labels are in the below path. change this to your dataset labels path
    labels = []
    with open('Genki_4K/labels.txt', 'r') as f:
        for line in f:
            element = line.split()[0]
            labels.append(element)

    return images, element

# detect and crop face
def detect_face_crop(images):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cropped_images = []
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Crop the image to show only the detected face(s)
        for (x, y, w, h) in faces:
            cropped_img = img[y:y+h, x:x+w]

        cropped_images.append(cropped_img)

    return cropped_images

# save all images
def save_images(images):
    for i, img in enumerate(images):
        cv2.imwrite(f'Genki_4k/cropped_images/file{i:04}.jpg', img)

def resize(images):
    resized_images = []
    # Define the new size. (258, 258) is the maximum image size when we crop
    new_size = (258, 258)

    for img in images:

        # Resize the image
        resized_img = cv2.resize(img, new_size)
        resized_images.append(resized_img)

    return resized_images
    
# main function
def main():
    images, _ = load_data()
    cropped_images = detect_face_crop(images)
    resized_images = resize(cropped_images)
    save_images(resized_images)


if __name__ == '__main__':
    main()