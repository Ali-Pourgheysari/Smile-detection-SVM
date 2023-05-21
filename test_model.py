import numpy as np
from skimage.color import rgb2gray
from sklearn import svm
import joblib
from image_normalize import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score