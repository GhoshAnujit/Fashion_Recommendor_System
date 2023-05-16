import pickle
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

tfr = tf.keras.applications.resnet50.preprocess_input
tfi = tf.keras.preprocessing.image
tfg = tf.keras.layers.GlobalMaxPooling2D


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    tfg()
])

img = tfi.load_img('sample/1547.jpg', target_size = (224,224))
img_array = tfi.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = tfr(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)
distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', temp_img)
    cv2.waitKey(0)