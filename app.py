import keras.applications
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

tfi = tf.keras.preprocessing.image
tfg = tf.keras.layers.GlobalMaxPooling2D
tfr = tf.keras.applications.resnet50.preprocess_input

model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    tfg()
])

#print(model.summary())

def extract_features(img_path,model):
    img = tfi.load_img(img_path, target_size = (224,224))
    img_array = tfi.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tfr(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

print(np.array(feature_list).shape)

#pickle.dump(feature_list,open('embeddings.pkl','wb'))
#pickle.dump(filenames,open('filenames.pkl','wb'))
