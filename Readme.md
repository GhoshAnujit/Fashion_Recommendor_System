Fashion Recommender System using ResNet50 and Streamlit

This repository contains code for a Fashion Recommender System that uses a pre-trained ResNet50 model for feature extraction and nearest neighbors for finding similar images. The system recommends 5 similar images to the image the user uploads. The user interface (UI) is built with Streamlit.

1. Table of Contents
2. Introduction
3. Installation
4. Usage
5. Model
6. UI
7. Dataset
8. Credits
9. License

Introduction

The Fashion Recommender System is a web application that recommends fashion items (e.g., clothes, shoes, bags) to users based on their preferences. It uses a pre-trained ResNet50 model for feature extraction and nearest neighbors for finding similar images. The UI is built with Streamlit to interact with users.

Installation

To install the Fashion Recommender System, follow these steps:

Clone the repository:

git clone (https://github.com/GhoshAnujit/Fashion_Recommendor_System.git)

Install the required packages:

pip install -r requirements.txt

Download the pre-trained ResNet50 model weights (see Model).

Download the fashion product images dataset (see Dataset).

Usage:

To use the Fashion Recommender System, run the Streamlit app:

streamlit run main.py

The app will open in your browser at http://localhost:8501.

Model:

The Fashion Recommender System uses a pre-trained ResNet50 model for feature extraction. The model was pre-trained on the ImageNet dataset, which contains millions of images of various objects, including fashion items. The pre-trained weights are available in the model folder.

To extract features from the fashion product images dataset, run the extract_features.py script. The script will save the features in a NumPy array in the data folder.

UI:

The Fashion Recommender System UI is built with Streamlit, a Python library for building web applications. The UI allows users to:

Upload an image of a fashion item:

View the uploaded image and 5 similar images
See the predicted class of the uploaded image
The UI is defined in app.py. To run the app, see Usage.

Dataset:

The fashion product images dataset used to find similar images is available at https://www.kaggle.com/paramaggarwal/fashion-product-images-small. The dataset contains 44,410 images of fashion products.

The dataset is not included in this repository. To download the dataset, follow these steps:

Create a data folder in the root directory of the repository:

mkdir data

Download the dataset files from https://www.kaggle.com/paramaggarwal/fashion-product-images-small and save them in the data folder.

Credits

The Fashion Recommender System was created by Anujit Ghosh. The ResNet50 model was pre-trained on the ImageNet dataset, which was created by Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. The UI was built with Streamlit, a Python library for building web applications.

License
The code in this repository is licensed under the MIT License. See the LICENSE file for more details.
