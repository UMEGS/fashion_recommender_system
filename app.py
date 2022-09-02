import pickle
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors



def extract_features(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expended_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expended_img_array)
    result = model.predict(preprocessed_img, verbose=False).flatten()
    normalized_result = result / np.linalg.norm(result)

    return normalized_result


def get_similar_images(query_image, features_list, filenames, model, top_n=5):
    query_features = extract_features(query_image, model)
    distances, indices = NearestNeighbors(n_neighbors=top_n, metric='cosine').fit(features_list).kneighbors([query_features])
    return distances, indices


st.title("Image Similarity Search Engine")
bar = st.progress(0)

bar.progress(25)
features_list = np.array(pickle.load(open('static/model/features_list_embeddings.pkl', 'rb')))
filenames = pickle.load(open('static/model/filenames.pkl', 'rb'))
bar.progress(50)

# model_resnet50 = load_model('static/model/model_resnet50.h5')
model_resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_resnet50.trainable = False
bar.progress(75)

model = Sequential([model_resnet50, GlobalMaxPooling2D(), ])
bar.progress(100)

bar.empty()

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=224)
    st.write("")
    disances, indices = get_similar_images(uploaded_file, features_list, filenames, model, top_n=6)
    st.write('Recommended Images')
    for index,col in enumerate(st.columns(5)):
        name = filenames[indices[0][index]].replace('/kaggle/input/fashion-product-images-small/images/', 'static/dataset/images/')
        col.image(name)






# if __name__ == '__main__':
#     st.starun()
