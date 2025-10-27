import streamlit as st
import tensorflow as tf
import numpy as np
from keras.applications.ConvNeXtLarge import preprocess_input
from PIL import Image


#####functions#########
def prediction(modelname, sample_image, IMG_SIZE = (224,224)):

    #labels
    labels =  ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Squamous Cell Carcinoma', 'Vascular Lesion']

    try:
        #loading the .h5 model
        load_model = tf.keras.models.load_model(modelname)

        sample_image = Image.open(sample_image)
        img_array = sample_image.resize(IMG_SIZE)
        img_batch = np.expand_dims(img_array, axis = 0)
        image_batch = img_batch.astype(np.float32)
        image_batch = preprocess_input(image_batch)
        prediction = load_model.predict(img_batch)
        return labels[int(np.argmax(prediction, axis = 1))]


    except Exception as e:
        st.write("ERROR: {}".format(str(e)))


#Building the website

#title of the web page
st.title("Skin Cancer Classification")

#setting the main picture
st.image(
    "https://www.azskin.com/wp-content/uploads/2021/12/Blog-Art-2-v2.jpg", 
    caption = "Skin Cancer Classification")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.subheader("Skin Cancer Predictions")
    st.write("This web app is about Multiclass Skin Cancer Classification")

#setting the tabs
tab1, tab2 = st.tabs(['Image Upload 👁️', 'Camera Upload 📷'])

#tab1
with tab1:
    #setting file uploader
    #you can change the label name as your preference
    image = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to classify them")

    if image:
        #validating the image type
        image_type = image.type.split("/")[-1]
        if image_type not in ['jpg','jpeg','png','jfif']:
            st.error("Invalid file type : {}".format(image.type), icon="🚨")
        else:
            #displaying the image
            st.image(image, caption = "Uploaded Image")

            #getting the predictions
            label = prediction("Convex_final_model_1", image)

            #displaying the predicted label
            st.success("Your Condition is **{}**".format(label))

with tab2:
    #camera input
    cam_image = st.camera_input("Please take a photo")

    if cam_image:
        #displaying the image
        st.image(cam_image)

        #getting the predictions
        label = prediction("Convex_final_model_1", cam_image)

        #displaying the predicted label
        st.success("Your Condition is **{}**".format(label))