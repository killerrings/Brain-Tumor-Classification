import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image

html_temp = """
<div style="background-color:tomato; padding:10px">
<h2 style ="color:white; text-align:center;">Brain Tumor MRI Classification</h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

st.write("")

activities=["Classification", "About"]
choice = st.sidebar.selectbox('Select Activity', activities)

if choice == 'Classification':
   
    uploaded_file = st.file_uploader("Choose a brain MRI", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', width=250)
        st.write("")
        predict_btn = st.button('Predict')
        if predict_btn:
            st.write("Classifying...")
            label = teachable_machine_classification(image, 'brain_tumor_model.h5')
            if label == 0:
                st.warning("The MRI scan has a brain tumor")
            else:
                st.success("The MRI scan is healthy")

elif choice == 'About':
        st.subheader('About')
        st.write("This is a simple Brain Tumor MRI Classification Deep Learning Web App made using Streamlit.")
        st.write("Streamlit’s open-source app framework is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.")
        html_temp_1 = """
        <a href="https://github.com/killerrings">Github</a>"""
        st.markdown(html_temp_1, unsafe_allow_html=True)