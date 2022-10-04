import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Add and resize an image to the top of the app
img_fuel = Image.open("../img/fuel_efficiency.png")
st.image(img_fuel, width=700)

st.markdown("<h1 style='text-align: center; color: black;'>Fuel Efficiency</h1>", unsafe_allow_html=True)

# Import train dataset to DataFrame
train_df = pd.read_csv("../dat/train_features.csv", index_col=0)
model_results_df = pd.read_csv("../dat/model_results_week_7.csv", index_col=0)
test_labels_df = pd.read_csv("../dat/test_labels.csv", index_col=0)

# Get error and r2 values
y = np.array(test_labels_df)
tpot_results = np.array(model_results_df['TPOT'])
dnn_results = np.array(model_results_df['DNN'])

tpot_error = tpot_results - y
dnn_error = dnn_results - y

tpot_r2 = r2_score(y, tpot_results)
dnn_r2 = r2_score(y, dnn_results)

# Create sidebar for user selection
with st.sidebar:
    # Add FB logo
    st.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png" )    

    # Available models for selection

    # YOUR CODE GOES HERE!
    models = ["DNN", "TPOT"]

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        (models)
    )
    
    # Remove selected model 1 from model list
    # App refreshes with every selection change.
    models.remove(model1_select)
    
    model2_select = st.selectbox(
        "Choose Model 2:",
        (models)
    )

# Create tabs for separation of tasks
tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 Model Results", "🤓 Model Explainability"])

with tab1:    
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(100))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)

with tab2:    
    
    # YOUR CODE GOES HERE!

    # Columns for side-by-side model comparison
    col1, col2 = st.columns(2)

    # Build the confusion matrix for the first model.
    with col1:
        st.header(model1_select)

        # YOUR CODE GOES HERE!


    # Build confusion matrix for second model
    with col2:
        st.header(model2_select)

        # YOUR CODE GOES HERE!


with tab3: 
    # YOUR CODE GOES HERE!
        # Use columns to separate visualizations for models
        # Include plots for local and global explanability!
     
    st.header(model1_select)
    
    st.header(model2_select)

    
