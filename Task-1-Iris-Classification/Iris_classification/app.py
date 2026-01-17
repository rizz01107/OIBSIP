import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# 1. Load Data and Train Model (Internal)
@st.cache_data # This keeps the model in memory so it doesn't retrain on every click
def train_model():
    df = pd.read_csv("Iris.csv")
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    X = df.drop('Species', axis=1)
    y = df['Species']
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = train_model()

# 2. Streamlit UI Setup
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Classification")
st.write("Enter the measurements below to predict the Iris species.")

# 3. Create Input Fields for User
st.sidebar.header("Input Features")
sepal_l = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_w = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_l = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_w = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# 4. Predict Button
if st.button("Predict Species"):
    # Prepare input data
    input_data = pd.DataFrame([[sepal_l, sepal_w, petal_l, petal_w]], columns=feature_names)
    
    # Get Prediction
    prediction = model.predict(input_data)
    
    # Display Result
    st.success(f"The predicted species is: **{prediction[0]}**")
    
    # Display an image based on prediction (Optional)
    st.info("Check the petal and sepal sizes in the sidebar to see how the prediction changes!")