# Streamlit ilovasi
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Modelni yuklash
with open("airmodel2.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit sarlavhasi
st.title("Linear Regression Model for Dataset")
st.write("100,000 qatorlik random dataset uchun bashorat qiluvchi ilova")

# Datasetni yuklash
dataset_file = "random_dataset.csv"
try:
    df = pd.read_csv(dataset_file)
except FileNotFoundError:
    st.error("Datasetni yaratish uchun kodni qayta ishlating.")
else:
    st.write("Datasetdan namunaviy ma'lumotlar:")
    st.dataframe(df.head(20))

# Foydalanuvchi kiritadigan qiymatlar
st.header("Bashorat uchun ma'lumot kiritish")
indicator_id = st.number_input("Indicator ID", min_value=1, max_value=1000, value=640)
geo_join_id = st.slider("Geo Join ID", min_value=100.0, max_value=500.0, value=409.0)
data_value = st.slider("Data Value", min_value=0.1, max_value=20.0, value=0.3)

# Bashorat
if st.button("Bashorat qilish"):
    input_data = np.array([[indicator_id, geo_join_id, data_value]])
    prediction = model.predict(input_data)
    st.success(f"Bashorat qilingan qiymat: {prediction[0]:.2f}")
