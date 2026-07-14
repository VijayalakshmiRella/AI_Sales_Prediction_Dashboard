import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
st.set_page_config(
    page_title="AI Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)
st.title("📈 AI Sales Prediction Dashboard")

st.write(
    "Predict future sales using Machine Learning."
)
df = pd.read_csv(
    "../processed_data/cleaned_data.csv"
)
model = joblib.load(
    "../models/sales_prediction_model.pkl"
)
scaler = joblib.load(
    "../models/scaler.pkl"
)
st.sidebar.title("Sales Prediction")
quantity = st.sidebar.number_input(
    "Quantity",
    min_value=1,
    value=5
)
discount = st.sidebar.slider(
    "Discount",
    0.0,
    1.0,
    0.1
)
profit = st.sidebar.number_input(
    "Profit",
    value=100.0
)
category = st.sidebar.selectbox(

    "Category",

    df["category"].unique()

)
region = st.sidebar.selectbox(

    "Region",

    df["region"].unique()

)
segment = st.sidebar.selectbox(

    "Segment",

    df["segment"].unique()

)
month = st.sidebar.selectbox(

    "Month",

    sorted(df["month"].unique())

)
year = st.sidebar.selectbox(

    "Year",

    sorted(df["year"].unique())

)
input_df = pd.DataFrame({

    "year":[year],

    "month":[month],

    "quantity":[quantity],

    "discount":[discount],

    "profit":[profit],

    "category":[category],

    "region":[region],

    "segment":[segment]

})
input_df = pd.get_dummies(input_df)
# Create the same features that were used during model training
features = [
    "year",
    "month",
    "quantity",
    "discount",
    "profit",
    "category",
    "region",
    "segment"
]

X = df[features]

# Apply one-hot encoding exactly as during training
X = pd.get_dummies(X, drop_first=True)
input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_scaled = scaler.transform(
    input_df
)
prediction = model.predict(
    input_scaled
)
st.subheader("Predicted Sales")

st.success(

    f"${prediction[0]:,.2f}"

)
monthly = df.groupby(
    "month"
)["sales"].sum().reset_index()

fig = px.bar(

    monthly,

    x="month",

    y="sales",

    title="Monthly Sales"

)

st.plotly_chart(fig)
category_sales = df.groupby(
    "category"
)["sales"].sum().reset_index()

fig = px.pie(

    category_sales,

    names="category",

    values="sales"

)

st.plotly_chart(fig)
region_sales = df.groupby(
    "region"
)["sales"].sum().reset_index()

fig = px.bar(

    region_sales,

    x="region",

    y="sales"

)

st.plotly_chart(fig)
year_sales = df.groupby(
    "year"
)["sales"].sum().reset_index()

fig = px.line(

    year_sales,

    x="year",

    y="sales",

    markers=True

)

st.plotly_chart(fig)
st.subheader(
    "Dataset Preview"
)

st.dataframe(df.head(20))
st.markdown("---")
col1, col2 = st.columns(2)
st.metric("Predicted Sales", prediction)
with st.expander("Dataset"):
    st.dataframe(df.head())
st.metric(
    "Average Sales",
    round(df["sales"].mean(),2)
)
csv = input_df.to_csv(index=False)

st.download_button(

    "Download Prediction",

    csv,

    "prediction.csv"

)
import os
from datetime import datetime

history_file = "prediction_history.csv"

history = pd.DataFrame({
    "Date": [datetime.now()],
    "Year": [year],
    "Month": [month],
    "Quantity": [quantity],
    "Discount": [discount],
    "Profit": [profit],
    "Category": [category],
    "Region": [region],
    "Segment": [segment],
    "Predicted Sales": [prediction[0]]
})

if os.path.exists(history_file):
    old_history = pd.read_csv(history_file)
    history = pd.concat([old_history, history], ignore_index=True)

history.to_csv(history_file, index=False)

st.success("Prediction saved successfully!")

st.subheader("Prediction History")

st.dataframe(history)

