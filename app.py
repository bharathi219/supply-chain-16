import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load trained model and scaler
sc = joblib.load("sc.pkl")  # scaler file
model = joblib.load("log_multi.pkl")  # trained logistic regression model


# UI Code
st.header("Product Estimation for the Given Features.")

# Displaying an image in the center column
p1, p2, p3 = st.columns(3)
with p2:
    st.image("PIC.jpg")

st.write("This app is built on the below features to estimate Order Priority.")

# Loading the dataset (just for display purposes)
userinpdata = pd.read_excel(r"C:\Users\mekal\OneDrive\Desktop\Supply_Chain_Final_Data.xlsx")
st.dataframe(userinpdata.head(5))

st.subheader("Enter Product Details to Estimate Order Priority:")

# Collect user inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    Lead_Time = st.number_input("Lead_Time:")
with col2:
    Demand_Forecast = st.number_input("Demand_Forecast:")
with col3:
    Inventory_Level = st.number_input("Inventory_Level:")
with col4:
    Stockout_Flag = st.number_input("Stockout_Flag:")

col5, col6, col7, col8 = st.columns(4)
with col5:
    Backorder_Flag = st.number_input("Backorder_Flag:")
with col6:
    Order_Quantity = st.number_input("Order_Quantity:")
with col7:
    Shipment_Quantity = st.number_input("Shipment_Quantity No:")
with col8:
    Product_Price = st.number_input("Product_Price:")

# Logic to handle prediction
if st.button("Estimate Order Priority"):
    # Create a DataFrame from user input
    row = pd.DataFrame([[Lead_Time, Demand_Forecast, Inventory_Level, Stockout_Flag, Backorder_Flag, Order_Quantity, Shipment_Quantity, Product_Price]], columns=userinpdata.columns)
    st.write("Given Input Data:")
    st.dataframe(row)
    
    # Feature scaling
    try:
        scaled_row = sc.transform(row)
    except Exception as e:
        st.error(f"Error in scaling input: {e}")
        scaled_row = row  # If scaling fails, use unscaled data

    # Model prediction
    try:
        prob0 = round(model.predict_proba(scaled_row)[0][0], 2)
        prob1 = round(model.predict_proba(scaled_row)[0][1], 2)
        prob2 = round(model.predict_proba(scaled_row)[0][2], 2)
        st.write(f"Predicted Probabilities: low - {prob0}, medium - {prob1}, high - {prob2}")

        # Get final prediction
        out = model.predict(scaled_row)[0]
        st.write(f"Prediction: {out}")
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
