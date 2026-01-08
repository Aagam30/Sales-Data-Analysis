# ============================
# Sales Data Analysis App
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# App Title
# ----------------------------
st.set_page_config(page_title="Sales Analysis App", layout="wide")
st.title("📊 Sales Data Analysis App")

# ----------------------------
# Load Data
# ----------------------------
st.sidebar.header("📁 Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sales Data (CSV or XLSX)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    # Read CSV or Excel
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.warning("No file uploaded. Using default sales_data.csv")
    df = pd.read_csv("sales_data.csv")

# ----------------------------
# Data Cleaning & Preparation
# ----------------------------
df["Date"] = pd.to_datetime(df["Date"])

# Calculations
df["Revenue"] = df["Quantity"] * df["Price"]
df["Profit"] = (df["Price"] - df["Cost"]) * df["Quantity"]
df["Month"] = df["Date"].dt.month

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filters")

selected_category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

selected_product = st.sidebar.multiselect(
    "Select Product",
    options=df["Product"].unique(),
    default=df["Product"].unique()
)

filtered_df = df[
    (df["Category"].isin(selected_category)) &
    (df["Product"].isin(selected_product))
]

# ----------------------------
# KPIs (NumPy)
# ----------------------------
total_revenue = np.sum(filtered_df["Revenue"])
total_profit = np.sum(filtered_df["Profit"])
avg_profit = np.mean(filtered_df["Profit"])

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
col2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
col3.metric("📊 Avg Profit", f"₹{avg_profit:,.0f}")

# ----------------------------
# Display Data
# ----------------------------
st.subheader("📄 Sales Data")
st.dataframe(filtered_df)

# ----------------------------
# Monthly Sales Trend (Matplotlib)
# ----------------------------
st.subheader("📅 Monthly Sales Trend")

monthly_sales = filtered_df.groupby("Month")["Revenue"].sum()

fig1, ax1 = plt.subplots()
ax1.plot(monthly_sales.index, monthly_sales.values, marker="o")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue")
ax1.set_title("Monthly Revenue Trend")
st.pyplot(fig1)

# ----------------------------
# Product-wise Revenue
# ----------------------------
st.subheader("📦 Product-wise Revenue")

product_sales = filtered_df.groupby("Product")["Revenue"].sum()

fig2, ax2 = plt.subplots()
ax2.bar(product_sales.index, product_sales.values)
ax2.set_xlabel("Product")
ax2.set_ylabel("Revenue")
ax2.set_title("Revenue by Product")
plt.xticks(rotation=45)
st.pyplot(fig2)

# ----------------------------
# Category-wise Revenue (Pie Chart)
# ----------------------------
st.subheader("🏷 Category-wise Revenue")

category_sales = filtered_df.groupby("Category")["Revenue"].sum()

fig3, ax3 = plt.subplots()
ax3.pie(category_sales.values, labels=category_sales.index, autopct="%1.1f%%")
ax3.set_title("Revenue Distribution by Category")
st.pyplot(fig3)

# ----------------------------
# Profit vs Loss Analysis
# ----------------------------
st.subheader("⚠ Profit & Loss Analysis")

loss_data = filtered_df[filtered_df["Profit"] < 0]

if not loss_data.empty:
    st.warning("Loss-making Transactions Detected")
    st.dataframe(loss_data)
else:
    st.success("No loss-making transactions found")

# ----------------------------
# Best & Worst Product
# ----------------------------
product_profit = filtered_df.groupby("Product")["Profit"].sum()

best_product = product_profit.idxmax()
worst_product = product_profit.idxmin()

st.subheader("🏆 Product Performance")
st.write(f"✅ **Best Product (Highest Profit):** {best_product}")
st.write(f"❌ **Worst Product (Lowest Profit):** {worst_product}")

# ----------------------------
# Download Cleaned Data
# ----------------------------
st.subheader("⬇ Download Processed Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="processed_sales_data.csv",
    mime="text/csv",
)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("✅ Built using **Python, Pandas, NumPy, Matplotlib & Streamlit**")
