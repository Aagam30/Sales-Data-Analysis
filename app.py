# ===============================
# Advanced Sales Data Analysis App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Data Analysis Dashboard")
st.markdown("Analyze sales performance, profit & trends interactively")

# -------------------------------
# Sidebar – File Upload
# -------------------------------
st.sidebar.header("📁 Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👈 Please upload a CSV or Excel file to start analysis.")
    st.stop()

# -------------------------------
# Load Data
# -------------------------------
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# -------------------------------
# Data Preparation
# -------------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["Revenue"] = df["Quantity"] * df["Price"]
df["Profit"] = (df["Price"] - df["Cost"]) * df["Quantity"]
df["Month"] = df["Date"].dt.month_name()

# -------------------------------
# Sidebar – Filters
# -------------------------------
st.sidebar.header("🔍 Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Date"].min(), df["Date"].max()]
)

category_filter = st.sidebar.multiselect(
    "Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

product_filter = st.sidebar.multiselect(
    "Product",
    options=df["Product"].unique(),
    default=df["Product"].unique()
)

filtered_df = df[
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1])) &
    (df["Category"].isin(category_filter)) &
    (df["Product"].isin(product_filter))
]

# -------------------------------
# KPI Section
# -------------------------------
st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Revenue", f"₹{np.sum(filtered_df['Revenue']):,.0f}")
col2.metric("📈 Total Profit", f"₹{np.sum(filtered_df['Profit']):,.0f}")
col3.metric("📦 Total Orders", len(filtered_df))
col4.metric("📊 Avg Order Value", f"₹{np.mean(filtered_df['Revenue']):,.0f}")

# -------------------------------
# Charts Row 1
# -------------------------------
st.subheader("📈 Sales Trends")

col5, col6 = st.columns(2)

with col5:
    monthly_sales = filtered_df.groupby("Month")["Revenue"].sum()
    fig, ax = plt.subplots()
    ax.plot(monthly_sales.index, monthly_sales.values, marker="o")
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col6:
    category_sales = filtered_df.groupby("Category")["Revenue"].sum()
    fig, ax = plt.subplots()
    ax.pie(category_sales.values, labels=category_sales.index, autopct="%1.1f%%")
    ax.set_title("Revenue by Category")
    st.pyplot(fig)

# -------------------------------
# Charts Row 2
# -------------------------------
st.subheader("📦 Product Performance")

col7, col8 = st.columns(2)

with col7:
    product_revenue = filtered_df.groupby("Product")["Revenue"].sum()
    fig, ax = plt.subplots()
    ax.bar(product_revenue.index, product_revenue.values)
    ax.set_title("Revenue by Product")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col8:
    profit_data = filtered_df.groupby("Product")["Profit"].sum()
    fig, ax = plt.subplots()
    ax.bar(profit_data.index, profit_data.values)
    ax.axhline(0)
    ax.set_title("Profit / Loss by Product")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------
# Top Products Section
# -------------------------------
st.subheader("🏆 Top 5 Products by Profit")

top_products = (
    filtered_df.groupby("Product")["Profit"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

st.dataframe(top_products)

# -------------------------------
# Loss Detection
# -------------------------------
st.subheader("⚠ Loss-Making Transactions")

loss_df = filtered_df[filtered_df["Profit"] < 0]

if loss_df.empty:
    st.success("✅ No loss-making transactions found")
else:
    st.warning("❌ Loss-making transactions detected")
    st.dataframe(loss_df)

# -------------------------------
# Raw Data View
# -------------------------------
with st.expander("📄 View Filtered Raw Data"):
    st.dataframe(filtered_df)

# -------------------------------
# Download Button
# -------------------------------
st.subheader("⬇ Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    csv,
    "filtered_sales_data.csv",
    "text/csv"
)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "✅ Built with **Python, Pandas, NumPy, Matplotlib & Streamlit**"
)
