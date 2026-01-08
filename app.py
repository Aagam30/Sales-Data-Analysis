# =====================================
# Sales Data Analysis Dashboard
# Dark/Light Theme + ML Forecasting
# =====================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------
# THEME TOGGLE (UI)
# -------------------------------------
st.sidebar.header("🎨 Appearance")

theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Data Analysis Dashboard")
st.markdown("Interactive sales analytics with forecasting")

# -------------------------------------
# DATA SOURCE
# -------------------------------------
st.sidebar.header("📁 Data Source")

data_mode = st.sidebar.radio(
    "Choose data source:",
    ["Use default dataset", "Upload your own file"]
)

if data_mode == "Upload your own file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"]
    )

    if uploaded_file is None:
        st.info("👈 Upload a file to start")
        st.stop()

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

else:
    if not os.path.exists("sales_data.csv"):
        st.error("❌ sales_data.csv not found")
        st.stop()
    df = pd.read_csv("sales_data.csv")
    st.success("✅ Using default dataset")

# -------------------------------------
# DATA PREPARATION
# -------------------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["Revenue"] = df["Quantity"] * df["Price"]
df["Profit"] = (df["Price"] - df["Cost"]) * df["Quantity"]
df["Month"] = df["Date"].dt.to_period("M")

# -------------------------------------
# FILTERS
# -------------------------------------
st.sidebar.header("🔍 Filters")

category_filter = st.sidebar.multiselect(
    "Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[df["Category"].isin(category_filter)]

# -------------------------------------
# KPIs
# -------------------------------------
st.subheader("📌 Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("💰 Revenue", f"₹{filtered_df['Revenue'].sum():,.0f}")
c2.metric("📈 Profit", f"₹{filtered_df['Profit'].sum():,.0f}")
c3.metric("📦 Orders", len(filtered_df))

# -------------------------------------
# MONTHLY SALES TREND
# -------------------------------------
st.subheader("📈 Monthly Revenue Trend")

monthly_sales = filtered_df.groupby("Month")["Revenue"].sum()

fig, ax = plt.subplots()
ax.plot(monthly_sales.index.astype(str), monthly_sales.values, marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.set_title("Monthly Revenue")
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------------------
# ML FORECASTING (LINEAR REGRESSION)
# -------------------------------------
st.subheader("🤖 Sales Forecast (Next 3 Months)")

# Prepare data
X = np.arange(len(monthly_sales))
y = monthly_sales.values

# Train model (simple linear regression)
coef = np.polyfit(X, y, 1)

# Predict future
future_X = np.arange(len(X), len(X) + 3)
future_y = coef[0] * future_X + coef[1]

future_months = pd.period_range(
    start=monthly_sales.index[-1] + 1,
    periods=3,
    freq="M"
)

forecast_df = pd.DataFrame({
    "Month": future_months.astype(str),
    "Forecasted Revenue": future_y.astype(int)
})

st.dataframe(forecast_df)

# Forecast Plot
fig2, ax2 = plt.subplots()
ax2.plot(monthly_sales.index.astype(str), y, label="Actual", marker="o")
ax2.plot(future_months.astype(str), future_y, label="Forecast", marker="x")
ax2.set_title("Actual vs Forecasted Revenue")
ax2.legend()
plt.xticks(rotation=45)
st.pyplot(fig2)

# -------------------------------------
# PRODUCT PROFIT ANALYSIS
# -------------------------------------
st.subheader("📦 Product Profit / Loss")

product_profit = filtered_df.groupby("Product")["Profit"].sum()

fig3, ax3 = plt.subplots()
ax3.bar(product_profit.index, product_profit.values)
ax3.axhline(0)
ax3.set_ylabel("Profit")
plt.xticks(rotation=45)
st.pyplot(fig3)

# -------------------------------------
# DOWNLOAD DATA
# -------------------------------------
st.subheader("⬇ Download Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "filtered_sales_data.csv", "text/csv")

# -------------------------------------
# FOOTER
# -------------------------------------
st.markdown("---")
st.markdown("✅ Built with Python, Pandas, NumPy, Matplotlib & Streamlit")
