# =====================================
# Sales Data Analysis Dashboard
# Light/Dark Theme + Forecasting + Accuracy
# =====================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------
# THEME HANDLER
# -------------------------------------
def apply_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp { background-color: #0e1117; color: white; }
            section[data-testid="stSidebar"] { background-color: #161b22; }
            div[data-testid="metric-container"] {
                background-color: #1f2933;
                padding: 15px;
                border-radius: 10px;
                color: white;
            }
            .stButton>button {
                background-color: #238636;
                color: white;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: white; color: black; }
            section[data-testid="stSidebar"] { background-color: #f0f2f6; }
            div[data-testid="metric-container"] {
                background-color: #ffffff;
                padding: 15px;
                border-radius: 10px;
                color: black;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------
# SIDEBAR: THEME
# -------------------------------------
st.sidebar.header("🎨 Appearance")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
apply_theme(theme)

# -------------------------------------
# TITLE
# -------------------------------------
st.title("📊 Sales Data Analysis Dashboard")
st.markdown("Interactive sales analytics with forecasting and accuracy metrics")

# -------------------------------------
# SIDEBAR: DATA SOURCE
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
        st.info("👈 Upload a file to start analysis.")
        st.stop()

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

else:
    if not os.path.exists("sales_data.csv"):
        st.error("❌ sales_data.csv not found in repository.")
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
st.subheader("📌 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Revenue", f"₹{filtered_df['Revenue'].sum():,.0f}")
c2.metric("📈 Profit", f"₹{filtered_df['Profit'].sum():,.0f}")
c3.metric("📦 Orders", len(filtered_df))
c4.metric("📊 Avg Order Value", f"₹{filtered_df['Revenue'].mean():,.0f}")

# -------------------------------------
# MONTHLY SALES
# -------------------------------------
st.subheader("📈 Monthly Revenue Trend")

monthly_sales = filtered_df.groupby("Month")["Revenue"].sum()

fig, ax = plt.subplots()
ax.plot(monthly_sales.index.astype(str), monthly_sales.values, marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.set_title("Monthly Revenue Trend")
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------------------
# FORECASTING (LINEAR REGRESSION)
# -------------------------------------
st.subheader("🤖 Sales Forecast (Next 3 Months)")

X = np.arange(len(monthly_sales))
y = monthly_sales.values

coef = np.polyfit(X, y, 1)

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
# FORECAST ACCURACY METRICS
# -------------------------------------
st.subheader("📏 Forecast Accuracy Metrics")

y_pred = coef[0] * X + coef[1]

mae = np.mean(np.abs(y - y_pred))
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
mape = np.mean(np.abs((y - y_pred) / y)) * 100

m1, m2, m3 = st.columns(3)
m1.metric("MAE (₹)", f"{mae:,.0f}")
m2.metric("RMSE (₹)", f"{rmse:,.0f}")
m3.metric("MAPE (%)", f"{mape:.2f}%")

# -------------------------------------
# PRODUCT PROFIT
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
# LOSS TRANSACTIONS
# -------------------------------------
st.subheader("⚠ Loss-Making Transactions")

loss_df = filtered_df[filtered_df["Profit"] < 0]

if loss_df.empty:
    st.success("✅ No loss-making transactions")
else:
    st.warning("❌ Loss-making transactions detected")
    st.dataframe(loss_df)

# -------------------------------------
# DOWNLOAD DATA
# -------------------------------------
st.subheader("⬇ Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "filtered_sales_data.csv", "text/csv")

# -------------------------------------
# FOOTER
# -------------------------------------
st.markdown("---")
st.markdown("✅ Built with **Python, Pandas, NumPy, Matplotlib & Streamlit**")
