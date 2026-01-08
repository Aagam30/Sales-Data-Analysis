# =====================================
# Sales Data Analysis Dashboard (Advanced)
# Features:
# 1. Login Authentication
# 2. Default CSV / Upload Support
# 3. Sales Forecasting
# 4. PDF Report Export
# =====================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------
# SIMPLE AUTHENTICATION
# -------------------------------------
USERS = {
    "admin": "admin123",
    "user": "user123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login to Sales Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

    st.stop()

# -------------------------------------
# MAIN APP
# -------------------------------------
st.title("📊 Sales Data Analysis Dashboard")
st.markdown("Advanced analytics with forecasting & reporting")

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
        st.info("👈 Upload a file to continue")
        st.stop()

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

else:
    if not os.path.exists("sales_data.csv"):
        st.error("❌ sales_data.csv not found in repository")
        st.stop()

    df = pd.read_csv("sales_data.csv")
    st.success("✅ Using default sales_data.csv")

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

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"₹{filtered_df['Revenue'].sum():,.0f}")
col2.metric("📈 Total Profit", f"₹{filtered_df['Profit'].sum():,.0f}")
col3.metric("📦 Total Orders", len(filtered_df))

# -------------------------------------
# SALES TREND
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
# SALES FORECASTING (SIMPLE)
# -------------------------------------
st.subheader("🔮 Sales Forecast (Next 3 Months)")

x = np.arange(len(monthly_sales))
y = monthly_sales.values

coef = np.polyfit(x, y, 1)
forecast_x = np.arange(len(x), len(x) + 3)
forecast_y = coef[0] * forecast_x + coef[1]

forecast_months = pd.period_range(
    start=monthly_sales.index[-1] + 1,
    periods=3,
    freq="M"
)

forecast_df = pd.DataFrame({
    "Month": forecast_months.astype(str),
    "Forecasted Revenue": forecast_y.astype(int)
})

st.dataframe(forecast_df)

# -------------------------------------
# PROFIT / LOSS CHART
# -------------------------------------
st.subheader("⚠ Profit vs Loss by Product")

product_profit = filtered_df.groupby("Product")["Profit"].sum()

fig2, ax2 = plt.subplots()
ax2.bar(product_profit.index, product_profit.values)
ax2.axhline(0)
ax2.set_ylabel("Profit")
ax2.set_title("Profit / Loss by Product")
plt.xticks(rotation=45)
st.pyplot(fig2)

# -------------------------------------
# PDF REPORT GENERATION
# -------------------------------------
st.subheader("📄 Export PDF Report")

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Sales Analysis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, f"Total Revenue: ₹{filtered_df['Revenue'].sum():,.0f}", ln=True)
    pdf.cell(200, 10, f"Total Profit: ₹{filtered_df['Profit'].sum():,.0f}", ln=True)
    pdf.cell(200, 10, f"Total Orders: {len(filtered_df)}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, "Top Products by Profit:", ln=True)

    top_products = (
        filtered_df.groupby("Product")["Profit"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    for product, profit in top_products.items():
        pdf.cell(200, 10, f"{product}: ₹{profit:,.0f}", ln=True)

    file_path = "sales_report.pdf"
    pdf.output(file_path)
    return file_path

if st.button("📥 Generate PDF Report"):
    pdf_file = generate_pdf()
    with open(pdf_file, "rb") as f:
        st.download_button(
            "⬇ Download PDF",
            f,
            file_name="sales_report.pdf",
            mime="application/pdf"
        )

# -------------------------------------
# LOGOUT
# -------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------------------------
# FOOTER
# -------------------------------------
st.markdown("---")
st.markdown("✅ Built with **Python, Pandas, NumPy, Matplotlib & Streamlit**")
