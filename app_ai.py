import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import zscore  # For anomaly detection
import plotly.express as px  # For interactive maps (if you want to add later)
from io import BytesIO  # For report export

# ================= UI CONFIG =================
st.set_page_config(
    page_title="Global Sales Analytics Dashboard",
    layout="wide"
)

st.title("🌍 Global Sales Analytics Dashboard")
st.caption("Filter-Driven Business Intelligence & Forecasting with AI Insights")

# ================= LOAD DATA =================
df = pd.read_csv("sales_data.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# ================= SIDEBAR FILTERS =================
st.sidebar.header("🔎 Filter Options")

# --- Country filter (Select All default) ---
countries = sorted(df["Country"].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=countries
)

# --- Product filter (Select All default) ---
products = sorted(df["Product"].unique())
selected_products = st.sidebar.multiselect(
    "Select Products",
    options=products,
    default=products
)

# --- Date range filter (SAFE) ---
min_date = df["Order_Date"].min()
max_date = df["Order_Date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# --- Defensive date handling ---
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = min_date
    end_date = max_date

# ================= APPLY FILTERS =================
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Product"].isin(selected_products)) &
    (df["Order_Date"] >= pd.to_datetime(start_date)) &
    (df["Order_Date"] <= pd.to_datetime(end_date))
]

# ================= NO DATA SAFETY CHECK =================
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()

# ================= SESSION STATE FOR PERSONALIZATION =================
# Initialize session state for tracking user interactions
if "selected_countries_history" not in st.session_state:
    st.session_state.selected_countries_history = []
if "selected_products_history" not in st.session_state:
    st.session_state.selected_products_history = []

# Update history (simple tracking)
st.session_state.selected_countries_history.append(selected_countries)
st.session_state.selected_products_history.append(selected_products)

# ================= KPI SECTION =================
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
avg_discount = filtered_df["Discount"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
k2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
k3.metric("🏷 Avg Discount", f"{avg_discount:.2f}%")

st.divider()

# =================================================
# 📈 MONTHLY SALES TREND (ADAPTIVE)
# =================================================
st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

if len(monthly_sales) < 2:
    st.info("ℹ️ Not enough months for a trend line. Showing single-month sales.")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.bar(monthly_sales["Month"], monthly_sales["Sales"])
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sales")
    plt.tight_layout()
    st.pyplot(fig1)
else:
    fig1, ax1 = plt.subplots(figsize=(5.5, 3))
    ax1.plot(monthly_sales["Month"], monthly_sales["Sales"], marker="o")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

# =================================================
# 🔮 FORECASTING (ONLY IF POSSIBLE)
# =================================================
st.subheader("🔮 Sales Forecast")

if len(monthly_sales) < 3:
    st.warning("⚠️ Forecasting requires at least 3 months of data.")
else:
    monthly_sales["EMA_Forecast"] = monthly_sales["Sales"].ewm(span=3).mean()
    fig2, ax2 = plt.subplots(figsize=(5.5, 3))
    ax2.plot(monthly_sales["Month"], monthly_sales["Sales"], label="Actual", marker="o")
    ax2.plot(monthly_sales["Month"], monthly_sales["EMA_Forecast"],
             linestyle="--", label="Forecast")
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

# =================================================
# 🚨 AI-POWERED ANOMALY DETECTION AND ALERTS (NEW FEATURE)
# =================================================
st.subheader("🚨 Anomaly Detection & Alerts")

if len(monthly_sales) >= 3:
    # Calculate Z-scores for anomaly detection
    monthly_sales["Z_Score"] = zscore(monthly_sales["Sales"])
    anomalies = monthly_sales[abs(monthly_sales["Z_Score"]) > 2]  # Threshold for anomalies
    
    if not anomalies.empty:
        st.warning(f"⚠️ Anomalies detected in: {', '.join(anomalies['Month'].tolist())}. Possible causes: Market changes or data errors.")
        fig_anom, ax_anom = plt.subplots(figsize=(5.5, 3))
        ax_anom.plot(monthly_sales["Month"], monthly_sales["Sales"], marker="o", label="Sales")
        ax_anom.scatter(anomalies["Month"], anomalies["Sales"], color="red", label="Anomalies", s=100)
        ax_anom.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_anom)
    else:
        st.success("✅ No anomalies detected in sales data.")
else:
    st.info("ℹ️ Anomaly detection requires at least 3 months of data.")

# =================================================
# 🎯 DYNAMIC SCENARIO PLANNING & WHAT-IF ANALYSIS (NEW FEATURE)
# =================================================
st.subheader("🎯 What-If Scenario Planning")

# Sliders for simulation
discount_change = st.slider("Adjust Discount (%)", -50, 50, 0, help="Simulate discount changes")
price_change = st.slider("Adjust Price (%)", -50, 50, 0, help="Simulate price changes")

# Calculate projected sales/profit
original_sales = total_sales
original_profit = total_profit
projected_sales = original_sales * (1 + price_change / 100) * (1 - discount_change / 100)
projected_profit = original_profit * (1 + price_change / 100) * (1 - discount_change / 100)  # Simplified assumption

col1, col2 = st.columns(2)
col1.metric("Projected Sales", f"₹{projected_sales:,.0f}", delta=f"{((projected_sales - original_sales) / original_sales * 100):.1f}%")
col2.metric("Projected Profit", f"₹{projected_profit:,.0f}", delta=f"{((projected_profit - original_profit) / original_profit * 100):.1f}%")

# =================================================
# 📊 PRODUCT-WISE SALES
# =================================================
st.subheader("📊 Product-wise Sales")

product_sales = filtered_df.groupby("Product")["Sales"].sum()

if product_sales.empty:
    st.warning("⚠️ No product sales data available.")
else:
    fig3, ax3 = plt.subplots(figsize=(5.5, 3))
    product_sales.sort_values(ascending=False).plot(kind="bar", ax=ax3)
    ax3.set_xlabel("Product")
    ax3.set_ylabel("Sales")
    plt.tight_layout()
    st.pyplot(fig3)

# =================================================
# 🔥 COUNTRY vs PRODUCT HEATMAP
# =================================================
st.subheader("🔥 Country vs Product Heatmap")

pivot = pd.pivot_table(
    filtered_df,
    values="Sales",
    index="Country",
    columns="Product",
    aggfunc="sum"
)

if pivot.empty:
    st.warning("⚠️ Heatmap not available for selected filters.")
else:
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="coolwarm", ax=ax4)
    plt.tight_layout()
    st.pyplot(fig4)

# =================================================
# 🥧 SALES DISTRIBUTION (PIE CHART)
# =================================================
st.subheader("🥧 Sales Distribution")

pie_choice = st.radio(
    "View Sales Distribution By:",
    ["Country", "Product"],
    horizontal=True
)

if pie_choice == "Country":
    pie_data = filtered_df.groupby("Country")["Sales"].sum()
else:
    pie_data = filtered_df.groupby("Product")["Sales"].sum()

if pie_data.empty:
    st.warning("⚠️ Not enough data for pie chart.")
else:
    fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
    ax_pie.pie(
        pie_data,
        labels=pie_data.index,
        autopct="%1.1f%%",
        startangle=140,
        radius=0.9,
        textprops={"fontsize": 8}
    )
    ax_pie.set_title(
        f"Sales Share by {pie_choice}",
        fontsize=10
    )
    plt.tight_layout()
    st.pyplot(fig_pie)

# =================================================
# 📊 COUNTRY vs PRODUCT (GROUPED BAR CHART)
# =================================================
st.subheader("📊 Country vs Product – Comparative Analysis")

pivot_bar = pd.pivot_table(
    filtered_df,
    values="Sales",
    index="Country",
    columns="Product",
    aggfunc="sum"
)

if pivot_bar.empty:
    st.warning("⚠️ No data available for comparison.")
else:
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    pivot_bar.plot(kind="bar", ax=ax_bar)
    ax_bar.set_xlabel("Country")
    ax_bar.set_ylabel("Sales")
    ax_bar.set_title("Country vs Product Sales Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bar)

# =================================================
# 💡 PERSONALIZED DASHBOARD RECOMMENDATIONS (NEW FEATURE)
# =================================================
st.subheader("💡 Personalized Recommendations")

# Simple logic: Recommend based on history
recent_countries = list(set(st.session_state.selected_countries_history[-1])) if st.session_state.selected_countries_history else []
recent_products = list(set(st.session_state.selected_products_history[-1])) if st.session_state.selected_products_history else []

if recent_countries:
    st.info(f"Based on your recent selections, consider exploring sales trends in {', '.join(recent_countries)} further or comparing with other countries.")
if recent_products:
    st.info(f"You've viewed {', '.join(recent_products)} often—try the heatmap for deeper product-country insights.")

# =================================================
# 📄 AUTOMATED REPORT GENERATION & EXPORT (NEW FEATURE)
# =================================================
st.subheader("📄 Generate & Export Report")

if st.button("Generate PDF Report"):
    # Simple PDF generation (basic text-based for demo; use ReportLab for full charts)
    report_content = f"""
    Global Sales Analytics Report
    =============================
    Countries: {', '.join(selected_countries)}
    Products: {', '.join(selected_products)}
    Period: {start_date} to {end_date}
    
    KPIs:
    - Total Sales: ₹{total_sales:,.0f}
    - Total Profit: ₹{total_profit:,.0f}
    - Avg Discount: {avg_discount:.2f}%
    
    Top Insights:
    - Best Month: {monthly_sales.loc[monthly_sales['Sales'].idxmax(), 'Month'] if not monthly_sales.empty else 'N/A'}
    - Top Product: {product_sales.idxmax() if not product_sales.empty else 'N/A'}
    
    Anomalies: {'Detected' if not anomalies.empty else 'None'}
    """
    
    # Create a downloadable PDF (simplified; for full charts, integrate with libraries like FPDF)
    buffer = BytesIO()
    buffer.write(report_content.encode('utf-8'))
    buffer.seek(0)
    st.download_button(
        label="Download Report",
        data=buffer,
        file_name="sales_report.txt",  # Change to .pdf with proper library
        mime="text/plain"
    )

# =================================================
# 🧾 EXECUTIVE SUMMARY
# =================================================
st.subheader("🧾 Detailed Report")

top_product = product_sales.idxmax() if not product_sales.empty else "N/A"
best_month = monthly_sales.loc[monthly_sales["Sales"].idxmax(), "Month"] if not monthly_sales.empty else "N/A"

st.write(f"""
**Countries:** {', '.join(selected_countries)}  
**Products:** {', '.join(selected_products)}  
**Period:** {start_date} → {end_date}  

📌 **Top Product:** {top_product}  
📌 **Best Month:** {best_month}  
""")

st.success("Analysis Complete! Adjust filters to explore more insights.")