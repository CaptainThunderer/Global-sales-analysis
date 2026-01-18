import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ================= UI CONFIG =================
st.set_page_config(
    page_title="Global Sales Analytics Dashboard",
    layout="wide"
)

st.title("🌍 Global Sales Analytics Dashboard")
st.caption("Filter-Driven Business Intelligence & Forecasting")

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
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie(
        pie_data,
        labels=pie_data.index,
        autopct="%1.1f%%",
        startangle=140
    )
    ax_pie.set_title(f"Sales Share by {pie_choice}")
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
# 📄 FINAL BUSINESS SUMMARY REPORT
# =================================================
st.subheader("📄 Global Sales Analytics Report")

# ---------- SAFE DATE FORMATTING ----------
start_date_str = str(start_date)
end_date_str = str(end_date)

# ---------- TOP CONTRIBUTORS ----------
top_country = filtered_df.groupby("Country")["Sales"].sum().idxmax()
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax()

# ---------- TOP MONTHS ----------
top_months = (
    monthly_sales.sort_values("Sales", ascending=False)
    .head(3)
)

top_months_lines = [
    f"{row['Month']}  →  ₹{row['Sales']:,.0f}"
    for _, row in top_months.iterrows()
]

# ---------- SIMPLE ANOMALY CHECK ----------
mean_sales = monthly_sales["Sales"].mean()
std_sales = monthly_sales["Sales"].std()

anomaly_months = monthly_sales[
    abs(monthly_sales["Sales"] - mean_sales) > 2 * std_sales
]["Month"].tolist()

anomaly_text = ", ".join(anomaly_months) if anomaly_months else "No major anomalies detected"

# ---------- REPORT TEXT ----------
final_report = f"""
GLOBAL SALES ANALYTICS REPORT
=========================================

FILTERS APPLIED
-----------------------------------------
Countries Selected : {", ".join(selected_countries)}
Products Selected  : {", ".join(selected_products)}
Date Range         : {start_date_str} to {end_date_str}

KEY PERFORMANCE INDICATORS
-----------------------------------------
Total Sales        : ₹{total_sales:,.0f}
Total Profit       : ₹{total_profit:,.0f}
Average Discount   : {avg_discount:.2f}%

TOP CONTRIBUTORS
-----------------------------------------
Top Country        : {top_country}
Top Product        : {top_product}

TOP MONTHS BY SALES
-----------------------------------------
{chr(10).join(top_months_lines)}

ANOMALY INSIGHTS
-----------------------------------------
{anomaly_text}

FORECASTING STATUS
-----------------------------------------
{"EMA Forecast Applied" if len(monthly_sales) >= 3 else "Not enough data for forecasting"}

STRATEGIC OBSERVATIONS
-----------------------------------------
- Focus on high-performing products and regions
- Review discount strategy to improve profitability
- Monitor unusual sales fluctuations
- Use trends for demand planning

Report Generated On:
{pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S")}
"""

# ---------- DISPLAY REPORT ----------
st.text(final_report)

# ---------- DOWNLOAD OPTION ----------
st.download_button(
    label="⬇️ Download Report",
    data=final_report,
    file_name="global_sales_report.txt",
    mime="text/plain"
)

