import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ================= UI CONFIG =================
st.set_page_config(page_title="Global Sales Analytics", layout="wide")
st.title("🌍 Global Sales Analytics Dashboard")
st.caption("Filter-driven Business Intelligence & Forecasting")

# ================= LOAD DATA =================
df = pd.read_csv("sales_data.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# ================= SIDEBAR FILTERS =================
st.sidebar.header("🔎 Filters")

# ---- DATE RANGE FILTER ----
min_date = df["Order_Date"].min()
max_date = df["Order_Date"].max()

date_range = st.sidebar.date_input(
    "Select Time Duration",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# ---- PRODUCT FILTER (SELECT ALL) ----
products = df["Product"].unique().tolist()
selected_products = st.sidebar.multiselect(
    "Select Products",
    options=["All"] + products,
    default=["All"]
)

# ---- COUNTRY FILTER (SELECT ALL) ----
countries = df["Country"].unique().tolist()
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=["All"] + countries,
    default=["All"]
)

# ================= APPLY FILTERS =================
filtered_df = df.copy()

# Date filter
filtered_df = filtered_df[
    (filtered_df["Order_Date"] >= pd.to_datetime(date_range[0])) &
    (filtered_df["Order_Date"] <= pd.to_datetime(date_range[1]))
]

# Product filter
if "All" not in selected_products:
    filtered_df = filtered_df[filtered_df["Product"].isin(selected_products)]

# Country filter
if "All" not in selected_countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

# ================= KPI SECTION =================
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
avg_discount = filtered_df["Discount"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("💰 Sales", f"₹{total_sales:,.0f}")
k2.metric("📈 Profit", f"₹{total_profit:,.0f}")
k3.metric("🏷 Avg Discount", f"{avg_discount:.2f}%")

st.divider()

# =================================================
# 📈 ITEM-WISE MONTHLY SALES (CLEAN)
# =================================================
st.subheader("📊 Item-wise Monthly Sales Trend")

monthly_item_sales = (
    filtered_df
    .groupby(["Month", "Product"])["Sales"]
    .sum()
    .reset_index()
)

fig1, ax1 = plt.subplots(figsize=(5.5, 3))
sns.lineplot(
    data=monthly_item_sales,
    x="Month",
    y="Sales",
    hue="Product",
    marker="o",
    ax=ax1
)
ax1.set_title("Monthly Sales by Item")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# =================================================
# 🔮 FORECASTING (EMA BASED ON FILTERS)
# =================================================
st.subheader("🔮 Sales Forecast (Based on Selected Filters)")

monthly_total = (
    filtered_df
    .groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

monthly_total["Forecast"] = monthly_total["Sales"].ewm(span=3).mean()

fig2, ax2 = plt.subplots(figsize=(5.5, 3))
ax2.plot(monthly_total["Month"], monthly_total["Sales"], marker="o", label="Actual")
ax2.plot(monthly_total["Month"], monthly_total["Forecast"],
         linestyle="--", label="Forecast")
ax2.set_title("Filtered Sales Forecast")
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# =================================================
# 🌍 GLOBAL PRODUCT SALES (CLEAN BAR)
# =================================================
st.subheader("🌍 Global Product Sales")

global_product_sales = (
    filtered_df
    .groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

fig3, ax3 = plt.subplots(figsize=(5, 3))
global_product_sales.plot(kind="bar", ax=ax3)
ax3.set_title("Product-wise Global Sales")
ax3.set_ylabel("Sales")
plt.tight_layout()
st.pyplot(fig3)

# =================================================
# 🔥 COUNTRY vs PRODUCT HEATMAP (FILTER AWARE)
# =================================================
st.subheader("🔥 Country vs Product Sales Heatmap")

pivot = pd.pivot_table(
    filtered_df,
    values="Sales",
    index="Country",
    columns="Product",
    aggfunc="sum"
)

fig4, ax4 = plt.subplots(figsize=(6, 3.5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="coolwarm", ax=ax4)
plt.tight_layout()
st.pyplot(fig4)

# =================================================
# 🧾 EXECUTIVE SUMMARY
# =================================================
st.subheader("🧾 Executive Summary")

if not filtered_df.empty:
    best_product = global_product_sales.idxmax()
    best_month = monthly_total.loc[monthly_total["Sales"].idxmax(), "Month"]

    st.write(f"""
    **Selected Duration:** {date_range[0]} to {date_range[1]}  
    **Products Considered:** {', '.join(selected_products)}  
    **Countries Considered:** {', '.join(selected_countries)}  

    🔹 **Top Product:** {best_product}  
    🔹 **Peak Sales Month:** {best_month}  
    🔹 **Total Revenue:** ₹{total_sales:,.0f}  

    📌 *Insights dynamically generated based on applied filters.*
    """)
else:
    st.warning("No data available for the selected filters.")
