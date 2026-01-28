import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ================= UI CONFIG =================
st.set_page_config(
    page_title="Global Sales Analytics Dashboard",
    layout="wide"
)

st.title("🌍 Global Sales Analytics Dashboard")
st.caption("Filter-Driven Business Intelligence & Forecasting")

st.markdown("""
### 📌 Project Aim
This dashboard allows users to upload their own sales dataset and analyze:
- Sales & profit trends
- Country & product performance
- Forecasting and anomaly insights
""")

# ================= LOAD DATA (USER INPUT) =================
uploaded_file = st.file_uploader(
    "📤 Upload Sales Data (CSV only)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ---- Mandatory column check ----
required_cols = {"Order_Date", "Country", "Product", "Sales", "Profit", "Discount"}
if not required_cols.issubset(df.columns):
    st.error("❌ CSV must contain Order_Date, Country, Product, Sales, Profit, Discount columns.")
    st.stop()

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# ================= SIDEBAR FILTERS =================
st.sidebar.header("🔎 Filter Options")

countries = sorted(df["Country"].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=countries
)

products = sorted(df["Product"].unique())
selected_products = st.sidebar.multiselect(
    "Select Products",
    options=products,
    default=products
)

min_date = df["Order_Date"].min()
max_date = df["Order_Date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

start_date, end_date = date_range

# ================= APPLY FILTERS =================
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Product"].isin(selected_products)) &
    (df["Order_Date"] >= pd.to_datetime(start_date)) &
    (df["Order_Date"] <= pd.to_datetime(end_date))
]

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
# 📈 MONTHLY SALES TREND (INTERACTIVE)
# =================================================
st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby("Month", as_index=False)["Sales"]
    .sum()
)

fig1 = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend",
    hover_data={"Sales": ":,.0f"}
)
st.plotly_chart(fig1, use_container_width=True)

# =================================================
# 🔮 FORECASTING (EMA)
# =================================================
st.subheader("🔮 Sales Forecast")

if len(monthly_sales) < 3:
    st.warning("⚠️ Forecasting requires at least 3 months of data.")
else:
    monthly_sales["EMA_Forecast"] = monthly_sales["Sales"].ewm(span=3).mean()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=monthly_sales["Month"],
        y=monthly_sales["Sales"],
        mode="lines+markers",
        name="Actual"
    ))
    fig2.add_trace(go.Scatter(
        x=monthly_sales["Month"],
        y=monthly_sales["EMA_Forecast"],
        mode="lines",
        name="Forecast",
        line=dict(dash="dash")
    ))

    fig2.update_layout(title="Sales Forecast (EMA)")
    st.plotly_chart(fig2, use_container_width=True)

# =================================================
# 📊 PRODUCT-WISE SALES
# =================================================
st.subheader("📊 Product-wise Sales")

product_sales = (
    filtered_df
    .groupby("Product", as_index=False)["Sales"]
    .sum()
)

fig3 = px.bar(
    product_sales,
    x="Product",
    y="Sales",
    title="Product-wise Sales",
    text_auto=".2s",
    hover_data={"Sales": ":,.0f"}
)
st.plotly_chart(fig3, use_container_width=True)

# =================================================
# 🔥 COUNTRY vs PRODUCT HEATMAP (IMPROVED)
# =================================================
st.subheader("🔥 Country vs Product Heatmap")

heatmap_type = st.radio(
    "Heatmap View",
    ["Sales", "Profit", "Normalized Sales (%)"],
    horizontal=True
)

if heatmap_type == "Profit":
    pivot = pd.pivot_table(
        filtered_df,
        values="Profit",
        index="Country",
        columns="Product",
        aggfunc="sum"
    )
elif heatmap_type == "Normalized Sales (%)":
    pivot = pd.pivot_table(
        filtered_df,
        values="Sales",
        index="Country",
        columns="Product",
        aggfunc="sum"
    )
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
else:
    pivot = pd.pivot_table(
        filtered_df,
        values="Sales",
        index="Country",
        columns="Product",
        aggfunc="sum"
    )

fig4 = px.imshow(
    pivot,
    text_auto=".1f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title=f"{heatmap_type} Heatmap"
)
st.plotly_chart(fig4, use_container_width=True)

# =================================================
# 🥧 SALES DISTRIBUTION (INTERACTIVE PIE)
# =================================================
st.subheader("🥧 Sales Distribution")

pie_choice = st.radio(
    "View Sales Distribution By:",
    ["Country", "Product"],
    horizontal=True
)

pie_data = (
    filtered_df
    .groupby(pie_choice, as_index=False)["Sales"]
    .sum()
)

fig_pie = px.pie(
    pie_data,
    values="Sales",
    names=pie_choice,
    hole=0.4,
    title=f"Sales Share by {pie_choice}"
)
st.plotly_chart(fig_pie)

# =================================================
# 📊 COUNTRY vs PRODUCT (GROUPED BAR)
# =================================================
st.subheader("📊 Country vs Product – Comparative Analysis")

pivot_bar = pd.pivot_table(
    filtered_df,
    values="Sales",
    index="Country",
    columns="Product",
    aggfunc="sum"
).reset_index()

fig_bar = px.bar(
    pivot_bar,
    x="Country",
    y=pivot_bar.columns[1:],
    barmode="group",
    title="Country vs Product Sales Comparison"
)
st.plotly_chart(fig_bar, use_container_width=True)

# =================================================
# 📄 FINAL BUSINESS SUMMARY REPORT
# =================================================
st.subheader("📄 Global Sales Analytics Report")

top_country = filtered_df.groupby("Country")["Sales"].sum().idxmax()
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax()

top_months = monthly_sales.sort_values("Sales", ascending=False).head(3)

top_months_lines = [
    f"{row['Month']} → ₹{row['Sales']:,.0f}"
    for _, row in top_months.iterrows()
]

mean_sales = monthly_sales["Sales"].mean()
std_sales = monthly_sales["Sales"].std()

anomaly_months = monthly_sales[
    abs(monthly_sales["Sales"] - mean_sales) > 2 * std_sales
]["Month"].tolist()

anomaly_text = ", ".join(anomaly_months) if anomaly_months else "No major anomalies detected"

final_report = f"""
GLOBAL SALES ANALYTICS REPORT
=========================================

FILTERS APPLIED
-----------------------------------------
Countries Selected : {", ".join(selected_countries)}
Products Selected  : {", ".join(selected_products)}
Date Range         : {start_date} to {end_date}

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

Report Generated On:
{pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S")}
"""

st.text(final_report)

st.download_button(
    label="⬇️ Download Report",
    data=final_report,
    file_name="global_sales_report.txt",
    mime="text/plain"
)
