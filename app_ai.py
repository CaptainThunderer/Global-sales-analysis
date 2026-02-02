import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error

# ======================================================
# ⚙️ UI CONFIG
# ======================================================
st.set_page_config(
    page_title="Global Sales Analytics Dashboard",
    layout="wide"
)

st.title("🌍 Global Sales Analytics Dashboard")
st.caption("AI-Powered Business Intelligence, Forecasting & Scenario Planning")

# ======================================================
# 📤 DATA UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "📤 Upload Sales CSV (Order_Date, Country, Product, Sales, Profit, Discount)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = {"Order_Date", "Country", "Product", "Sales", "Profit", "Discount"}
if not required_cols.issubset(df.columns):
    st.error("❌ CSV must contain: Order_Date, Country, Product, Sales, Profit, Discount")
    st.stop()

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# ======================================================
# 🔎 SIDEBAR FILTERS
# ======================================================
st.sidebar.header("🔎 Filters")

countries = sorted(df["Country"].unique())
products = sorted(df["Product"].unique())

selected_countries = st.sidebar.multiselect(
    "Countries", countries, default=countries
)
selected_products = st.sidebar.multiselect(
    "Products", products, default=products
)

min_date, max_date = df["Order_Date"].min(), df["Order_Date"].max()
start_date, end_date = st.sidebar.date_input(
    "Date Range", (min_date, max_date)
)

# ======================================================
# 🧹 APPLY FILTERS
# ======================================================
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Product"].isin(selected_products)) &
    (df["Order_Date"] >= pd.to_datetime(start_date)) &
    (df["Order_Date"] <= pd.to_datetime(end_date))
]

if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# ======================================================
# 🧠 SESSION PERSONALIZATION
# ======================================================
if "country_history" not in st.session_state:
    st.session_state.country_history = []
if "product_history" not in st.session_state:
    st.session_state.product_history = []

st.session_state.country_history.append(selected_countries)
st.session_state.product_history.append(selected_products)

# ======================================================
# 📊 KPI SECTION
# ======================================================
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
avg_discount = filtered_df["Discount"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
k2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
k3.metric("🏷 Avg Discount", f"{avg_discount:.2f}%")

st.divider()

# ======================================================
# 📈 MONTHLY SALES TREND (PLOTLY)
# ======================================================
st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df.groupby("Month", as_index=False)["Sales"].sum()
)

fig_trend = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend",
    hover_data={"Sales": ":,.0f"}
)
st.plotly_chart(fig_trend, use_container_width=True)

# ======================================================
# 🔮 AI FORECASTING (ARIMA)
# ======================================================
st.subheader("🔮 AI Sales Forecast (ARIMA)")

ts = (
    filtered_df
    .set_index("Order_Date")
    .resample("M")["Sales"]
    .sum()
)

if len(ts) < 8:
    st.warning("⚠️ Minimum 8 months required for ARIMA forecasting.")
else:
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()

    fitted = model_fit.predict(
        start=ts.index[1],
        end=ts.index[-1],
        typ="levels"
    )

    forecast_steps = 6
    forecast = model_fit.forecast(steps=forecast_steps)

    future_dates = pd.date_range(
        ts.index[-1] + pd.offsets.MonthEnd(1),
        periods=forecast_steps,
        freq="M"
    )

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode="lines+markers", name="Actual"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=fitted.index, y=fitted.values,
        mode="lines", name="Model Fit"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=future_dates, y=forecast.values,
        mode="lines+markers", name="Forecast"
    ))

    fig_forecast.update_layout(
        title="ARIMA Forecast – Next 6 Months",
        xaxis_title="Month",
        yaxis_title="Sales"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    mae = mean_absolute_error(ts.loc[fitted.index], fitted)
    st.info(f"📏 Forecast Reliability (MAE): ₹{mae:,.2f}")

# ======================================================
# 🚨 AI ANOMALY DETECTION (ISOLATION FOREST)
# ======================================================
st.subheader("🚨 AI Anomaly Detection")

if len(monthly_sales) >= 5:
    iso = IsolationForest(contamination=0.1, random_state=42)
    monthly_sales["anomaly"] = iso.fit_predict(monthly_sales[["Sales"]])
    anomalies = monthly_sales[monthly_sales["anomaly"] == -1]

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=monthly_sales["Month"],
        y=monthly_sales["Sales"],
        mode="lines+markers",
        name="Sales"
    ))
    fig_anom.add_trace(go.Scatter(
        x=anomalies["Month"],
        y=anomalies["Sales"],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Anomalies"
    ))

    fig_anom.update_layout(title="AI Detected Sales Anomalies")
    st.plotly_chart(fig_anom, use_container_width=True)

    if not anomalies.empty:
        st.warning(f"🚨 Anomalies detected in: {', '.join(anomalies['Month'])}")
    else:
        st.success("✅ No significant anomalies detected.")
else:
    st.info("ℹ️ Insufficient data for anomaly detection.")

# ======================================================
# 🎯 WHAT-IF SCENARIO PLANNING
# ======================================================
st.subheader("🎯 What-If Scenario Planning")

discount_change = st.slider("Discount Change (%)", -30, 30, 0)
price_change = st.slider("Price Change (%)", -30, 30, 0)

discount_effect = max(0.8, 1 + discount_change * 0.004)
price_effect = max(0.7, 1 - price_change * 0.006)

projected_sales = total_sales * discount_effect * price_effect
profit_margin = total_profit / total_sales if total_sales > 0 else 0
projected_profit = projected_sales * profit_margin

c1, c2 = st.columns(2)
c1.metric("Projected Sales", f"₹{projected_sales:,.0f}")
c2.metric("Projected Profit", f"₹{projected_profit:,.0f}")

# ======================================================
# 📊 PRODUCT-WISE SALES
# ======================================================
st.subheader("📊 Product-wise Sales")

product_sales = (
    filtered_df.groupby("Product", as_index=False)["Sales"].sum()
)

fig_prod = px.bar(
    product_sales,
    x="Product",
    y="Sales",
    title="Product-wise Sales",
    hover_data={"Sales": ":,.0f"}
)
st.plotly_chart(fig_prod, use_container_width=True)

# ======================================================
# 🔥 COUNTRY vs PRODUCT HEATMAP (PLOTLY)
# ======================================================
st.subheader("🔥 Country vs Product Heatmap")

pivot = pd.pivot_table(
    filtered_df,
    values="Sales",
    index="Country",
    columns="Product",
    aggfunc="sum"
)

fig_heat = px.imshow(
    pivot,
    text_auto=".0f",
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="Country vs Product Sales Heatmap"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ======================================================
# 🥧 SALES DISTRIBUTION (RADIO SWITCH)
# ======================================================
st.subheader("🥧 Sales Distribution")

pie_choice = st.radio(
    "View Sales Distribution By",
    ["Country", "Product"],
    horizontal=True
)

pie_df = (
    filtered_df.groupby(pie_choice, as_index=False)["Sales"].sum()
)

fig_pie = px.pie(
    pie_df,
    values="Sales",
    names=pie_choice,
    hole=0.4,
    title=f"Sales Share by {pie_choice}"
)
st.plotly_chart(fig_pie, use_container_width=True)

# ======================================================
# 📊 COUNTRY vs PRODUCT (GROUPED BAR)
# ======================================================
st.subheader("📊 Country vs Product – Comparative Analysis")

pivot_bar = pivot.reset_index()

fig_bar = px.bar(
    pivot_bar,
    x="Country",
    y=pivot.columns,
    barmode="group",
    title="Country vs Product Sales Comparison"
)
st.plotly_chart(fig_bar, use_container_width=True)

# ======================================================
# 💡 AI RECOMMENDATIONS
# ======================================================
st.subheader("💡 AI-Powered Recommendations")

recommendations = []

top_country = filtered_df.groupby("Country")["Sales"].sum().idxmax()
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax()

recommendations.append(
    f"🌍 **{top_country}** is the strongest market. Consider increasing inventory or targeted campaigns."
)
recommendations.append(
    f"📦 **{top_product}** is the top product. Bundling or premium pricing could improve margins."
)

if avg_discount > 30:
    recommendations.append("🏷 Discounts are high. Reducing discounts may improve profitability.")
else:
    recommendations.append("✅ Discount strategy is balanced.")

if "anomalies" in locals() and not anomalies.empty:
    recommendations.append("🚨 Investigate anomaly months for supply chain or demand shocks.")

for rec in recommendations:
    st.success(rec)

# ======================================================
# 📄 AI BUSINESS REPORT + DOWNLOAD
# ======================================================
st.subheader("📄 AI Sales Report")

report = f"""
GLOBAL SALES ANALYTICS REPORT
=============================

Total Sales   : ₹{total_sales:,.0f}
Total Profit  : ₹{total_profit:,.0f}
Avg Discount  : {avg_discount:.2f}%

Top Country   : {top_country}
Top Product   : {top_product}

Forecasting   : ARIMA (Short-term)
Anomalies     : {'Yes' if 'anomalies' in locals() and not anomalies.empty else 'No'}

Generated on  : {pd.Timestamp.now().strftime('%d-%m-%Y %H:%M:%S')}
"""

with st.expander("📊 View Detailed Report"):
    st.text(report)

buffer = BytesIO()
buffer.write(report.encode("utf-8"))
buffer.seek(0)

st.download_button(
    "⬇ Download AI Report",
    data=buffer,
    file_name="AI_Sales_Report.txt",
    mime="text/plain"
)
