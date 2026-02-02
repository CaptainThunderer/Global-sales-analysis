import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
from scipy.stats import zscore
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error

# =====================================================
# ⚙️ UI CONFIG
# =====================================================
st.set_page_config(
    page_title="Global Sales Analytics Dashboard",
    layout="wide"
)

st.title("🌍 Global Sales Analytics Dashboard")
st.caption("Machine Learning Powered Business Intelligence System")

# =====================================================
# 📤 DATA UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "📤 Upload Sales CSV (Order_Date, Country, Product, Sales, Profit, Discount)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to start analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = {"Order_Date", "Country", "Product", "Sales", "Profit", "Discount"}
if not required_cols.issubset(df.columns):
    st.error("❌ CSV must contain Order_Date, Country, Product, Sales, Profit, Discount")
    st.stop()

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# =====================================================
# 🔎 SIDEBAR FILTERS
# =====================================================
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

# =====================================================
# 🧹 APPLY FILTERS
# =====================================================
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Product"].isin(selected_products)) &
    (df["Order_Date"] >= pd.to_datetime(start_date)) &
    (df["Order_Date"] <= pd.to_datetime(end_date))
]

if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# =====================================================
# 🧠 SESSION HISTORY (PERSONALIZATION)
# =====================================================
if "country_history" not in st.session_state:
    st.session_state.country_history = []
if "product_history" not in st.session_state:
    st.session_state.product_history = []

st.session_state.country_history.append(selected_countries)
st.session_state.product_history.append(selected_products)

# =====================================================
# 📊 KPI SECTION
# =====================================================
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
avg_discount = filtered_df["Discount"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
k2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
k3.metric("🏷 Avg Discount", f"{avg_discount:.2f}%")

st.divider()

# =====================================================
# 📈 MONTHLY SALES TREND
# =====================================================
monthly_sales = filtered_df.groupby("Month", as_index=False)["Sales"].sum()

fig_trend = px.line(
    monthly_sales, x="Month", y="Sales",
    markers=True, title="Monthly Sales Trend"
)
st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# 🔮 ARIMA FORECASTING (ML)
# =====================================================
st.subheader("🔮 Sales Forecasting (ARIMA)")

ts = (
    filtered_df
    .set_index("Order_Date")
    .resample("M")["Sales"]
    .sum()
)

if len(ts) >= 8:
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()

    fitted = model_fit.predict(start=ts.index[1], end=ts.index[-1], typ="levels")

    future_steps = 6
    forecast = model_fit.forecast(steps=future_steps)
    future_dates = pd.date_range(
        ts.index[-1] + pd.offsets.MonthEnd(1),
        periods=future_steps, freq="M"
    )

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=ts.index, y=ts, name="Actual"))
    fig_forecast.add_trace(go.Scatter(x=fitted.index, y=fitted, name="Model Fit"))
    fig_forecast.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast"))

    fig_forecast.update_layout(title="ARIMA Forecast (Next 6 Months)")
    st.plotly_chart(fig_forecast, use_container_width=True)

    mae = mean_absolute_error(ts.loc[fitted.index], fitted)
    st.info(f"📏 Forecast MAE: ₹{mae:,.2f}")
else:
    st.warning("⚠️ Not enough data for ARIMA forecasting.")

# =====================================================
# 🚨 ANOMALY DETECTION (TWO METHODS)
# =====================================================
st.subheader("🚨 Anomaly Detection")

monthly_sales["Z_Score"] = zscore(monthly_sales["Sales"])
monthly_sales["Z_Anomaly"] = monthly_sales["Z_Score"].abs() > 2.5

iso = IsolationForest(contamination=0.1, random_state=42)
monthly_sales["IF_Anomaly"] = iso.fit_predict(monthly_sales[["Sales"]]) == -1

fig_anom = go.Figure()
fig_anom.add_trace(go.Scatter(
    x=monthly_sales["Month"], y=monthly_sales["Sales"],
    mode="lines+markers", name="Sales"
))
fig_anom.add_trace(go.Scatter(
    x=monthly_sales[monthly_sales["Z_Anomaly"]]["Month"],
    y=monthly_sales[monthly_sales["Z_Anomaly"]]["Sales"],
    mode="markers", marker=dict(color="orange", size=10),
    name="Z-Score Anomaly"
))
fig_anom.add_trace(go.Scatter(
    x=monthly_sales[monthly_sales["IF_Anomaly"]]["Month"],
    y=monthly_sales[monthly_sales["IF_Anomaly"]]["Sales"],
    mode="markers", marker=dict(color="red", size=12),
    name="Isolation Forest Anomaly"
))
fig_anom.update_layout(title="Anomaly Detection Comparison")
st.plotly_chart(fig_anom, use_container_width=True)

# =====================================================
# 🎯 WHAT-IF SCENARIO PLANNING
# =====================================================
st.subheader("🎯 What-If Scenario Analysis")

discount_change = st.slider("Discount Change (%)", -30, 30, 0)
price_change = st.slider("Price Change (%)", -30, 30, 0)

discount_effect = max(0.7, 1 + discount_change * 0.004)
price_effect = max(0.7, 1 - price_change * 0.006)

projected_sales = total_sales * discount_effect * price_effect
profit_margin = total_profit / total_sales if total_sales > 0 else 0
projected_profit = projected_sales * profit_margin

c1, c2 = st.columns(2)
c1.metric("Projected Sales", f"₹{projected_sales:,.0f}")
c2.metric("Projected Profit", f"₹{projected_profit:,.0f}")

scenario_df = pd.DataFrame({
    "Metric": ["Sales", "Profit"],
    "Original": [total_sales, total_profit],
    "Projected": [projected_sales, projected_profit]
})

fig_scenario = px.bar(
    scenario_df, x="Metric",
    y=["Original", "Projected"],
    barmode="group",
    title="What-If Scenario Impact"
)
st.plotly_chart(fig_scenario, use_container_width=True)

# =====================================================
# 🔥 HEATMAP
# =====================================================
# =====================================================
# 🔥 PREPARE DATA FOR HEATMAP (FIX)
# =====================================================
pivot = pd.pivot_table(
    filtered_df,
    values="Sales",
    index="Country",
    columns="Product",
    aggfunc="sum"
)

st.subheader("🔥 Country vs Product Correlation Heatmap")

fig_heat = px.imshow(
    pivot,
    text_auto=".0f",
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="Sales Intensity by Country & Product"
)

fig_heat.update_layout(
    height=700,                 # 🔥 makes it tall & readable
    width=1100,                 # 🔥 makes it wide
    xaxis_title="Product",
    yaxis_title="Country",
    font=dict(size=13),
    coloraxis_colorbar=dict(
        title="Sales (₹)",
        thickness=18
    )
)

fig_heat.update_xaxes(side="bottom")
st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================
# 🥧 PIE CHART
# =====================================================
pie_choice = st.radio("Sales Distribution By", ["Country", "Product"], horizontal=True)

pie_df = filtered_df.groupby(pie_choice, as_index=False)["Sales"].sum()

fig_pie = px.pie(
    pie_df, values="Sales", names=pie_choice,
    hole=0.4, title=f"Sales Distribution by {pie_choice}"
)
st.plotly_chart(fig_pie, use_container_width=True)

# =====================================================
# 💡 AI RECOMMENDATIONS
# =====================================================
st.subheader("💡 AI Recommendations")

top_country = filtered_df.groupby("Country")["Sales"].sum().idxmax()
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax()

st.success(f"🌍 Focus more on **{top_country}** – highest revenue market.")
st.success(f"📦 **{top_product}** is the best-performing product.")

if avg_discount > 30:
    st.warning("🏷 Discounts are high. Consider optimizing pricing.")
else:
    st.success("✅ Discount levels are healthy.")

# =====================================================
# 📄 AI REPORT CONTENT (GUARANTEED)
# =====================================================
# ---- SAFE MAE TEXT FOR REPORT ----
if 'mae' in locals() and isinstance(mae, (int, float)):
    mae_text = f"₹{mae:,.2f}"
else:
    mae_text = "N/A"

report = f"""
GLOBAL SALES ANALYTICS – DETAILED AI REPORT
==========================================

DATA CONTEXT
------------
Countries Selected : {", ".join(selected_countries)}
Products Selected  : {", ".join(selected_products)}
Date Range         : {start_date} to {end_date}

KEY METRICS
-----------
Total Sales        : ₹{total_sales:,.0f}
Total Profit       : ₹{total_profit:,.0f}
Average Discount   : {avg_discount:.2f}%
Profit Margin      : {(total_profit/total_sales*100):.2f}%

MARKET INSIGHTS
---------------
Top Country        : {top_country}
Top Product        : {top_product}

FORECASTING
-----------
Method             : ARIMA
Forecast Horizon   : 6 Months
MAE                : {mae_text}


ANOMALY DETECTION
-----------------
Techniques Used    : Z-Score, Isolation Forest
Anomalies Detected : {"Yes" if 'IF_Anomaly' in monthly_sales.columns and monthly_sales['IF_Anomaly'].any() else "No"}

STRATEGIC NOTES
---------------
• Focus marketing on {top_country}
• Leverage demand for {top_product}
• Maintain current discount strategy
• Use forecast for inventory planning

Generated On:
{pd.Timestamp.now().strftime('%d-%m-%Y %H:%M:%S')}
"""
st.subheader("📄 AI Report")

with st.expander("📊 View AI Report", expanded=False):
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
