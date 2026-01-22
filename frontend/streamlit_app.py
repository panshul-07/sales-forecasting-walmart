import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="Walmart Demand Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .kpi-card {
            background-color: #111;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #222;
        }
        .kpi-title {
            font-size: 14px;
            color: #aaa;
        }
        .kpi-value {
            font-size: 32px;
            font-weight: 600;
            margin-top: 4px;
        }
        .section {
            margin-top: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Walmart Demand Forecasting")

@st.cache_data
def load_data():
    try:
        if "csv_data" in st.secrets and st.secrets["csv_data"].strip():
            df = pd.read_csv(io.StringIO(st.secrets["csv_data"]))
        else:
            st.error("csv_data not found in Streamlit secrets.")
            return None

        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        return df

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input Parameters")

store_id = st.sidebar.number_input(
    "Store",
    min_value=int(df["Store"].min()),
    max_value=int(df["Store"].max()),
    value=int(df["Store"].min())
)

holiday_flag = st.sidebar.toggle("Holiday Week")

temperature = st.sidebar.slider("Temperature (°F)", 20.0, 120.0, 70.0)
fuel_price = st.sidebar.slider("Fuel Price ($/gal)", 2.0, 5.0, 3.5)
cpi = st.sidebar.slider("CPI", 200.0, 300.0, 220.0)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 3.0, 15.0, 7.0)

# ---------------- MODEL ----------------
store_df = df[df["Store"] == store_id].sort_values("Date")
avg_sales = store_df["Weekly_Sales"].mean()

CONST = avg_sales
COEF_HOLIDAY = 6634.0369
COEF_FUEL = 11830.0
COEF_CPI = -9.8499
COEF_UNEMP = -418.9919

TEMP_OPTIMAL = 70.0
TEMP_CURVATURE = -196.8391

def temperature_effect(t):
    return TEMP_CURVATURE * (t - TEMP_OPTIMAL) ** 2

predicted_sales = (
    CONST
    + COEF_HOLIDAY * int(holiday_flag)
    + temperature_effect(temperature)
    + COEF_FUEL * fuel_price
    + COEF_CPI * cpi
    + COEF_UNEMP * unemployment
)

# ---------------- KPI SECTION ----------------
st.markdown('<div class="section"></div>', unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Predicted Weekly Sales</div>
            <div class="kpi-value">${predicted_sales:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Store Sales</div>
            <div class="kpi-value">${avg_sales:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    delta = ((predicted_sales - avg_sales) / avg_sales) * 100
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Change vs Average</div>
            <div class="kpi-value">{delta:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- TREND ----------------
st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.subheader("Recent Sales Trend")

recent = store_df.tail(20)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(recent["Date"], recent["Weekly_Sales"], marker="o", label="Actual")
ax.axhline(predicted_sales, linestyle="--", label="Predicted")
ax.legend()
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter('${x:,.0f}')

st.pyplot(fig, use_container_width=True)

st.caption("Walmart Demand Forecasting")
