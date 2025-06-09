import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import datetime as dt
import base64
from scipy.stats import norm
from jinja2 import Template
import warnings
warnings.filterwarnings("ignore")

# App Title
st.set_page_config(page_title="Value_at_risk Analysis App devloped by Suman_econ_UAS(B)", layout="wide")
st.title("📉 Value at Risk Models Developed by Suman_econ_UAS(B)")

# Instructions
st.markdown("""
### 📌 Instructions Before Upload:
- Ensure your data contains a Date column. If found, it will be automatically converted to datetime and set as index.
- File format: .csv, .xls, or .xlsx
- Data should contain at least one price series (e.g., weekly modal prices).
- Missing values will be automatically handled by imputation or dropped if minor.
""")

# File upload
uploaded_file = st.file_uploader("Upload your crop market data", type=["csv", "xls", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Convert Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df = df.sort_index()

    st.markdown(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")

    # Drop non-numeric columns except date index
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_col = st.selectbox("Select Market Price Column", options=['All'] + numeric_cols)

    # Select date range
    date_range = st.date_input("Select date range to analyze", [df.index.min(), df.index.max()])
    df = df.loc[(df.index >= pd.to_datetime(date_range[0])) & (df.index <= pd.to_datetime(date_range[1]))]

    # Data preprocessing
    if selected_col == 'All':
        analysis_cols = numeric_cols
    else:
        analysis_cols = [selected_col]

    results = []
    briefs = []

    for col in analysis_cols:
        series = df[col].copy()
        series.replace(0, np.nan, inplace=True)
        series.dropna(inplace=True)

        if len(series) < 30:
            st.warning(f"Not enough data points in {col} to compute risk models.")
            continue

        # Compute log returns
        log_returns = np.log(series / series.shift(1)).dropna()
        mu = log_returns.mean()
        sigma = log_returns.std()
        alpha = 0.95
        z_score = norm.ppf(1 - alpha)

        # Historical VaR
        hist_var = np.percentile(log_returns, (1 - alpha) * 100)

        # Parametric VaR
        param_var = mu + z_score * sigma

        # Monte Carlo Simulation
        num_sim = 10000
        sim_returns = np.random.normal(mu, sigma, size=num_sim)
        mc_var = np.percentile(sim_returns, (1 - alpha) * 100)
        mc_cvar = sim_returns[sim_returns <= mc_var].mean()

        results.append({
            "Market": col,
            "Historical VaR (%)": round((np.exp(hist_var)-1)*100, 2),
            "Parametric VaR (%)": round((np.exp(param_var)-1)*100, 2),
            "Monte Carlo VaR (%)": round((np.exp(mc_var)-1)*100, 2),
            "Monte Carlo CVaR (%)": round((np.exp(mc_cvar)-1)*100, 2),
        })

        # Automated Policy Brief
        brief_template = Template("""
        In the recent analysis for {{ market }}, we found the following risk indicators:
        - Historical VaR at 95%%: {{ hist }}%%
        - Parametric VaR at 95%%: {{ param }}%%
        - Monte Carlo VaR at 95%%: {{ mc }}%%
        - Monte Carlo CVaR (Expected Shortfall) at 95%%: {{ cvar }}%%

        Based on this, we recommend initiating policy measures such as buffer stock planning or early market alerts for {{ market }} especially during volatile periods.
        """)
        brief = brief_template.render(market=col, hist=round((np.exp(hist_var)-1)*100, 2),
                                      param=round((np.exp(param_var)-1)*100, 2),
                                      mc=round((np.exp(mc_var)-1)*100, 2),
                                      cvar=round((np.exp(mc_cvar)-1)*100, 2))
        briefs.append((col, brief))

        # Plot histogram
        fig, ax = plt.subplots()
        sns.histplot(sim_returns, bins=50, kde=True, ax=ax, color='orange')
        ax.axvline(mc_var, color='red', linestyle='--', label=f"VaR 95%")
        ax.axvline(mc_cvar, color='purple', linestyle='--', label=f"CVaR 95%")
        ax.set_title(f"Monte Carlo Simulated Returns: {col}")
        ax.set_xlabel("Log Returns")
        ax.legend()
        st.pyplot(fig)

        # Interpretation
        st.markdown("#### 📌 Interpretation for {}:".format(col))
        st.markdown("- The **historical VaR** implies a weekly loss of up to **{:.2f}%** in 5% of the worst cases.".format((np.exp(hist_var)-1)*100))
        st.markdown("- The **parametric VaR** uses normality assumptions and gives **{:.2f}%**.".format((np.exp(param_var)-1)*100))
        st.markdown("- The **Monte Carlo simulation** shows a possible loss beyond **{:.2f}%**, with an **expected loss (CVaR)** of **{:.2f}%** in extreme weeks.".format((np.exp(mc_var)-1)*100, (np.exp(mc_cvar)-1)*100))

    # Summary Table
    if results:
        st.subheader("📋 Model Comparison Table")
        result_df = pd.DataFrame(results)
        st.dataframe(result_df.set_index("Market"))

    # Policy Briefs
    if briefs:
        st.subheader("📝 Automated Policy Briefs")
        for market, text in briefs:
            with st.expander(f"Policy Brief for {market}"):
                st.markdown(text)

else:
    st.info("Upload your weekly crop market data to begin analysis.")
