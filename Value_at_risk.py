import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from scipy.stats import norm, chi2
from jinja2 import Template
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Value_at_risk Analysis App developed by Suman_econ_UAS(B)", layout="wide")
st.title("📉 Value at Risk Models Developed by Suman_econ_UAS(B)")

st.markdown("""
### 📌 Instructions Before Upload:
- it will be automatically converted to datetime and set as index.
- File format: `.csv`, `.xls`, or `.xlsx`
- Data should contain at least one price series
-Missing values will be automatically handled by imputation or dropped if minor.
""")

uploaded_file = st.file_uploader("Upload your crop market data", type=["csv", "xls", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df = df.sort_index()

    st.markdown(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_col = st.selectbox("Select Market Price Column", options=['All'] + numeric_cols)

    date_range = st.date_input("Select date range to analyze", [df.index.min(), df.index.max()])
    df = df.loc[(df.index >= pd.to_datetime(date_range[0])) & (df.index <= pd.to_datetime(date_range[1]))]

    analysis_cols = numeric_cols if selected_col == 'All' else [selected_col]

    confidence_level = 0.95
    z_score = norm.ppf(1 - confidence_level)
    num_sim = 10000
    window = 52

    results, briefs = [], []

    for col in analysis_cols:
        series = df[col].copy()
        series.replace(0, np.nan, inplace=True)
        series.dropna(inplace=True)

        if len(series) < 30:
            st.warning(f"Not enough data points in {col} to compute risk models.")
            continue

        log_returns = np.log(series / series.shift(1)).dropna()
        mu, sigma = log_returns.mean(), log_returns.std()

        hist_var = np.percentile(log_returns, (1 - confidence_level) * 100)
        param_var = mu + z_score * sigma

        sim_returns = np.random.normal(mu, sigma, size=num_sim)
        mc_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
        mc_cvar = sim_returns[sim_returns <= mc_var].mean()

        results.append({
            "Market": col,
            "Historical VaR (%)": round((np.exp(hist_var)-1)*100, 2),
            "Parametric VaR (%)": round((np.exp(param_var)-1)*100, 2),
            "Monte Carlo VaR (%)": round((np.exp(mc_var)-1)*100, 2),
            "Monte Carlo CVaR (%)": round((np.exp(mc_cvar)-1)*100, 2),
        })

        brief_template = Template("""
        In the recent analysis for {{ market }}, we found the following risk indicators:
        - Historical VaR at 95%%: {{ hist }}%%
        - Parametric VaR at 95%%: {{ param }}%%
        - Monte Carlo VaR at 95%%: {{ mc }}%%
        - Monte Carlo CVaR (Expected Shortfall) at 95%%: {{ cvar }}%%
        """)
        brief = brief_template.render(market=col,
                                      hist=round((np.exp(hist_var)-1)*100, 2),
                                      param=round((np.exp(param_var)-1)*100, 2),
                                      mc=round((np.exp(mc_var)-1)*100, 2),
                                      cvar=round((np.exp(mc_cvar)-1)*100, 2))
        briefs.append((col, brief))

        fig, ax = plt.subplots()
        sns.histplot(sim_returns, bins=50, kde=True, ax=ax, color='orange')
        ax.axvline(mc_var, color='red', linestyle='--', label=f"VaR 95%")
        ax.axvline(mc_cvar, color='purple', linestyle='--', label=f"CVaR 95%")
        ax.set_title(f"Monte Carlo Simulated Returns: {col}")
        ax.set_xlabel("Log Returns")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"#### 📌 Interpretation for {col}:")
        st.markdown(f"- Historical VaR: **{(np.exp(hist_var)-1)*100:.2f}%**")
        st.markdown(f"- Parametric VaR: **{(np.exp(param_var)-1)*100:.2f}%**")
        st.markdown(f"- Monte Carlo VaR: **{(np.exp(mc_var)-1)*100:.2f}%**")
        st.markdown(f"- Monte Carlo CVaR: **{(np.exp(mc_cvar)-1)*100:.2f}%**")

    if results:
        st.subheader("📋 Model Comparison Table")
        st.dataframe(pd.DataFrame(results).set_index("Market"))

    if briefs:
        st.subheader("📝 Automated Policy Briefs")
        for market, text in briefs:
            with st.expander(f"Policy Brief for {market}"):
                st.markdown(text)

    # Optional: Backtesting Section
    if 'Modal' in df.columns and 'Arrivals' in df.columns:
        st.subheader("🔎 Monte Carlo VaR Backtesting (Log Returns ~ Log Arrivals)")
        ds = df[['Modal', 'Arrivals']].copy()
        ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
        ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
        ds.dropna(inplace=True)

        backtest_results = []

        for i in range(window, len(ds)):
            train = ds.iloc[i - window:i]
            test = ds.iloc[i]
            X = sm.add_constant(train['Log_Arrivals'])
            y = train['Log_Returns']
            model = sm.OLS(y, X).fit()

            sim_arr = np.random.normal(train['Log_Arrivals'].mean(), train['Log_Arrivals'].std(), num_sim)
            sim_mu = model.params[0] + model.params[1] * sim_arr
            sim_ret = np.random.normal(sim_mu, model.resid.std(), size=num_sim)

            mc_var_bt = np.percentile(sim_ret, (1 - confidence_level) * 100)
            mc_cvar_bt = sim_ret[sim_ret <= mc_var_bt].mean()

            actual_ret = test['Log_Returns']
            backtest_results.append({
                'Date': ds.index[i],
                'Actual_Return': actual_ret,
                'MC_VaR': mc_var_bt,
                'MC_CVaR': mc_cvar_bt,
                'Breach_VaR': actual_ret < mc_var_bt,
                'Breach_CVaR': actual_ret < mc_cvar_bt
            })

        bt_df = pd.DataFrame(backtest_results).set_index('Date')
        bt_df['Breach_VaR'] = bt_df['Breach_VaR'].astype(int)

        st.line_chart(bt_df[['Actual_Return', 'MC_VaR']])
        st.markdown(f"**Observed Breach Rate (VaR):** {bt_df['Breach_VaR'].mean()*100:.2f}%")

        # Kupiec Test
        F = bt_df['Breach_VaR'].sum()
        T = len(bt_df)
        p = 1 - confidence_level
        observed_p = bt_df['Breach_VaR'].mean()
        stat = -2 * np.log(((1 - p)**(T - F) * p**F) / ((1 - observed_p)**(T - F) * observed_p**F))
        p_val = 1 - chi2.cdf(stat, df=1)

        st.markdown(f"**Kupiec's POF Test Statistic:** {stat:.4f}")
        st.markdown(f"**P-value:** {p_val:.4f}")
        if p_val < 0.05:
            st.error("❌ VaR model does not fit well (reject null hypothesis).")
        else:
            st.success("✅ VaR model fits well (fail to reject null hypothesis).")
else:
    st.info("Upload your weekly crop market data to begin analysis.")

# Footer
st.markdown("""
<hr style="border:1px solid #ccc" />

<div style="text-align: center; font-size: 14px; color: gray;">
    🚀 This app was built by <b>Suman L</b> <br>
    📬 For support or collaboration, contact: <a href="mailto:sumanecon.uas@outlook.com">sumanecon.uas@outlook.com</a>
</div>
""", unsafe_allow_html=True)
