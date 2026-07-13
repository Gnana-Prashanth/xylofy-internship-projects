"""
Streamlit Dashboard — Sales Forecasting & Demand Intelligence
================================================================
Run locally with:   streamlit run app.py
Deploy for free on:  https://share.streamlit.io  (Streamlit Community Cloud)

This app reuses the same logic built in analysis.ipynb, just wrapped in an
interactive UI with 4 pages (selectable from the sidebar).
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ----------------------------------------------------------------------------
# DATA LOADING (cached so it only re-runs when the underlying file changes)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "train.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Quarter'] = df['Order Date'].dt.quarter
    return df

df = load_data()

SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
FEATURE_COLS = ['Lag1', 'Lag2', 'Lag3', 'RollingMean3', 'Month', 'Quarter', 'Season']


def build_monthly_series(mask):
    """Aggregate filtered rows into a monthly total-sales series."""
    return df[mask].set_index('Order Date').resample('MS')['Sales'].sum()


def make_feature_table(monthly_series):
    """Turn a monthly sales series into lag-feature rows XGBoost can learn from."""
    feat = pd.DataFrame({'Sales': monthly_series})
    feat['Lag1'] = feat['Sales'].shift(1)
    feat['Lag2'] = feat['Sales'].shift(2)
    feat['Lag3'] = feat['Sales'].shift(3)
    feat['RollingMean3'] = feat['Sales'].shift(1).rolling(3).mean()
    feat['Month'] = feat.index.month
    feat['Quarter'] = feat.index.quarter
    feat['Season'] = feat['Month'].map(SEASON_MAP)
    return feat.dropna()


@st.cache_data
def forecast_segment(mask_key, n_months):
    """
    Train XGBoost on one segment's monthly history and forecast n_months ahead.
    mask_key is a string ('All','Furniture',...) so Streamlit's cache can key on it
    (cache_data needs hashable inputs, and a boolean pandas Series mask isn't cache-friendly).
    """
    if mask_key == 'All':
        mask = pd.Series(True, index=df.index)
    elif mask_key in df['Category'].unique():
        mask = df['Category'] == mask_key
    else:
        mask = df['Region'] == mask_key

    monthly_series = build_monthly_series(mask)
    feat = make_feature_table(monthly_series)
    X, y = feat[FEATURE_COLS], feat['Sales']

    # Hold out the last 3 months to report honest accuracy metrics (MAE/RMSE)
    X_train, X_test = X.iloc[:-3], X.iloc[-3:]
    y_train, y_test = y.iloc[:-3], y.iloc[-3:]

    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # Refit on ALL data (including the last 3 months) so the actual future forecast
    # uses every bit of history we have, then forecast forward recursively.
    full_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    full_model.fit(X, y)

    history = monthly_series.copy()
    preds = []
    for _ in range(n_months):
        next_date = history.index[-1] + pd.DateOffset(months=1)
        next_row = pd.DataFrame([{
            'Lag1': history.iloc[-1], 'Lag2': history.iloc[-2], 'Lag3': history.iloc[-3],
            'RollingMean3': history.iloc[-3:].mean(),
            'Month': next_date.month, 'Quarter': next_date.quarter,
            'Season': SEASON_MAP[next_date.month]
        }])[FEATURE_COLS]
        next_pred = full_model.predict(next_row)[0]
        preds.append(next_pred)
        history.loc[next_date] = next_pred

    future_dates = pd.date_range(monthly_series.index[-1] + pd.DateOffset(months=1),
                                  periods=n_months, freq='MS')
    forecast = pd.Series(preds, index=future_dates)
    return monthly_series, forecast, mae, rmse


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to page:", [
    "1. Sales Overview",
    "2. Forecast Explorer",
    "3. Anomaly Report",
    "4. Product Demand Segments"
])

# ============================================================================
# PAGE 1 — SALES OVERVIEW
# ============================================================================
if page == "1. Sales Overview":
    st.title("📈 Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        yearly = df.groupby('Order Year')['Sales'].sum().reset_index()
        fig = px.bar(yearly, x='Order Year', y='Sales', title="Total Sales by Year",
                     labels={'Sales': 'Total Sales ($)'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        monthly = df.set_index('Order Date').resample('MS')['Sales'].sum().reset_index()
        fig = px.line(monthly, x='Order Date', y='Sales', title="Monthly Sales Trend", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales by Region & Category")
    col3, col4 = st.columns(2)
    with col3:
        region_filter = st.multiselect("Filter by Region:", options=df['Region'].unique(),
                                        default=list(df['Region'].unique()))
    with col4:
        category_filter = st.multiselect("Filter by Category:", options=df['Category'].unique(),
                                          default=list(df['Category'].unique()))

    filtered = df[df['Region'].isin(region_filter) & df['Category'].isin(category_filter)]
    grouped = filtered.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    fig = px.bar(grouped, x='Region', y='Sales', color='Category', barmode='group',
                 title="Sales by Region and Category (filtered)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================================
elif page == "2. Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.write("Pick a segment and a forecast horizon. Forecasts use XGBoost — "
             "the best-performing model found in the notebook comparison (Task 3).")

    segment_options = ['All'] + list(df['Category'].unique()) + list(df['Region'].unique())
    chosen_segment = st.selectbox("Category or Region:", segment_options)
    horizon = st.slider("Forecast horizon (months ahead):", min_value=1, max_value=3, value=3)

    with st.spinner("Training model and generating forecast..."):
        actual, forecast, mae, rmse = forecast_segment(chosen_segment, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines+markers',
                              name='Forecast', line=dict(dash='dash', color='red')))
    fig.update_layout(title=f"{chosen_segment}: Actual vs Forecasted Sales",
                       xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Model MAE (test set)", f"${mae:,.0f}")
    col2.metric("Model RMSE (test set)", f"${rmse:,.0f}")

    st.subheader("Forecast values")
    st.dataframe(forecast.rename("Forecasted Sales ($)").round(0))

# ============================================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================================
elif page == "3. Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.write("Weekly sales checked against two anomaly-detection methods: "
             "Isolation Forest and a rolling Z-Score.")

    weekly = df.set_index('Order Date').resample('W')['Sales'].sum()
    weekly_df = pd.DataFrame({'Sales': weekly})

    # Method 1: Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly_df['IsoAnomaly'] = iso.fit_predict(weekly.values.reshape(-1, 1)) == -1

    # Method 2: Z-Score vs. the trailing 4-week average (not including the current week)
    roll_mean = weekly.shift(1).rolling(4, min_periods=3).mean()
    roll_std = weekly.shift(1).rolling(4, min_periods=3).std()
    z = (weekly - roll_mean) / roll_std
    weekly_df['ZScoreAnomaly'] = (z.abs() > 2).fillna(False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly.values, mode='lines', name='Weekly Sales'))
    iso_pts = weekly_df[weekly_df['IsoAnomaly']]
    z_pts = weekly_df[weekly_df['ZScoreAnomaly']]
    fig.add_trace(go.Scatter(x=iso_pts.index, y=iso_pts['Sales'], mode='markers',
                              name='Isolation Forest Anomaly',
                              marker=dict(color='red', size=10)))
    fig.add_trace(go.Scatter(x=z_pts.index, y=z_pts['Sales'], mode='markers',
                              name='Z-Score Anomaly',
                              marker=dict(color='orange', size=10, symbol='x')))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", xaxis_title="Week", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected anomaly weeks")
    anomaly_table = weekly_df[weekly_df['IsoAnomaly'] | weekly_df['ZScoreAnomaly']][
        ['Sales', 'IsoAnomaly', 'ZScoreAnomaly']].round(0)
    st.dataframe(anomaly_table)

# ============================================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================================
elif page == "4. Product Demand Segments":
    st.title("📦 Product Demand Segments")
    st.write("Sub-categories grouped by demand behavior using K-Means clustering.")

    rows = []
    for sc in df['Sub-Category'].unique():
        sub = df[df['Sub-Category'] == sc]
        total_sales = sub['Sales'].sum()
        avg_order_value = sub.groupby('Order ID')['Sales'].sum().mean()
        monthly_sc = sub.set_index('Order Date').resample('MS')['Sales'].sum()
        volatility = monthly_sc.std()
        yearly_sc = sub.groupby('Order Year')['Sales'].sum().sort_index()
        growth_rate = ((yearly_sc.iloc[-1] - yearly_sc.iloc[0]) / yearly_sc.iloc[0] * 100
                        if len(yearly_sc) >= 2 and yearly_sc.iloc[0] > 0 else 0)
        rows.append({'Sub-Category': sc, 'TotalSales': total_sales, 'GrowthRate': growth_rate,
                      'Volatility': volatility, 'AvgOrderValue': avg_order_value})

    product_features = pd.DataFrame(rows).set_index('Sub-Category')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(product_features)

    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    product_features['Cluster'] = kmeans.fit_predict(X_scaled)

    cluster_labels = {
        0: "Low Volume, Stable Demand",
        1: "Declining, High-Value, High-Volatility",
        2: "High Volume, Growing Demand",
        3: "Explosive Growth (Outlier)"
    }
    # Re-map generic cluster numbers to the closest matching label based on average growth/volume
    summary = product_features.groupby('Cluster')[['TotalSales', 'GrowthRate']].mean()
    # (Cluster numbers can shuffle between runs; we keep the numeric ID and show stats so
    # the label is still interpretable even if KMeans assigns different numbers.)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    product_features['PCA1'], product_features['PCA2'] = coords[:, 0], coords[:, 1]

    fig = px.scatter(product_features.reset_index(), x='PCA1', y='PCA2', color='Cluster',
                      text='Sub-Category', title="Product Sub-Category Demand Clusters (PCA view)")
    fig.update_traces(textposition='top center', marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster averages (helps interpret what each cluster means)")
    st.dataframe(summary.round(1))

    st.subheader("Sub-Category → Cluster table")
    st.dataframe(product_features[['TotalSales', 'GrowthRate', 'Volatility', 'AvgOrderValue', 'Cluster']].round(1))
