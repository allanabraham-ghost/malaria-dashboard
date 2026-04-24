import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def main():
    # =========================
    # PAGE CONFIG
    # =========================
    st.set_page_config(
        page_title="Disease Early Warning System",
        layout="wide"
    )

    # =========================
    # LOAD DATA
    # =========================
    @st.cache_data
    def load_data():
        data = pd.read_csv("final_combined_dataset.csv")
        data['date'] = pd.to_datetime(data['date'])
        return data

    data = load_data()

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.title("Dashboard Controls")

    district = st.sidebar.selectbox(
        "Select District",
        sorted(data['district'].unique())
    )

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [data['date'].min(), data['date'].max()]
    )

    threshold = st.sidebar.slider(
        "Alert Threshold (Cases)",
        10, 200, 50
    )

    # =========================
    # FILTER DATA
    # =========================
    filtered = data[
        (data['district'] == district) &
        (data['date'] >= pd.to_datetime(date_range[0])) &
        (data['date'] <= pd.to_datetime(date_range[1]))
    ]

    # Add the threshold to the filtered data as a new column
    filtered = filtered.copy() # Ensure we're working on a copy to avoid SettingWithCopyWarning
    filtered['threshold'] = threshold

    # =========================
    # TITLE
    # =========================
    st.title("Disease Outbreak Early Warning System")
    st.markdown("### ARIMA + Random Forest Prediction Dashboard")

    # =========================
    # SUMMARY METRICS
    # =========================
    st.subheader("Key Indicators")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Cases", int(filtered['cases'].sum()))
    col2.metric("Average Rainfall", round(filtered['rainfall'].mean(), 2))
    col3.metric("Max Cases", int(filtered['cases'].max()))
    col4.metric("Avg Prediction (RF)", round(filtered['rf_prediction'].mean(), 2))

    # =========================
    # ACTUAL VS PREDICTED
    # =========================
    st.subheader("Actual vs Predicted Cases")

    fig1, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(filtered['date'], filtered['cases'], label="Actual", linewidth=2)
    ax1.plot(filtered['date'], filtered['arima_forecast'], linestyle="--", label="ARIMA")
    ax1.plot(filtered['date'], filtered['rf_prediction'], linestyle="--", label="Random Forest")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cases")
    ax1.legend()

    st.pyplot(fig1)
    plt.close(fig1) # Close the figure to prevent display issues in some environments

    # =========================
    # RAINFALL + CASES (DUAL VIEW)
    # =========================
    st.subheader("Rainfall vs Cases")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.plot(filtered['date'], filtered['rainfall'], label="Rainfall")
    ax2.set_ylabel("Rainfall")

    ax3 = ax2.twinx()
    ax3.plot(filtered['date'], filtered['cases'], linestyle="--", label="Cases")
    ax3.set_ylabel("Cases")

    st.pyplot(fig2)
    plt.close(fig2)

    # =========================
    # CORRELATION HEATMAP
    # =========================
    st.subheader("Correlation Analysis")

    corr = filtered[['cases', 'rainfall', 'arima_forecast', 'rf_prediction', 'threshold']].corr() # Added 'threshold' to correlation

    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)

    st.pyplot(fig3)
    plt.close(fig3)

    # =========================
    # MODEL PERFORMANCE
    # =========================
    st.subheader("Model Performance")

    perf = filtered.dropna(subset=['cases', 'arima_forecast', 'rf_prediction'])

    if not perf.empty:
        rmse_arima = mean_squared_error(
            perf['cases'], perf['arima_forecast'], squared=False
        )
        rmse_rf = mean_squared_error(
            perf['cases'], perf['rf_prediction'], squared=False
        )

        colA, colB = st.columns(2)
        colA.metric("ARIMA RMSE", round(rmse_arima, 2))
        colB.metric("Random Forest RMSE", round(rmse_rf, 2))
    else:
        st.warning("Not enough data for performance calculation")

    # =========================
    # ALERT SYSTEM
    # =========================
    st.subheader("Outbreak Alert System")

    if not filtered['rf_prediction'].dropna().empty:
        latest_rf = filtered['rf_prediction'].dropna().iloc[-1]

        if latest_rf > threshold:
            st.error(f"⚠ HIGH RISK: Predicted Cases = {round(latest_rf,2)}")
        else:
            st.success(f" LOW RISK: Predicted Cases = {round(latest_rf,2)}")
    else:
        st.warning("No prediction data available")

    # =========================
    # DISTRICT COMPARISON
    # =========================
    st.subheader("Compare Districts")

    multi_district = st.multiselect(
        "Select Multiple Districts",
        data['district'].unique(),
        default=[district]
    )

    compare_data = data[data['district'].isin(multi_district)]

    fig4, ax4 = plt.subplots(figsize=(10,5))

    for d in multi_district:
        temp = compare_data[compare_data['district'] == d]
        ax4.plot(temp['date'], temp['cases'], label=d)

    ax4.legend()
    st.pyplot(fig4)
    plt.close(fig4)

    # =========================
    # DATA TABLE
    # =========================
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered.tail(20))

    # =========================
    # DOWNLOAD
    # =========================
    st.subheader("Download Data")

    st.download_button(
        label="Download Filtered Dataset",
        data=filtered.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

if __name__ == '__main__':
    main()