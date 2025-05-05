import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from PIL import Image  # Import PIL for image handling

# --- Data Loading & Cleaning ---
@st.cache_data
def load_and_merge(
    loan_file_path: str = 'Comp Prestamos.xlsx',
    chargeoff_file_path: str = 'prest_chargeoff.csv'
) -> pd.DataFrame:
    # Load loan balances
    loan_excel = pd.ExcelFile(loan_file_path)
    loan_df = loan_excel.parse('Sheet1')
    loan_df.columns = loan_df.columns.str.strip()
    loan_df['Trimestre'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(loan_df['Trimestre'], unit='D')

    # Load charge-offs
    chargeoff_df = pd.read_csv(chargeoff_file_path)
    chargeoff_df.columns = chargeoff_df.columns.str.strip()
    chargeoff_df['Trimestre'] = (
        chargeoff_df['Trimestre']
        .str.replace(r'[\$,]', '', regex=True)
        .astype(float)
    )
    chargeoff_df['Trimestre'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(chargeoff_df['Trimestre'], unit='D')
    monetary_cols = ['Personales', 'Restructurados', 'Mastercard', 'Automóviles', 'Hipotecarios']
    def clean_money(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        s = s.str.replace(r'[\$,()]', '', regex=True)
        s = s.str.strip().replace(['-', ' ', '', 'nan', 'NaN', None], '0', regex=True)
        return s.astype(float)
    for col in monetary_cols:
        chargeoff_df[col] = clean_money(chargeoff_df[col])

    # Merge and compute loss rates
    merged = pd.merge(loan_df, chargeoff_df, on='Trimestre', suffixes=('_Balance', '_ChargeOff'))
    loan_types = {
        'Personales_Balance': 'Personales_ChargeOff',
        'Restructurados_Balance': 'Restructurados_ChargeOff',
        'Master Card': 'Mastercard',
        'Automóviles_Balance': 'Automóviles_ChargeOff',
        'Hipotecarios_Balance': 'Hipotecarios_ChargeOff'
    }
    for bal_col, co_col in loan_types.items():
        lr_col = f"{bal_col.replace('_Balance','')}_LossRate"
        merged[lr_col] = merged[co_col] / merged[bal_col] * 100
    return merged

@st.cache_data
def load_reserves(reserve_file_path: str = 'Reserve quarter.csv') -> pd.DataFrame:
    # Load reserves CSV
    df = pd.read_csv(reserve_file_path)
    # Ensure year filled
    df['Year'] = pd.to_numeric(df['Año'], errors='coerce').ffill().astype(int)
    # Extract quarter number and forward-fill any missing
    q = df['Quarter'].str.extract(r"(\d+)")[0].astype(float)
    df['QuarterNum'] = q.ffill().astype(int)
    # Map quarter to first month
    month_map = {1: 1, 2: 4, 3: 7, 4: 10}
    df['Trimestre'] = pd.to_datetime({
        'year': df['Year'],
        'month': df['QuarterNum'].map(month_map),
        'day': 1
    })
    # Numeric reserves
    df['Reserves'] = pd.to_numeric(df['Cargos Netos a la Reserva'], errors='coerce').fillna(0)
    return df[['Trimestre', 'Reserves']]

# --- Main Dashboard ---
def main():
    st.set_page_config(page_title="Coop Bank Loan Analytics", layout="wide")
    
    # --- ADD LOGO HERE ---
    # Create a container for the header with logo and title
    header_container = st.container()
    with header_container:
        cols = st.columns([1, 3])
        with cols[0]:
            # Add the logo image
            try:
                logo = Image.open('logo2x_gray-2.png')  # Replace with your logo file path
                st.image(logo, width=150)  # Adjust width as needed
            except FileNotFoundError:
                st.error("Logo file not found. Please place 'logo.png' in your project directory.")
        with cols[1]:
            st.title("Loan Analytics Dashboard")
    
    # Load data
    data = load_and_merge()
    reserves = load_reserves()

    # Custom CSS to change the color of the multiselect options to #225e38
    st.markdown("""
    <style>
    /* Change the color of the multiselect filter button */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #225e38 !important;
    }
    /* Change the text color to white for better contrast on green background */
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar filters
    st.sidebar.header("Filters")
    
    loan_map = {
        "Personales": {"bal": "Personales_Balance", "co": "Personales_ChargeOff", "lr": "Personales_LossRate"},
        "Restructurados": {"bal": "Restructurados_Balance", "co": "Restructurados_ChargeOff", "lr": "Restructurados_LossRate"},
        "Credit Cards": {"bal": "Master Card", "co": "Mastercard", "lr": "Master Card_LossRate"},
        "Automóviles": {"bal": "Automóviles_Balance", "co": "Automóviles_ChargeOff", "lr": "Automóviles_LossRate"},
        "Hipotecarios": {"bal": "Hipotecarios_Balance", "co": "Hipotecarios_ChargeOff", "lr": "Hipotecarios_LossRate"}
    }
    loan_types = list(loan_map.keys())
    selected = st.sidebar.multiselect("Loan Types", loan_types, default=loan_types)

    # Date range filter
    min_date = data['Trimestre'].min().date()
    max_date = data['Trimestre'].max().date()
    dr = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    df = data[(data['Trimestre'].dt.date >= dr[0]) & (data['Trimestre'].dt.date <= dr[1])]
    res_df = reserves[(reserves['Trimestre'].dt.date >= dr[0]) & (reserves['Trimestre'].dt.date <= dr[1])]

    # --- Reserves Overview ---
    latest_q = df['Trimestre'].max()
    # find last reserves record on or before latest_q
    valid = res_df[res_df['Trimestre'] <= latest_q]
    if not valid.empty:
        last_q = valid['Trimestre'].max()
        latest_res = valid.loc[valid['Trimestre'] == last_q, 'Reserves'].iloc[0]
        q_num = last_q.quarter
        y_num = last_q.year
        st.metric(f"Net Reserves (Q{q_num} {y_num})", f"${latest_res:,.0f}")
    else:
        st.metric("Net Reserves", "N/A")

    # KPI Section
    st.header("Key Performance Indicators")
    kpis = st.columns(len(selected) * 2)
    latest = df[df['Trimestre'] == latest_q].iloc[0]
    for i, lt in enumerate(selected):
        bal = latest[loan_map[lt]['bal']]
        lr = latest[loan_map[lt]['lr']]
        kpis[2*i].metric(f"{lt} Balance", f"${bal:,.0f}")
        kpis[2*i+1].metric(f"{lt} Loss Rate", f"{lr:.2f}%")

    # Trend Lines
    st.header("Loss Rate Trends")
    trend_df = pd.concat([
        df[['Trimestre', loan_map[lt]['lr']]].rename(columns={loan_map[lt]['lr']: 'LossRate'}).assign(LoanType=lt)
        for lt in selected
    ])
    fig1 = px.line(trend_df, x='Trimestre', y='LossRate', color='LoanType',
                   labels={'LossRate': 'Loss Rate (%)', 'Trimestre': 'Quarter'})
    st.plotly_chart(fig1, use_container_width=True)

    # Forecasting
    st.header("Charge-Off Forecast")
    horiz = st.slider("Quarters to Forecast", 1, 8, 4)
    for lt in selected:
        ts = data[['Trimestre', loan_map[lt]['co']]].rename(columns={'Trimestre': 'ds', loan_map[lt]['co']: 'y'})
        model = Prophet()
        model.fit(ts)
        future = model.make_future_dataframe(periods=horiz, freq='Q')
        fc = model.predict(future)
        figf = px.line(fc, x='ds', y='yhat', labels={'yhat': 'Charge-Off', 'ds': 'Quarter'},
                       title=f"{lt} Forecast")
        st.plotly_chart(figf, use_container_width=True)

   

if __name__ == '__main__':
    main()