# Recommender System - Streamlit App for Distributor Recommendations

import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------------------
# 1. LOAD DATA
# ------------------------------
@st.cache_data

def load_data():
    prev_df = pd.read_excel(r"D:\OneDrive - Nilons Enterprises Pvt Ltd\Desktop\Anadi\Data\YTD 2024-2025 NC_E.xlsx")
    curr_df = pd.read_excel(r"D:\OneDrive - Nilons Enterprises Pvt Ltd\Desktop\Anadi\Data\SAP25_apr1st_may31st_E.xlsx")

    prev_df.columns = prev_df.columns.str.strip()
    curr_df.columns = curr_df.columns.str.strip()

    prev_df['Invoice Date'] = pd.to_datetime(prev_df['Invoice Date'])
    curr_df['Invoice Date'] = pd.to_datetime(curr_df['Invoice Date'])

    prev_df.rename(columns={'Distributor Code': 'Distributor','Billing Amount': 'Bill Amount'}, inplace=True)
    curr_df.rename(columns={'C. No': 'Distributor','C. Name': 'Distributor Name','C. Area': 'Area'}, inplace=True)

    return prev_df, curr_df

prev_df, curr_df = load_data()

# ------------------------------
# 2. PREPROCESSING & FEATURE ENGINEERING
# ------------------------------

def prepare_features(prev_df, curr_df):
    # Mappings
    channel_map = {'EXP': 'Export','RL': 'Institutional','INST': 'Institutional','GT': 'GT','MT': 'MT','PL': 'Private Label','SMT': 'SMT','GOVT': 'Command','E-COM': 'E-Commerce','GT HO': 'Horeca'}
    category_map = {'VERMICELLI-ROASTED': 'ROAST VERMICELLI','VERMICELLI-CUT': 'CUT VERMICELLI','TOOTY FRUTI': 'TOOTY FRUITY','RE 1 & 2': 'PICKLE-RE 1&2','BLENDED - WESTERN': 'SPICE-WESTERN BLEND','BLENDED - INDIAN': 'SPICE-INDIAN BLEND','SPICES-BLENDED': 'SPICE-BASIC','SPICES-CTC': 'SPICE-CTC','SPICES-RTC': 'SPICE-RTC'}
    area_map = {'ORISSA': 'ODISHA','UTTRAKHAND': 'UTTARAKHAND','BAREILY': 'BAREILLY','PUNE & GOA': 'PUNE','GUJARAT-RAJKOT': 'RAJKOT','GUJARAT-AHMEDABAD': 'AHMEDABAD','CHHATISGARH': 'CHHATTISGARH','BIHAR-MUZAFFARPUR (N)': 'NORTH BIHAR','BIHAR-MUZAFFARPUR (J)': 'SOUTH BIHAR','BIHAR-PATNA': 'NORTH BIHAR','ROM 1': 'ROM','MODERN TRADE': 'MT','PRIVATE LABLE': 'Private Label'}

    prev_df['Distribution Channel'] = prev_df['Distribution Channel'].replace(channel_map)
    prev_df['Category'] = prev_df['Category'].replace(category_map)
    prev_df['Area'] = prev_df['Area'].replace(area_map)

    prev_df['Distributor'] = prev_df['Distributor'].astype(str)
    curr_df['Distributor'] = curr_df['Distributor'].astype(str)

    this_month = pd.to_datetime("2025-05-01")
    this_month_str = this_month.strftime('%Y-%m')
    last_year_same_month = (this_month.replace(year=this_month.year - 1)).strftime('%Y-%m')
    last_3_months = [(this_month.replace(month=m)).strftime('%Y-%m') for m in [3, 4, 5]]

    prev_df['period'] = prev_df['Invoice Date'].dt.to_period('M').astype(str)
    curr_df['period'] = curr_df['Invoice Date'].dt.to_period('M').astype(str)

    target = curr_df[curr_df['period'] == this_month_str][['Distributor', 'Item Code']].copy()
    target['label'] = 1

    last_year_sales = prev_df[prev_df['period'] == last_year_same_month][['Distributor', 'Item Code']]
    recent_sales = curr_df[curr_df['period'].isin(last_3_months)][['Distributor', 'Item Code']]
    candidates = pd.concat([last_year_sales, recent_sales]).drop_duplicates()

    target['Distributor'] = target['Distributor'].astype(str)
    candidates['Distributor'] = candidates['Distributor'].astype(str)

    data = candidates.merge(target, on=['Distributor', 'Item Code'], how='left')
    data['label'] = data['label'].fillna(0)

    freq = curr_df[curr_df['period'].isin(last_3_months)].groupby(['Distributor', 'Item Code']).agg(
        times_sold_last_3=('Invoice Date', 'count'),
        qty_last_3=('Bill Amount', 'sum'),
        last_sold=('Invoice Date', 'max')
    ).reset_index()
    freq['Distributor'] = freq['Distributor'].astype(str)

    seasonal = prev_df[prev_df['period'] == last_year_same_month][['Distributor', 'Item Code']]
    seasonal['sold_last_year_same_month'] = 1

    features = data.merge(freq, on=['Distributor', 'Item Code'], how='left')
    features = features.merge(seasonal, on=['Distributor', 'Item Code'], how='left')

    features['sold_last_year_same_month'] = features['sold_last_year_same_month'].fillna(0)
    features['times_sold_last_3'] = features['times_sold_last_3'].fillna(0)
    features['qty_last_3'] = features['qty_last_3'].fillna(0)
    features['last_sold'] = pd.to_datetime(features['last_sold'], errors='coerce')
    features['days_since_last_sold'] = (this_month - features['last_sold']).dt.days.fillna(999)

    # Add Item Name for display
    item_lookup = curr_df[['Item Code', 'Item Name']].drop_duplicates()
    features = features.merge(item_lookup, on='Item Code', how='left')

    return features

features = prepare_features(prev_df, curr_df)

# ------------------------------
# 3. TRAIN MODEL
# ------------------------------

X = features[['sold_last_year_same_month', 'times_sold_last_3', 'qty_last_3', 'days_since_last_sold']]
y = features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

features['probability'] = model.predict_proba(X)[:, 1]

# ------------------------------
# 4. STREAMLIT APP UI
# ------------------------------

st.title("📦 Product Recommendation for Distributors")

# Merge distributor names for display
curr_distributor_map = curr_df[['Distributor', 'Distributor Name']].drop_duplicates()
curr_distributor_map['Distributor'] = curr_distributor_map['Distributor'].astype(str)
curr_distributor_map['Distributor Name'] = curr_distributor_map['Distributor Name'].astype(str)
curr_distributor_map['display'] = curr_distributor_map['Distributor Name'] + ' [' + curr_distributor_map['Distributor'] + ']'
distributor_lookup = dict(zip(curr_distributor_map['display'], curr_distributor_map['Distributor']))

selected_display = st.selectbox("Select a Distributor", sorted(distributor_lookup.keys()))
selected_distributor = distributor_lookup[selected_display]

filtered = features[(features['Distributor'] == selected_distributor) & (features['label'] == 0)]

top_recs = filtered.sort_values(by='probability', ascending=False).head(10)
st.subheader("🔍 Recommended Products")
st.dataframe(top_recs[['Item Code', 'Item Name', 'probability', 'sold_last_year_same_month', 'times_sold_last_3', 'days_since_last_sold']])
