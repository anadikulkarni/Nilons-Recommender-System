import streamlit as st
import pandas as pd
from datetime import datetime

st.title("🔍 Product Gap Highlighter")

@st.cache_data

def load_data():
    prev_df = pd.read_excel(r"D:\OneDrive - Nilons Enterprises Pvt Ltd\Desktop\Anadi\Data\YTD 2024-2025 NC_E.xlsx")
    curr_df = pd.read_excel(r"D:\OneDrive - Nilons Enterprises Pvt Ltd\Desktop\Anadi\Data\SAP25_apr1st_may31st_E.xlsx")
    
    prev_df.columns = prev_df.columns.str.strip()
    curr_df.columns = curr_df.columns.str.strip()
    
    prev_df['Invoice Date'] = pd.to_datetime(prev_df['Invoice Date'])
    curr_df['Invoice Date'] = pd.to_datetime(curr_df['Invoice Date'])
    
    prev_df.rename(columns={'Billing Amount': 'Bill Amount'}, inplace=True)
    curr_df.rename(columns={'C. No': 'Distributor Code','C. Name': 'Distributor Name','C. Area': 'Area'}, inplace=True)
    
    return prev_df, curr_df

# Load and preprocess
data_load_state = st.text('Loading data...')
prev_df, curr_df = load_data()
data_load_state.text('Loading data... done!')

channel_map = {'EXP': 'Export','RL': 'Institutional','INST': 'Institutional','GT': 'GT','MT': 'MT','PL': 'Private Label','SMT': 'SMT','GOVT': 'Command','E-COM': 'E-Commerce','GT HO': 'Horeca'}
category_map = {'VERMICELLI-ROASTED': 'ROAST VERMICELLI','VERMICELLI-CUT': 'CUT VERMICELLI','TOOTY FRUTI': 'TOOTY FRUITY','RE 1 & 2': 'PICKLE-RE 1&2','BLENDED - WESTERN': 'SPICE-WESTERN BLEND','BLENDED - INDIAN': 'SPICE-INDIAN BLEND','SPICES-BLENDED': 'SPICE-BASIC','SPICES-CTC': 'SPICE-CTC','SPICES-RTC': 'SPICE-RTC'}
area_map = {'ORISSA': 'ODISHA','UTTRAKHAND': 'UTTARAKHAND','BAREILY': 'BAREILLY','PUNE & GOA': 'PUNE','GUJARAT-RAJKOT': 'RAJKOT','GUJARAT-AHMEDABAD': 'AHMEDABAD','CHHATISGARH': 'CHHATTISGARH','BIHAR-MUZAFFARPUR (N)': 'NORTH BIHAR','BIHAR-MUZAFFARPUR (J)': 'SOUTH BIHAR','BIHAR-PATNA': 'NORTH BIHAR','ROM 1': 'ROM','MODERN TRADE': 'MT','PRIVATE LABLE': 'Private Label'}

prev_df['Distribution Channel'] = prev_df['Distribution Channel'].replace(channel_map)
prev_df['Category'] = prev_df['Category'].replace(category_map)
prev_df['Area'] = prev_df['Area'].replace(area_map)

prev_df['Distributor Code'] = prev_df['Distributor Code'].astype(str)
curr_df['Distributor Code'] = curr_df['Distributor Code'].astype(str)

today = pd.to_datetime("2025-05-01")  # Assume current date is May 2025
last_month = (today.replace(day=1) - pd.DateOffset(days=1)).strftime('%Y-%m')  # Apr 2025
last_year_same_month = (today.replace(year=today.year - 1)).strftime('%Y-%m')  # May 2024

prev_df['period'] = prev_df['Invoice Date'].dt.to_period('M').astype(str)
curr_df['period'] = curr_df['Invoice Date'].dt.to_period('M').astype(str)

# Products sold last year same month
last_year_sales = prev_df[prev_df['period'] == last_year_same_month][['Distributor Code', 'Item Code']].drop_duplicates()
last_year_sales['sold_last_year_same_month'] = 1

# Products sold last month
last_month_sales = curr_df[curr_df['period'] == last_month][['Distributor Code', 'Item Code']].drop_duplicates()
last_month_sales['sold_last_month'] = 1

# Merge to detect gaps
gap_check = last_year_sales.merge(last_month_sales, on=['Distributor Code', 'Item Code'], how='left')
gap_check['sold_last_month'] = gap_check['sold_last_month'].fillna(0)

# Highlight products sold last year same month but NOT last month
gaps = gap_check[gap_check['sold_last_month'] == 0].copy()

# Add Item Name for reference
item_lookup = pd.concat([prev_df[['Item Code', 'Item Name']], curr_df[['Item Code', 'Item Name']]]).drop_duplicates()
gaps = gaps.merge(item_lookup, on='Item Code', how='left')

# UI

# Restrict mapping to only distributors that exist in gaps
used_distributors = gaps[['Distributor Code']].drop_duplicates()
dist_map = pd.concat([
    curr_df[['Distributor Code', 'Distributor Name']],
    prev_df[['Distributor Code', 'Distributor Name']]
]).drop_duplicates()
dist_map['Distributor Code'] = dist_map['Distributor Code'].astype(str)
dist_map['Distributor Name'] = dist_map['Distributor Name'].astype(str)
dist_map = used_distributors.merge(dist_map, on='Distributor Code', how='left')
dist_map['Distributor Name'] = dist_map['Distributor Name'].fillna('Unknown')
dist_map['display'] = dist_map['Distributor Name'] + ' [' + dist_map['Distributor Code'] + ']'
display_to_code = dict(zip(dist_map['display'], dist_map['Distributor Code']))

selected_display = st.selectbox("Select Distributor", sorted(display_to_code.keys()))
selected_dist = display_to_code[selected_display]

result = (
    gaps[gaps['Distributor Code'] == selected_dist][['Item Code', 'Item Name']]
    .sort_values('Item Name')  # optional: sort for consistent results
    .drop_duplicates(subset='Item Code')
)

st.subheader("🛒 Gap Products for Distributor")
st.dataframe(result)

if result.empty:
    st.info("No gap products for this distributor.")
