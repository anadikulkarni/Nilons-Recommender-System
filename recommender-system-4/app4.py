import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Distributor Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data(prev_df, curr_df):
    """Prepare and clean the data for analysis"""
    prev_df_clean = prev_df.copy()
    curr_df_clean = curr_df.copy()
    
    # Standardize column names
    column_mapping = {
        'Channel': 'Distribution Channel',
        'CATEGORY': 'Material Group', 
        'VARIANTS': 'Material Sub Group',
        'Qty': 'Invoice Qty',
        'Billing Amount': 'Bill Amount'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in prev_df_clean.columns:
            prev_df_clean = prev_df_clean.rename(columns={old_col: new_col})
    
    # Convert data types
    prev_df_clean['Invoice Date'] = pd.to_datetime(prev_df_clean['Invoice Date'])
    curr_df_clean['Invoice Date'] = pd.to_datetime(curr_df_clean['Invoice Date'])
    prev_df_clean['C. No'] = prev_df_clean['C. No'].astype(str)
    prev_df_clean['Item Code'] = prev_df_clean['Item Code'].astype(str)
    curr_df_clean['C. No'] = curr_df_clean['C. No'].astype(str)
    curr_df_clean['Item Code'] = curr_df_clean['Item Code'].astype(str)
    
    # Add time-based columns
    prev_df_clean['Year_Month'] = prev_df_clean['Invoice Date'].dt.to_period('M')
    curr_df_clean['Year_Month'] = curr_df_clean['Invoice Date'].dt.to_period('M')
    prev_df_clean['Month_Name'] = prev_df_clean['Invoice Date'].dt.month_name()
    curr_df_clean['Month_Name'] = curr_df_clean['Invoice Date'].dt.month_name()
    
    return prev_df_clean, curr_df_clean

@st.cache_data
def calculate_recommendation_scores(prev_df, curr_df):
    """Calculate recommendation scores based on seasonality and trends"""
    
    # Get May 2024 data
    may_2024_data = prev_df[prev_df['Month_Name'] == 'May'].groupby(['C. No', 'C. Name', 'Item Code', 'Item Name']).agg({
        'Invoice Qty': 'sum',
        'Bill Amount': 'sum',
        'Gross Weight': 'sum'
    }).reset_index()
    may_2024_data.columns = ['C. No', 'C. Name', 'Item Code', 'Item Name', 'May_2024_Qty', 'May_2024_Amount', 'May_2024_Weight']
    
    # Get April-May 2025 data
    recent_data = curr_df.groupby(['C. No', 'C. Name', 'Item Code', 'Item Name']).agg({
        'Invoice Qty': 'sum',
        'Bill Amount': 'sum',
        'Gross Weight': 'sum'
    }).reset_index()
    recent_data.columns = ['C. No', 'C. Name', 'Item Code', 'Item Name', 'Recent_Qty', 'Recent_Amount', 'Recent_Weight']
    recent_data['Avg_Monthly_Recent_Qty'] = recent_data['Recent_Qty'] / 2
    
    # Merge datasets
    recommendation_df = pd.merge(may_2024_data, recent_data, 
                                on=['C. No', 'C. Name', 'Item Code', 'Item Name'], 
                                how='outer').fillna(0)
    
    # Calculate scores
    recommendation_df['Seasonality_Ratio'] = np.where(
        recommendation_df['Avg_Monthly_Recent_Qty'] > 0,
        recommendation_df['May_2024_Qty'] / recommendation_df['Avg_Monthly_Recent_Qty'],
        np.where(recommendation_df['May_2024_Qty'] > 0, 5, 0)  # High score if only May 2024 has sales
    )
    
    recommendation_df['Growth_Potential'] = recommendation_df['May_2024_Qty'] - recommendation_df['Avg_Monthly_Recent_Qty']
    recommendation_df['Value_Score'] = recommendation_df['May_2024_Amount'] / (recommendation_df['May_2024_Qty'] + 1)
    
    # Normalize scores
    if recommendation_df['Seasonality_Ratio'].std() > 0:
        recommendation_df['Normalized_Seasonality'] = (recommendation_df['Seasonality_Ratio'] - recommendation_df['Seasonality_Ratio'].mean()) / recommendation_df['Seasonality_Ratio'].std()
    else:
        recommendation_df['Normalized_Seasonality'] = 0
        
    if recommendation_df['Growth_Potential'].std() > 0:
        recommendation_df['Normalized_Growth'] = (recommendation_df['Growth_Potential'] - recommendation_df['Growth_Potential'].mean()) / recommendation_df['Growth_Potential'].std()
    else:
        recommendation_df['Normalized_Growth'] = 0
    
    # Final score
    max_value = recommendation_df['Value_Score'].max() if recommendation_df['Value_Score'].max() > 0 else 1
    recommendation_df['Final_Score'] = (
        0.6 * recommendation_df['Normalized_Seasonality'] + 
        0.3 * recommendation_df['Normalized_Growth'] + 
        0.1 * (recommendation_df['Value_Score'] / max_value)
    )
    
    return recommendation_df

@st.cache_data
def analyze_distributor_performance(prev_df, curr_df):
    """Analyze distributor performance patterns"""
    
    # Previous year performance
    prev_perf = prev_df.groupby(['C. No', 'C. Name', 'C. Area']).agg({
        'Invoice Qty': 'sum',
        'Bill Amount': 'sum',
        'Item Code': 'nunique',
        'Invoice Date': 'count'
    }).reset_index()
    prev_perf.columns = ['C. No', 'C. Name', 'C. Area', 'Prev_Total_Qty', 'Prev_Total_Amount', 'Prev_Unique_Products', 'Prev_Transactions']
    
    # Current year performance
    curr_perf = curr_df.groupby(['C. No', 'C. Name', 'C. Area']).agg({
        'Invoice Qty': 'sum',
        'Bill Amount': 'sum',
        'Item Code': 'nunique',
        'Invoice Date': 'count'
    }).reset_index()
    curr_perf.columns = ['C. No', 'C. Name', 'C. Area', 'Curr_Total_Qty', 'Curr_Total_Amount', 'Curr_Unique_Products', 'Curr_Transactions']
    
    # Merge and calculate metrics
    distributor_analysis = pd.merge(prev_perf, curr_perf, on=['C. No', 'C. Name', 'C. Area'], how='outer').fillna(0)
    
    # Calculate growth rates (annualized current performance)
    distributor_analysis['Annualized_Curr_Qty'] = distributor_analysis['Curr_Total_Qty'] * 6
    distributor_analysis['Qty_Growth_Rate'] = ((distributor_analysis['Annualized_Curr_Qty'] - distributor_analysis['Prev_Total_Qty']) / 
                                              (distributor_analysis['Prev_Total_Qty'] + 1)) * 100
    
    distributor_analysis['Annualized_Curr_Amount'] = distributor_analysis['Curr_Total_Amount'] * 6
    distributor_analysis['Amount_Growth_Rate'] = ((distributor_analysis['Annualized_Curr_Amount'] - distributor_analysis['Prev_Total_Amount']) / 
                                                  (distributor_analysis['Prev_Total_Amount'] + 1)) * 100
    
    # Performance categories
    def categorize_performance(growth_rate):
        if growth_rate > 20:
            return 'High Growth'
        elif growth_rate > 0:
            return 'Moderate Growth'
        elif growth_rate > -20:
            return 'Stable'
        else:
            return 'Declining'
    
    distributor_analysis['Performance_Category'] = distributor_analysis['Qty_Growth_Rate'].apply(categorize_performance)
    
    return distributor_analysis

def generate_personalized_recommendations(recommendation_df, distributor_analysis, selected_distributor, top_n=10):
    """Generate personalized recommendations for selected distributor"""
    
    # Filter for selected distributor
    distributor_recs = recommendation_df[recommendation_df['C. No'] == selected_distributor].copy()
    
    if len(distributor_recs) == 0:
        return None, None
    
    # Get distributor performance info
    dist_performance = distributor_analysis[distributor_analysis['C. No'] == selected_distributor]
    
    # Filter for meaningful recommendations
    meaningful_recs = distributor_recs[
        (distributor_recs['May_2024_Qty'] > 0) | 
        (distributor_recs['Recent_Qty'] > 0)
    ].copy()
    
    # Sort by final score and take top N
    top_recommendations = meaningful_recs.nlargest(top_n, 'Final_Score')
    
    return top_recommendations, dist_performance

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Distributor Recommendation System</h1>', unsafe_allow_html=True)
    
    prev_df = pd.read_excel(r"D:\OneDrive - Nilons Enterprises Pvt Ltd\Desktop\Anadi\Data\YTD 2024-2025 NC_E2.xlsx")
    curr_df = pd.read_excel(r"D:\OneDrive - Nilons Enterprises Pvt Ltd\Desktop\Anadi\Data\SAP_apr1st_may31st_NCE2.xlsx")
        
    # Load data
    try:
        # Process data
        with st.spinner("Processing data..."):
            prev_clean, curr_clean = load_and_prepare_data(prev_df, curr_df)
            recommendation_df = calculate_recommendation_scores(prev_clean, curr_clean)
            distributor_analysis = analyze_distributor_performance(prev_clean, curr_clean)
        
        # Display data overview
        st.success("✅ Data loaded successfully!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Previous Year Records", f"{len(prev_clean):,}")
        with col2:
            st.metric("Current Year Records", f"{len(curr_clean):,}")
        with col3:
            st.metric("Total Distributors", f"{prev_clean['C. No'].nunique():,}")
        with col4:
            st.metric("Total Products", f"{prev_clean['Item Code'].nunique():,}")
        
        # Distributor selector
        st.header("🏪 Select Distributor for Analysis")
        
        # Create distributor options with names
        # Only keep distributors that have recommendations
        valid_distributors = recommendation_df.groupby('C. No')['Final_Score'].sum().reset_index()
        valid_distributors = valid_distributors[valid_distributors['Final_Score'] > 0]['C. No']

        # Filter distributor_analysis to only those with non-zero scores
        distributor_options = distributor_analysis[distributor_analysis['C. No'].isin(valid_distributors)][['C. No', 'C. Name']].drop_duplicates()
        distributor_options['Display'] = distributor_options['C. Name'] + ' (' + distributor_options['C. No'] + ')'

        
        selected_display = st.selectbox(
            "Choose a distributor:",
            options=distributor_options['Display'].tolist(),
            help="Select a distributor to view personalized recommendations"
        )
        
        if selected_display:
            selected_distributor = distributor_options[distributor_options['Display'] == selected_display]['C. No'].iloc[0]
            selected_name = distributor_options[distributor_options['Display'] == selected_display]['C. Name'].iloc[0]
            
            # Generate recommendations for selected distributor
            dist_recommendations, dist_performance = generate_personalized_recommendations(
                recommendation_df, distributor_analysis, selected_distributor
            )
            
            if dist_recommendations is not None and len(dist_recommendations) > 0:
                
                # Distributor header
                st.markdown(f"## 📊 Analysis for {selected_name}")
                
                # Performance overview
                if len(dist_performance) > 0:
                    perf = dist_performance.iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Performance Category", perf['Performance_Category'])
                    with col4:
                        st.metric("Products Handled", f"{int(perf['Prev_Unique_Products'])}")
                
                # Product recommendations table
                st.header("🎯 Top Product Recommendations")
                
                # Filter and display top recommendations
                top_recs = dist_recommendations.head(10)
                
                for idx, row in top_recs.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{row['Item Name']} ({row['Item Code']})</h4>
                            <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                                <div>
                                    <strong>Seasonality Ratio:</strong> {row['Seasonality_Ratio']:.2f}x<br>
                                    <strong>May 2024 Sales:</strong> {row['May_2024_Qty']:.0f} units<br>
                                    <strong>Recent Avg/Month:</strong> {row['Avg_Monthly_Recent_Qty']:.0f} units
                                </div>
                                <div>
                                    <strong>Growth Potential:</strong> {row['Growth_Potential']:.0f} units<br>
                                    <strong>Value/Unit:</strong> ₹{row['Value_Score']:.2f}<br>
                                    <strong>Recommendation Score:</strong> {row['Final_Score']:.2f}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed data table
                st.header("📋 Detailed Recommendations Data")
                
                display_cols = ['Item Name', 'Item Code', 'Seasonality_Ratio', 'May_2024_Qty', 
                                'Avg_Monthly_Recent_Qty', 'Growth_Potential', 'Value_Score', 'Final_Score']
                
                display_df = top_recs[display_cols].copy()
                display_df.columns = ['Product Name', 'Product Code', 'Seasonality Ratio', 
                                        'May 2024 Qty', 'Recent Avg/Month', 'Growth Potential', 
                                        'Value/Unit', 'Recommendation Score']
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download recommendations
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations as CSV",
                    data=csv,
                    file_name=f"recommendations_{selected_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning(f"⚠️ No recommendations found for {selected_name}. This distributor may not have sufficient historical data.")
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.error("Please check your data format and try again.")


if __name__ == "__main__":
    main()