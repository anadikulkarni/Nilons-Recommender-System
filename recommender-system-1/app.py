import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules

# Load the saved basket and compute frequent items + rules
@st.cache_data
def load_rules():
    return pd.read_pickle("recommender-system-1\\rules.pkl")

# Load rules
rules = load_rules()

# Extract unique product names
all_products = sorted(set().union(*rules["antecedents"]))

# Title and instructions
st.title("🛍️ Nilons Product Co-Purchase Indicator")
st.markdown("Select a product to see what items are commonly bought with it.")

# Dropdown to select product
selected_product = st.selectbox("Choose a product:", all_products)

# Recommendation function
def recommend_products(product, rules_df, top_n=5):
    recommendations = rules_df[rules_df["antecedents"] == frozenset([product])]
    recommendations = recommendations.sort_values(by="lift", ascending=False)
    return recommendations[["consequents", "confidence", "lift"]].head(top_n)

# Display recommendations
if selected_product:
    st.subheader(f"🧠 Recommended products for: {selected_product}")
    recs = recommend_products(selected_product, rules)

    if not recs.empty:
        # Format frozensets as strings
        recs["consequents"] = recs["consequents"].apply(lambda x: ", ".join(list(x)))
        st.dataframe(recs.rename(columns={
            "consequents": "Recommended Product(s)",
            "confidence": "Confidence",
            "lift": "Lift"
        }))
    else:
        st.info("No strong recommendations found for this product yet.")
