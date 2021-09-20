import streamlit as st
import pandas as pd
import numpy as np
from utils.extract_transform import *
from utils.charts import *

# Load and clean dataframes
portfolio_df = cachedLoadAndCleanPortfolio()
profile, profile_df = cachedLoadAndCleanProfile(return_raw=True)
transcript_df = cachedLoadAndCleanTranscript()

# Create demographic groups
demographics = createDemographicGroups(profile)
# Feature engineering
transcript_feats = cachedCreateTranscriptFeatures(transcript_df, portfolio_df, profile_df)
# Create target
Y_df = cachedCreateTargets(transcript_feats, portfolio_df)
# Create full dataset for model fitting
df_full, df = getTrainingDataset(transcript_feats, Y_df, return_df_full=True)

# Page options
pages = [
  "Offers Portfolio",
  "Demographic Groups",
  "Offer Responsiveness - Descriptive Approach",
  "Feature Engineering"
]
page = st.sidebar.radio("Select page", pages)

# Page contents
st.title(page)
if page == "Offers Portfolio":
  promo_funnel = getPromoFunnel(transcript_df, portfolio_df)

  st.subheader("Offer Funnel")
  st.plotly_chart(promoFunnelFig(promo_funnel))
  st.write("Note that customers can complelte an offer without ever viewing it.")

  st.subheader("Sent Offers Distribution (deviation from uniform distribution)")
  offer_dist = getOffersDist(transcript_df, portfolio_df)
  st.plotly_chart(sentOffersDistributionFig(offer_dist))

  st.subheader("Data")
  st.write(promo_funnel)

elif page == "Demographic Groups":
  st.subheader("Distribution of Demographic Groups")

  col1, col2 = st.columns(2)
  demog_feat = col2.radio("Demographic Feature", ["Age","Income","Cohort","Gender"])
  col1.pyplot(demographicDistributionBarH(demographics, demog_feat))

  if demog_feat != "Gender":
    st.plotly_chart(demographicDistributionHist(demographics, demog_feat))

  st.subheader("Data")
  st.write(demographics.head(50))

elif page == "Offer Responsiveness - Descriptive Approach":
  time_windows = sorted(24*portfolio_df["duration"].unique())
  spendings = createSpendingsPerGroup(df_full, demographics, time_windows)
  st.write(spendings)

elif page == "Feature Engineering":
  st.subheader("Features Engineering")  
  st.write(transcript_feats.head(50))

  st.subheader("Targets (Spendings)")
  st.write(Y_df.head(50))
