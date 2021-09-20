import streamlit as st
import pandas as pd
import numpy as np
from utils.extract_transform import *
from utils.charts import *

# Load and clean dataframes
portfolio_df = cachedLoadAndCleanPortfolio()
profile_df = cachedLoadAndCleanProfile()
transcript_df = cachedLoadAndCleanTranscript()

pages = ["Offers Portfolio", "Demographic Groups", "Feature Engineering"]
page = st.sidebar.radio("Select page", pages)

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
  profile = cachedLoadAndCleanProfile(raw=True)
  demographics = createDemographicGroups(profile)

  st.subheader("Distribution of Demographic Groups")

  col1, col2 = st.columns(2)
  demog_feat = col2.radio("Demographic Feature", ["Age","Income","Cohort","Gender"])
  col1.pyplot(demographicDistributionBarH(demographics, demog_feat))

  if demog_feat != "Gender":
    st.plotly_chart(demographicDistributionHist(demographics, demog_feat))

  st.subheader("Data")
  st.write(demographics.head())

elif page == "Feature Engineering":
  st.subheader("Features Engineering")
  transcript_feats = cachedCreateTranscriptFeatures(transcript_df, portfolio_df, profile_df)
  st.write(transcript_feats.head(50))

  st.subheader("Targets (Spendings)")
  Y_df = cachedCreateTargets(transcript_feats, portfolio_df)
  st.write(Y_df.head(50))
