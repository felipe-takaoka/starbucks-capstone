import streamlit as st
import pandas as pd
import numpy as np
from utils.extract_transform import *
from utils.inference import *
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
  "Offer Responsiveness - Predictive Approach"
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
  demog_spendings, spendings = createSpendingsPerGroup(df_full, demographics, time_windows, return_raw=True)
  feat_cols = ["age_group", "income_group", "cohort_group", "gender", "offer_code"]

  st.subheader("Spendings per Demographic Feature")
  col1, col2 = st.columns(2)
  feat = col2.radio("Demographic Feature", feat_cols)
  col1.pyplot(spendingsPerDemographicsBar(demog_spendings, feat))

  st.subheader("Best Demographic Groups per Offer")
  st.write("Select the features for defining the demographic groups")
  col1, col2 = st.columns(2)
  with col1:
    cb_age = st.checkbox("Age", True, key="Age1")
    cb_income = st.checkbox("Income", True, key="Income1")
    offer_types = st.multiselect("Offer Types", portfolio_df["code"])
  with col2:
    cb_gender = st.checkbox("Gender", True, key="Gender1")
    cb_cohort = st.checkbox("Cohort", False, key="Cohort1")
    min_group_size = st.slider("Minimum Group Size", 0, 100, 30)

  demog_feats = zip(feat_cols[:-1], [cb_age, cb_income, cb_cohort, cb_gender])
  demog_feats = [f for f,use in demog_feats if use]

  spendings_offers = spendingsForOffers(demog_spendings, offer_types, demog_feats, min_group_size)
  spendings_offers = spendings_offers.style.bar(subset=["spending_median"], color="#F63366")
  st.write(spendings_offers)

  st.subheader("Best Offer per Demographic Group")
  col1, col2, col3, col4 = st.columns(4)
  group_def = []
  with col1:
    cb_age = st.checkbox("Age", True, key="Age2")
    age = st.selectbox("Age Group", spendings["age_group"].unique(), key="GroupsAge")
    if cb_age: group_def.append(("age_group", age))
  with col2:
    cb_income = st.checkbox("Income", True, key="Income2")
    income = st.selectbox("Income Group", spendings["income_group"].unique(), key="GroupsIncome")
    if cb_income: group_def.append(("income_group", income))
  with col3:
    cb_gender = st.checkbox("Gender", True, key="Gender2")
    gender = st.selectbox("Gender Group", spendings["gender"].unique(), key="GroupsGender")
    if cb_gender: group_def.append(("gender", gender))
  with col4:
    cb_cohort = st.checkbox("Cohort", False, key="Cohort2")
    cohort = st.selectbox("Cohort Group", spendings["cohort_group"].unique(), key="GroupsCohort")
    if cb_cohort: group_def.append(("cohort_group", cohort))

  metrics = getGroupStats(group_def, demographics, demog_spendings)
  n_customers, n_offers_sent, n_unique_offers = metrics
  col1, col2, col3 = st.columns(3)
  col1.metric("Customers in Group", n_customers)
  col2.metric("Offers Sent", n_offers_sent)
  col3.metric("Offer Types Sent", n_unique_offers)

  st.markdown("**Top 5 Offers to Send to Demographic Group**")
  best_offers = bestOfferForGroup(demog_spendings, portfolio_df, group_def)
  best_offers = best_offers.style.bar(subset=["spending_median"], color="#F63366")
  st.write(best_offers)


elif page == "Offer Responsiveness - Predictive Approach":
  models = loadModels()

  st.header("Spending Inference")
  person = st.selectbox("Select the customer", profile_df["person"])
  last_sim_time = int(transcript_df["time"].max())
  time = st.slider("Time to send offer", last_sim_time+1, 800)

  with st.expander("See Customer Timeline and Features"):
    st.subheader("Customer Timeline")
    st.write(getCustomerTimeline(transcript_df, person))

    st.subheader("Customer Features")
    customerFeats = getCustomerFeatures(person, time, transcript_feats, portfolio_df)
    st.write(dropAuxFeatures(customerFeats))

  st.subheader("Spending Predictions")
  customerSpendings = predictCustomerSpendings(customerFeats)
  customerSpendings = customerSpendings.style.background_gradient("rocket")
  st.write(customerSpendings)
