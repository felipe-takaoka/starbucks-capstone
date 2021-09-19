import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from utils.extract_transform import *

# Set the default plotly theme
pio.templates.default = "none"

# Load and clean dataframes
portfolio_df = cachedLoadAndCleanPortfolio()
profile_df = cachedLoadAndCleanProfile()
transcript_df = cachedLoadAndCleanTranscript()

pages = ["Offers Portfolio", "Customer Timeline", "Feature Engineering"]
page = st.sidebar.radio("Select page", pages)

st.title(page)

if page == "Offers Portfolio":
  promo_funnel = getPromoFunnel(transcript_df, portfolio_df)

  st.subheader("Offer Funnel")
  fig = go.Figure(data=[
    go.Bar(name=step, x=promo_funnel[step], y=promo_funnel["code"], orientation="h")
    for step in ["offer received", "offer viewed", "offer completed"]
  ])
  st.plotly_chart(fig)

  st.subheader("Data")
  st.write(promo_funnel)

elif page == "Customer Timeline":
  pass

elif page == "Feature Engineering":
  transcript_feats = createTranscriptFeatures(transcript_df, portfolio_df, profile_df)
  st.write(transcript_feats.head(50))