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
  st.write("Note that customers can complelte an offer without ever viewing it.")

  st.subheader("Sent Offers Distribution (deviation from uniform distribution)")
  offer_dist = getOffersDist(transcript_df, portfolio_df)
  fig = go.Figure(data=
    go.Bar(x=offer_dist["size"], y=offer_dist["code"], orientation="h")
  )
  delta_max = offer_dist["size"].abs().max() + .01
  fig.update_layout(xaxis={
    "tickformat": ',.0%',
    "range": [-delta_max, delta_max]
  })
  st.plotly_chart(fig)

  st.subheader("Data")
  st.write(promo_funnel)

elif page == "Customer Timeline":
  pass

elif page == "Feature Engineering":
  st.subheader("Features Engineering")
  transcript_feats = cachedCreateTranscriptFeatures(transcript_df, portfolio_df, profile_df)
  st.write(transcript_feats.head(50))

  st.subheader("Targets (Spendings)")
  Y_df = cachedCreateTargets(transcript_feats, portfolio_df)
  st.write(Y_df.head(50))
