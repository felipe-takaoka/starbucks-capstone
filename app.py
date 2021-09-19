import streamlit as st
import pandas as pd
import numpy as np
from utils.extract_transform import *

portfolio_df = cachedLoadAndCleanPortfolio()
profile_df = cachedLoadAndCleanProfile()
transcript_df = cachedLoadAndCleanTranscript()

pages = ["Cleaned Inputs", "Customer Timeline", "Feature Engineering"]
page = st.sidebar.radio("Select page", pages)

if page == "Cleaned Inputs":
  st.write(portfolio_df)
  st.write(profile_df.head(50))
  st.write(transcript_df.head(50))
elif page == "Customer Timeline":
  st.title("Customer Timeline")
elif page == "Feature Engineering":
  transcript_feats = createTranscriptFeatures(transcript_df, portfolio_df, profile_df)
  st.write(transcript_feats.head(50))