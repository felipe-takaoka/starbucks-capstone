import streamlit as st
import pandas as pd
import numpy as np
from utils.extract_transform import *

portfolio_df = loadAndCleanPortfolio()
profile_df = loadAndCleanProfile()
transcript_df = loadAndCleanTranscript()

st.write(portfolio_df)
st.write(profile_df)
st.write(transcript_df.head(30))

transcript_feats = createTranscriptFeatures(transcript_df, portfolio_df, profile_df)
st.write(transcript_feats.head(50))