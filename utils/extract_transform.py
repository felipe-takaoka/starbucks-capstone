import streamlit as st
import pandas as pd
import numpy as np


@st.cache
def loadAndCleanPortfolio():
  """ Load and clean portfolio data
  """

  portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
  
  # Available channels
  channels = ["email", "mobile", "social", "web"]

  # Create a one hot encoding for channel type
  portfolioWithChannelsEncoded = portfolio.copy()
  for channel in channels:
    portfolioWithChannelsEncoded[channel] = portfolio["channels"].apply(lambda cs: 1*(channel in cs))

  portfolio_df = portfolioWithChannelsEncoded.drop(columns="channels")

  # Rename and order columns
  portfolio_df = portfolio_df.rename(columns={"id": "offer_id"})
  cols = ["offer_id","offer_type","difficulty","reward","duration","email","mobile","social","web"]
  assert set(cols) == set(portfolio_df.columns)
  portfolio_df = portfolio_df[cols]

  return portfolio_df


@st.cache
def loadAndCleanProfile():
  """ Load and clean profile data
  """

  profile = pd.read_json('data/profile.json', orient='records', lines=True)

  profile_df = profile.copy()

  # Convert date to an int
  profile_df["became_member_on"] = pd.to_datetime(profile["became_member_on"].astype(str)).astype(int)
  # Get dummies for gender
  profile_df = pd.concat([profile_df, pd.get_dummies(profile_df["gender"])], axis=1)
  # Rename and order columns
  profile_df = profile_df.rename(columns={"id": "person"})
  profile_df = profile_df[["person", "age", "income", "became_member_on", "F", "M"]]

  return profile_df


@st.cache
def loadAndCleanTranscript():
  """ Load and clean transcript data
  """

  transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

  transcript_temp = transcript.copy()

  # Create event number per person
  transcript_temp = transcript_temp.sort_values(["person", "time"])
  transcript_temp["event_no"] = transcript_temp.groupby("person").cumcount() + 1

  # Explode value column to a list of key-values pairs
  transcript_temp["aux"] = transcript_temp["value"].apply(lambda d: d.items())
  transcript_temp = transcript_temp.explode("aux")

  # Extract key and value for each exploded row
  transcript_temp["k"] = transcript_temp["aux"].str[0]
  transcript_temp["v"] = transcript_temp["aux"].str[1]

  # Pivot
  transcript_df = transcript_temp.pivot(
      index=["person", "event_no", "event", "time"],
      columns="k",
      values="v"
    )

  # Reset index
  transcript_df = transcript_df.reset_index()
  transcript_df.columns.name = None

  # Clean up values
  transcript_df["offer_id"] = transcript_df["offer_id"].combine_first(transcript_df["offer id"])
  transcript_df = transcript_df.drop(columns="offer id")
  transcript_df["amount"] = pd.to_numeric(transcript_df["amount"])

  # Fill amount and reward with 0 when nan for ease of manipulation
  transcript_df["amount"] = transcript_df["amount"].fillna(0)
  transcript_df["reward"] = transcript_df["reward"].fillna(0)

  return transcript_df
