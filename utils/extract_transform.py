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

  # Give a descriptive name
  portfolio_df["offer"] = portfolio_df["offer_type"].str[0] + "." + \
    portfolio_df["difficulty"].astype(str) + "."  + \
    portfolio_df["reward"].astype(str) + "."  + \
    portfolio_df["duration"].astype(str)

  # Rename and order columns and rows
  portfolio_df = portfolio_df.rename(columns={"id": "offer_id"})
  cols = ["offer_id","offer","offer_type","difficulty","reward","duration","email","mobile","social","web"]
  portfolio_df = portfolio_df[cols]
  portfolio_df = portfolio_df.sort_values(["offer_type", "difficulty", "reward", "duration"])

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


@st.cache
def createTranscriptFeatures(transcript_df, portfolio_df, profile_df):
  """ xxx
  """

  transcript_feats = transcript_df.copy()

  # Create dummy variables for the events (to perform a cumulative sum)
  transcript_feats = pd.concat([transcript_feats, pd.get_dummies(transcript_feats["event"])], axis=1)

  # Define the columns to aggregate, their function and their new name
  agg_cols = {
      # original column name: (agg function, agg column name)
      "event_no": ("max", "event_no"),
      "amount": ("sum", "spending"),
      "reward": ("sum", "cum_reward"),
      "transaction": ("sum", "transactions"),
      "offer received": ("sum", "offers_received"),
      "offer viewed": ("sum", "offers_viewed"),
      "offer completed": ("sum", "offers_completed"),
      "time": ("min", "min_time")
  }
  aggs = {k: v[0] for k, v in agg_cols.items()}
  rename = {k:v[1] for k,v in agg_cols.items()}

  # Perform the aggregations on the expanding window partitioned by each person
  transcript_aggs = transcript_feats.groupby("person").expanding().agg(aggs)
  transcript_aggs = transcript_aggs.rename(columns=rename)
  transcript_aggs = transcript_aggs.reset_index().drop(columns="level_1")
  transcript_feats = transcript_feats.merge(transcript_aggs, on=["person", "event_no"])

  # Subtract the current "event" so that they account only for the past (without information not available on inference time)
  cols_subtract = [col for col in agg_cols.keys() if col not in ["event_no", "time"]]
  cols_keep = [rename[col] for col in cols_subtract]
  transcript_feats.loc[:, cols_keep] -= transcript_feats.loc[:, cols_subtract].values

  # Time since each person's first event
  transcript_feats["time_customer"] = transcript_feats["time"] - transcript_feats["min_time"]
  transcript_feats = transcript_feats.drop(columns="min_time")

  # Average transaction value (up to that point)
  transcript_feats["atv"] = transcript_feats["spending"] / transcript_feats["transactions"]
  # Percentage of offers completed (completed / received) - up to that point
  transcript_feats["offer_usage"] = transcript_feats["offers_completed"] / transcript_feats["offers_received"]

  # Time since last transaction, offer received, offer viewed, and offer completed
  events = ["offer received", "offer viewed", "transaction", "offer completed"]
  transcript_feats = transcript_feats.drop(columns=events)
  for event in events:
      # Define auxiliary variables
      event_count = rename[event]
      event_name = event.replace(' ','_')
      cols_aux = ["person", event_count, "time"]
      col_last_event_at = f"last_{event_name}_at"
      # Create filtered dataframe with the last event times
      event_times = transcript_feats.loc[transcript_feats["event"]==event, cols_aux]
      event_times[event_count] += 1
      # Join the two dataframes and calculate the time since the last event
      event_times = event_times.rename(columns={"time": col_last_event_at})
      transcript_feats = transcript_feats.merge(event_times, on=["person", event_count], how="left")
      transcript_feats[f"time_since_last_{event_name}"] = transcript_feats["time"] - transcript_feats[col_last_event_at]
      transcript_feats = transcript_feats.drop(columns=col_last_event_at)

  # Add offer data
  portfolio_renamed = portfolio_df.copy()
  portfolio_renamed.columns = [f"offer_{col}" if col!="offer_id" else col for col in portfolio_renamed]
  transcript_feats = transcript_feats.merge(portfolio_renamed, on="offer_id", how="left")

  # Add demografic data
  transcript_feats = transcript_feats.merge(profile_df, on="person", how="left")

  return transcript_feats
