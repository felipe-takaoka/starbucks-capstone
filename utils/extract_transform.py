import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@st.cache
def cachedLoadAndCleanPortfolio():
  return loadAndCleanPortfolio()

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
  portfolio_df["code"] = portfolio_df["offer_type"].str[0] + "." + \
    portfolio_df["difficulty"].astype(str) + "."  + \
    portfolio_df["reward"].astype(str) + "."  + \
    portfolio_df["duration"].astype(str)

  # Rename and order columns and rows
  portfolio_df = portfolio_df.rename(columns={"id": "offer_id", "offer_type": "type"})
  cols = ["offer_id","code","type","difficulty","reward","duration","email","mobile","social","web"]
  portfolio_df = portfolio_df[cols]
  portfolio_df = portfolio_df.sort_values(["type", "difficulty", "reward", "duration"])

  return portfolio_df


@st.cache
def cachedLoadAndCleanProfile():
  return loadAndCleanProfile()

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
def cachedLoadAndCleanTranscript():
  return loadAndCleanTranscript()

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


def getPromoFunnel(transcript_df, portfolio_df):
  """Get a dataframe containing funnel data of offers"""
  # Effectiveness of each channel in converting with promotion
  promo_funnel = transcript_df.groupby(["offer_id", "event"]).size().unstack()
  promo_funnel = promo_funnel.fillna(0).astype(int)
  promo_funnel["view_rate"] = promo_funnel["offer viewed"] / promo_funnel["offer received"]
  promo_funnel["comp_rate"] = promo_funnel["offer completed"] / promo_funnel["offer viewed"]
  promo_funnel = promo_funnel.reset_index()

  return portfolio_df.merge(promo_funnel, on="offer_id").sort_values(
      ["type", "difficulty", "reward", "duration"],
      ascending=[True, True, False, False]
  )


def getOffersDist(transcript_df, portfolio_df):
  """Get a dataframe containing the distribution of offers received"""
  received_offers = transcript_df[transcript_df["event"]=="offer received"]
  offers_dist = transcript_df.groupby("offer_id", as_index=False).size()
  offers_dist["size"] /= offers_dist["size"].sum()
  offers_dist["size"] -= 1/offers_dist.shape[0]
  offers_dist = offers_dist.merge(portfolio_df, on="offer_id")
  offers_dist = offers_dist.sort_values("size")

  return offers_dist


@st.cache
def cachedCreateTranscriptFeatures(transcript_df, portfolio_df, profile_df):
  return createTranscriptFeatures(transcript_df, portfolio_df, profile_df)

def createTranscriptFeatures(transcript_df, portfolio_df, profile_df):
  """ Returns dataframe containing useful features for predicting customer behaviour
  """

  transcript_feats = transcript_df.copy()

  # Create dummy variables for the events (to perform a cumulative sum)
  transcript_feats = pd.concat([transcript_feats, pd.get_dummies(transcript_feats["event"])], axis=1)
  transcript_feats["event_no_aux"] = 1

  # Define the columns to aggregate, their function and their new name
  agg_cols = {
      # original column name: (agg function, agg column name)
      "event_no_aux": ("sum", "event_no"),
      "amount": ("sum", "cum_spending"),
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
  transcript_feats = transcript_feats.drop(columns="event_no_aux")
  transcript_feats = transcript_feats.merge(transcript_aggs, on=["person", "event_no"])

  # Subtract the current "event" so that they account only for the past (without information not available on inference time)
  cols_subtract = [col for col in agg_cols.keys() if col not in ["event_no_aux", "time"]]
  cols_keep = [rename[col] for col in cols_subtract]
  transcript_feats.loc[:, cols_keep] -= transcript_feats.loc[:, cols_subtract].values

  # Time since each person's first event
  transcript_feats["time_since_first_event"] = transcript_feats["time"] - transcript_feats["min_time"]
  transcript_feats = transcript_feats.drop(columns="min_time")

  # Average transaction value (up to that point)
  transcript_feats["atv"] = transcript_feats["cum_spending"] / transcript_feats["transactions"]
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
      # Create filtered dataframe with the times of previous events
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
  # Change offer duration to hours and create time until offers are valid
  transcript_feats["offer_duration"] = 24*transcript_feats["offer_duration"]

  # Add customer offer timeline
  # 1. Select only relevant rows and columns
  received_offers = transcript_feats[transcript_feats["event"]=="offer received"]
  received_offers = received_offers[["person","event_no","time","event","offer_code","offer_duration"]]
  offer_dummies = pd.get_dummies(received_offers["offer_code"], prefix="active")
  received_offers = pd.concat([received_offers, offer_dummies], axis=1)
  # 2. Repeat rows of offers by their time duration in hours
  rep_index = received_offers.index.repeat(received_offers["offer_duration"])
  active_offers = received_offers.drop(columns=["event","offer_code","offer_duration"]).loc[rep_index]
  active_offers["time"] += active_offers.groupby(["person","event_no"]).cumcount()
  # 3. Aggregate the offers by person and time, taking all the valid offers at that time and join
  active_offers = active_offers.drop(columns="event_no").groupby(["person","time"], as_index=False).max()
  transcript_feats = transcript_feats.merge(active_offers, on=["person","time"], how="left")
  # 4. Fill NA and convert value of active offer columns (if it's na, customer didn\t receive it)
  offer_cols = [f"active_{offer}" for offer in portfolio_df["code"]]
  transcript_feats[offer_cols] = transcript_feats[offer_cols].fillna(0).astype(int)

  # Transform remaining offer data
  offer_type_dummies = pd.get_dummies(transcript_feats["offer_type"], prefix="offer_type")
  transcript_feats = pd.concat([transcript_feats.drop(columns="offer_type"), offer_type_dummies], axis=1)

  # Add demografic data
  transcript_feats = transcript_feats.merge(profile_df, on="person", how="left")

  return transcript_feats
