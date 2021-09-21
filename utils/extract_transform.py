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
def cachedLoadAndCleanProfile(return_raw=False):
  return loadAndCleanProfile(return_raw)

def loadAndCleanProfile(return_raw=False):
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

  if return_raw:
    return profile, profile_df
  else:
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


@st.cache
def cachedCreateTargets(transcript_feats, portfolio_df):
  return createTargets(transcript_feats, portfolio_df)

def createTargets(transcript_feats, portfolio_df):
  """Returns a dataframe containing the spendings for every time window of offer durations"""

  # Auxiliary variables
  dummy_base_ts = pd.Timestamp("2021-01-01")
  last_event_ts = dummy_base_ts + pd.to_timedelta(transcript_feats["time"].max(), "h")
  time_windows = sorted(24*portfolio_df["duration"].unique())

  # Filter only relevant columns and rows
  target_df = transcript_feats[["person","time","event","amount"]].copy()
  target_df = target_df[target_df["event"].isin(["offer received", "transaction"])]
  Y_df = target_df.copy()

  # Add dummy events in the future (to make sure it exists an event at time t+time_window
  # for every event occurring at time t) - this is done because rolling is only performed
  # for previous rows
  dummy_events = target_df.copy().rename(columns={"time": "orig_time"})
  dummy_events["amount"] = 0
  for time_window in time_windows:
      dummy_events["time"] = dummy_events["orig_time"] + time_window
      target_df = pd.concat([target_df, dummy_events.drop(columns="orig_time")])

  # Group concurrent events
  # A dummy base timestamp is used because it's needed in the following rolling function
  target_df["ts"] = dummy_base_ts + pd.to_timedelta(target_df["time"], "h")
  target_df = target_df.drop(columns="time")
  target_df = target_df.groupby(["person","ts"], as_index=False).sum()

  # Calculate the future spending for each time window (offer durations)
  for time_window in time_windows:
      # Rolling sums (closed on the left endpoint so that transactions occurring at the same time, i.e. hour,
      # are included in the future spending)
      rs = target_df.groupby("person", as_index=False).rolling(f"{time_window}h", on="ts", closed="left").sum()
      rs_spending_col_name = f"spending_next_{time_window}h"
      rs = rs.rename(columns={"amount": rs_spending_col_name})

      # Set spendings to NA if time window contains events after the last time in the dataset
      rs[rs_spending_col_name] = rs[rs_spending_col_name].where(rs["ts"] < last_event_ts)

      # Shifts events by the time_window so that the rolling sum is over the future
      rs["time"] = (rs["ts"]-dummy_base_ts).dt.total_seconds()/3600 - time_window
      rs["time"] = rs["time"].astype(int)
      rs = rs.drop(columns="ts")

      Y_df = Y_df.merge(rs, on=["person","time"], how="left")

  Y_df = Y_df[Y_df["event"]=="offer received"]
  Y_df = Y_df.drop(columns=["event","amount"])

  return Y_df


def getTrainingDataset(transcript_feats, Y_df, return_df_full=False):
  """Returns the training dataset by joining the features and target and filtering
  for received offer events
  """

  df_full = transcript_feats[transcript_feats["event"]=="offer received"].copy()
  df_full = df_full.merge(Y_df, on=["person","time"], how="left").reset_index(drop=True)
  df = df_full.drop(columns=[
      "person","event_no","time","event","amount","reward","offer_id","offer_code"]
    ).copy()

  if return_df_full:
    return df_full, df
  else:
    return df


def createDemographicGroups(profile):
  """ Returns a dataframe containing the demographic groups defined
  """

  demographics = profile.copy()

  # Age groups
  demographics["age_group"] = pd.cut(demographics["age"], [0,30,50,70,100,200], labels=np.arange(5)+1)
  # Income groups
  demographics["income_group"] = pd.qcut(demographics["income"], [0,.2,.8,1], labels=[0,1,2])
  # Cohort groups
  demographics["became_member_on"] = pd.to_datetime(demographics["became_member_on"].astype(str)).astype(int)
  bins = demographics["became_member_on"].value_counts(bins=100, sort=False).to_frame()
  bins.columns = ["count"]
  bins["dcount"] = bins["count"].shift(1) - bins["count"]
  breaks = bins[abs(bins["dcount"])>100]
  cohort_breaks = [0] + list(breaks.index.left) + list([demographics["became_member_on"].max()])
  demographics["cohort_group"] = pd.cut(demographics["became_member_on"], cohort_breaks, labels=[1,2,3,4])

  return demographics


def createSpendingsPerGroup(df_full, demographics, time_windows, return_raw=False):
  """ Returns a dataframe containing the customer spendings upon receiving an offer
  up until its validity grouped by each demographic group
  """

  # Merge the demographic data
  demog_cols = ["age_group", "income_group", "cohort_group", "gender"]
  demographics_groups = demographics[["id"] + demog_cols]
  demographics_groups = demographics_groups.rename(columns={"id": "person"})
  demog_spendings = df_full.merge(demographics_groups, on="person", how="left")

  # Coalesce spending windows into a spending for the specific duration of the offer received
  # and normalize by the offer duration
  target_col = "daily_offer_spending"
  for t in time_windows:
      spend_col = f"spending_next_{t}h"
      window_mask = demog_spendings["offer_duration"]==t
      demog_spendings.loc[window_mask, target_col] = demog_spendings.loc[window_mask, spend_col] / (t/24)

  # Subset columns
  feat_cols = demog_cols + ["offer_code"]
  demog_spendings = demog_spendings[feat_cols + [target_col]]

  # Drop rows where the end of the offer exceeded the last observed time
  demog_spendings = demog_spendings.dropna(subset=[target_col])

  # Get aggregate spendings by demographic groups and offers (median to filter outliers)
  agg_metrics = {target_col: ["median","size"]}
  metric_names = ["spending_median", "size"]
  spendings_per_groups = demog_spendings.groupby(feat_cols).agg(agg_metrics).reset_index()
  spendings_per_groups.columns = feat_cols + metric_names

  # Drop groups with small sample size
  spendings_per_groups = spendings_per_groups[spendings_per_groups["size"] >= 30]

  if return_raw:
    return demog_spendings, spendings_per_groups
  else:
    return spendings_per_groups


def spendingsForOffers(df, offers, demog_feats, min_group_size):
  """Returns a dataframe containing the spendings filtered by offers and grouped by demographic groups"""

  # Filter by offer types and group by demographic groups
  agg_metrics = {"daily_offer_spending": ["median","size"]}
  metric_names = ["spending_median", "size"]
  spendings = df[df["offer_code"].isin(offers)].groupby(demog_feats).agg(agg_metrics).reset_index()
  spendings.columns = demog_feats + metric_names

  # Filter by group size and sort by spending
  spendings = spendings[spendings["size"]>=min_group_size]
  spendings = spendings.sort_values("spending_median", ascending=False)
  
  return spendings


def bestOfferForGroup(df, portfolio, group_def):
  """ Returns the best offer for a demographic group
  """

  # Filter demographic group
  for feat_col, group in group_def:
    df = df[df[feat_col]==group]

  # Group by offer code
  agg_metrics = {"daily_offer_spending": ["median","size"]}
  metric_names = ["spending_median", "size"]
  df = df.groupby("offer_code").agg(agg_metrics).reset_index()
  df.columns = ["offer_code"] + metric_names

  # Sort
  df = df.sort_values("spending_median", ascending=False)  

  return df.head()


def getGroupStats(group_def, demographics, demog_spendings):
  """ Returns some metrics corresponding to a demographic group
  """

  # Filter demographic group
  for feat_col, group in group_def:
    demographics = demographics[demographics[feat_col]==group]
    demog_spendings = demog_spendings[demog_spendings[feat_col]==group]

  # Calculate metrics
  n_customers = demographics.shape[0]
  n_offers_sent = demog_spendings.shape[0]
  n_unique_offers = demog_spendings["offer_code"].nunique()
  
  return n_customers, n_offers_sent, n_unique_offers
