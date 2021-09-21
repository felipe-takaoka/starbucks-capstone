import pickle
import streamlit as st
from .extract_transform import *

inference_time_windows = [72, 96, 120, 168, 240]

def loadModels():
  """ Loads the fitted predictive models to infer the customer's spendings for different offers
  """

  with open("models.pickle", "rb") as handle:
    models = pickle.load(handle)

  return models


def splitFeaturesTarget(df):
  """ Given the full dataset, it splits into two dataframes: the features and the targets
  """

  target_cols = [f"spending_next_{time_window}h" for time_window in inference_time_windows]
  return df.drop(columns=target_cols), df[target_cols]


def getCustomerFeatures(customer, time, df, portfolio_df):
  """ Returns a dataframe containing a single row representing the customer with the features for
  inputting into the predictive model
  """
  
  customer_df = df[df["person"]==customer].copy()
  row_customer = customer_df.nlargest(1, "event_no")

  # Calculate time since last event occurred
  delta_time = time - row_customer["time"]
  # Add this delta time to the time features
  time_features = [
    "time_since_first_event",
    "time_since_last_offer_received",
    "time_since_last_offer_viewed",
    "time_since_last_transaction",
    "time_since_last_offer_completed",
  ]
  for time_feat in time_features:
    row_customer[time_feat] += delta_time

  # Determine if each offer received up until that time will still be valid
  received = customer_df[customer_df["event"]=="offer received"].sort_values("event_no", ascending=False)
  received = received.drop_duplicates(subset=["offer_code"])
  for _, offer in received.iterrows():
    valid_until = offer["time"] + offer["offer_duration"]
    offer_code = offer["offer_code"]
    row_customer[f"active_{offer_code}"] = 1*(time < valid_until)

  # Set the offers
  customer_with_offers = []
  offer_cols = ["code", "difficulty", "reward", "duration", "email", "mobile", "social", "web"]
  for _, row in portfolio_df.iterrows():
    offer = row["code"]
    row_customer_offer = row_customer.copy()

    # Set the offer to active
    row_customer_offer[f"active_{offer}"] = 1

    # Set all features related to the offer being sent
    for offer_col in offer_cols:
      row_customer_offer[f"offer_{offer_col}"] = row[offer_col]

    # Plus the type of the offer
    offer_type = row["type"]
    row_customer_offer[f"offer_type_{offer_type}"] = 1

    # Append the row with the offer
    customer_with_offers.append(row_customer_offer)
  
  # Concatenate all the offers to simulate
  customerFeats = pd.concat(customer_with_offers)

  return customerFeats


def predictCustomerSpendings(df):
  """ Given a dataframe containing the features, predicts the spendings of all time windows
  """

  # Drop auxiliary features
  df_with_pred = df[["offer_code"]].copy().reset_index(drop=True)
  df = dropAuxFeatures(df)

  models = loadModels()

  prev_spending = 0
  for model in models:
    col_target = model["target"].replace("spending_next_", "")

    # Predict
    df_with_pred[col_target] = model["classifier"].predict(df)

    # Recreate cumulative spending target (since they were trained as incremental)
    df_with_pred[col_target] += prev_spending

    prev_spending = df_with_pred[col_target]

  return df_with_pred
