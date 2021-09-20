import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st

# Set the chart themes
pio.templates.default = "none"
plt.style.use('dark_background')

# Auxiliary variables and lookup dictionaries
demog_cols = {
  "Age": "age",
  "Income": "income",
  "Cohort": "became_member_on"
}
demog_group_cols = {
  "Age": "age_group",
  "Income": "income_group",
  "Cohort": "cohort_group",
  "Gender": "gender"
}


def promoFunnelFig(promo_funnel):
  """ Returns a figure with the bar chart of the offer's marketing funnel
  """

  return go.Figure(data=[
    go.Bar(name=step, x=promo_funnel[step], y=promo_funnel["code"], orientation="h")
    for step in ["offer received", "offer viewed", "offer completed"]
  ])


def sentOffersDistributionFig(offer_dist):
  """ Returns a figure with the distribution of offer types sent
  """

  fig = go.Figure(data=
    go.Bar(x=offer_dist["size"], y=offer_dist["code"], orientation="h")
  )
  delta_max = offer_dist["size"].abs().max() + .01
  fig.update_layout(xaxis={
    "tickformat": ',.0%',
    "range": [-delta_max, delta_max]
  })

  return fig


def demographicDistributionBarH(df, feat):
  """ Returns a figure with the value counts of a demographic feature in a horizontal bar chart
  """

  group_dist = df[demog_group_cols[feat]].value_counts(normalize=True, sort=False)
  fig, ax = plt.subplots(figsize=(5,2))
  group_dist.plot.barh(ax=ax)
  ax.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
  ax.grid(axis="x")

  return fig


def demographicDistributionHist(df, feat):
  """ Returns a figure with the histogram of a demographic feature
  """
  df_subset = df.dropna(subset=[demog_cols[feat]])
  return px.histogram(df_subset, x=demog_cols[feat], color=demog_group_cols[feat])
