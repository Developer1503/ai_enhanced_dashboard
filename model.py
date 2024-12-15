# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:32:52 2024

@author: VEDANT SHINDE
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]
cash_flows = np.random.uniform(1000, 5000, size=365)  # Random values between 1000 and 5000

# Additional attributes
categories = ["Revenue", "Expense", "Investment"]
transaction_types = ["Credit", "Debit"]
regions = ["North", "South", "East", "West"]
account_types = ["Savings", "Current"]
seasonality_indices = [round(np.sin(2 * np.pi * i / 365) + 1, 2) for i in range(365)]  # Simulate seasonality
payment_delays = np.random.randint(0, 15, size=365)  # Payment delays in days
discounts_applied = np.random.uniform(0, 20, size=365)  # Discounts in percentage

# Create DataFrame
data = pd.DataFrame({
    "Date": dates,
    "Daily Cash Flow": cash_flows,
    "Category": np.random.choice(categories, size=365),
    "Transaction Type": np.random.choice(transaction_types, size=365),
    "Region": np.random.choice(regions, size=365),
    "Account Type": np.random.choice(account_types, size=365),
    "Seasonality Index": seasonality_indices,
    "Payment Delay (days)": payment_delays,
    "Discount Applied (%)": discounts_applied
})

data.to_csv("preprocessed_cash_flow.csv", index=False)
# Load preprocessed data
data = pd.read_csv("preprocessed_cash_flow.csv")
data["Date"] = pd.to_datetime(data["Date"])

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Enhanced Cash Flow Dashboard"

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Enhanced Cash Flow Dashboard", style={"text-align": "center"}),

    # Dropdown for selecting category
    html.Div([
        html.Label("Select Category:"),
        dcc.Dropdown(
            id="category-dropdown",
            options=[{"label": category, "value": category} for category in data["Category"].unique()],
            value="Revenue",
            clearable=False,
        ),
    ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),

    # Dropdown for selecting region
    html.Div([
        html.Label("Select Region:"),
        dcc.Dropdown(
            id="region-dropdown",
            options=[{"label": region, "value": region} for region in data["Region"].unique()],
            value="North",
            clearable=False,
        ),
    ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),

    # Graph for daily cash flow
    dcc.Graph(id="cash-flow-graph"),

    # Graph for feature comparison
    dcc.Graph(id="feature-comparison-graph"),

    # Histogram for payment delay
    dcc.Graph(id="payment-delay-histogram"),
])

# Callbacks for interactivity
@app.callback(
    Output("cash-flow-graph", "figure"),
    [Input("category-dropdown", "value"),
     Input("region-dropdown", "value")]
)
def update_cash_flow_graph(selected_category, selected_region):
    filtered_data = data[(data["Category"] == selected_category) & (data["Region"] == selected_region)]
    fig = px.line(
        filtered_data,
        x="Date",
        y="Daily Cash Flow",
        title=f"Daily Cash Flow: {selected_category} in {selected_region}",
        labels={"Daily Cash Flow": "Cash Flow (USD)", "Date": "Date"}
    )
    fig.update_layout(hovermode="x unified")
    return fig

@app.callback(
    Output("feature-comparison-graph", "figure"),
    [Input("category-dropdown", "value"),
     Input("region-dropdown", "value")]
)
def update_feature_comparison(selected_category, selected_region):
    filtered_data = data[(data["Category"] == selected_category) & (data["Region"] == selected_region)]
    fig = px.scatter(
        filtered_data,
        x="Seasonality Index",
        y="Daily Cash Flow",
        color="Transaction Type",
        size="Discount Applied (%)",
        title="Feature Comparison: Seasonality vs Cash Flow",
        labels={"Seasonality Index": "Seasonality Index", "Daily Cash Flow": "Cash Flow (USD)"}
    )
    return fig

@app.callback(
    Output("payment-delay-histogram", "figure"),
    [Input("category-dropdown", "value"),
     Input("region-dropdown", "value")]
)
def update_payment_delay_histogram(selected_category, selected_region):
    filtered_data = data[(data["Category"] == selected_category) & (data["Region"] == selected_region)]
    fig = px.histogram(
        filtered_data,
        x="Payment Delay (days)",
        nbins=15,
        title="Payment Delay Distribution",
        labels={"Payment Delay (days)": "Payment Delay (days)"}
    )
    fig.update_layout(bargap=0.1)
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
