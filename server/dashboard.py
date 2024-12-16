# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:32:52 2024

@author: VEDANT SHINDE
"""

!pip install dash
!pip install dash-bootstrap-components
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import io
import urllib.parse

# Load the existing dataset
existing_data = pd.read_csv("/content/preprocessed_cash_flow.csv")

# Save data to SQLite database
conn = sqlite3.connect('cash_flow_data.db')
existing_data.to_sql('cash_flow', conn, if_exists='replace', index=False)
conn.close()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Enhanced Cash Flow Dashboard"

# Layout of the dashboard
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Enhanced Cash Flow Dashboard", style={"text-align": "center", "color": "#2c3e50"}),
                width=12
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Filters", className="card-title"),
                                html.Div([
                                    html.Label("Select Categories:", style={"font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id="category-dropdown",
                                        multi=True,
                                        style={"width": "100%"}
                                    ),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Label("Select Regions:", style={"font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id="region-dropdown",
                                        multi=True,
                                        style={"width": "100%"}
                                    ),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Label("Select Transaction Types:", style={"font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id="transaction-type-dropdown",
                                        multi=True,
                                        style={"width": "100%"}
                                    ),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Label("Select Date Range:", style={"font-weight": "bold"}),
                                    dcc.DatePickerRange(
                                        id="date-range-picker",
                                        start_date=existing_data["Date"].min(),
                                        end_date=existing_data["Date"].max(),
                                        display_format="YYYY-MM-DD",
                                        style={"width": "100%"}
                                    ),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Label("Select Date Granularity:", style={"font-weight": "bold"}),
                                    dcc.RadioItems(
                                        id="date-granularity-radio",
                                        options=[
                                            {"label": "Daily", "value": "D"},
                                            {"label": "Weekly", "value": "W"},
                                            {"label": "Monthly", "value": "M"},
                                            {"label": "Quarterly", "value": "Q"}
                                        ],
                                        value="D",
                                        labelStyle={"display": "inline-block", "margin-right": "10px"}
                                    ),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Label("Filter by Discount Applied (%):", style={"font-weight": "bold"}),
                                    dcc.RangeSlider(
                                        id="discount-slider",
                                        min=0,
                                        max=20,
                                        step=1,
                                        value=[0, 20],
                                        marks={i: str(i) for i in range(0, 21, 5)}
                                    ),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Label("Filter by Payment Delay (days):", style={"font-weight": "bold"}),
                                    dcc.RangeSlider(
                                        id="delay-slider",
                                        min=0,
                                        max=15,
                                        step=1,
                                        value=[0, 15],
                                        marks={i: str(i) for i in range(0, 16, 3)}
                                    ),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Label("Select Theme:", style={"font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id="theme-dropdown",
                                        options=[
                                            {"label": "Light", "value": "plotly_white"},
                                            {"label": "Dark", "value": "plotly_dark"},
                                            {"label": "Solar", "value": "solar"},
                                            {"label": "Cyborg", "value": "cyborg"}
                                        ],
                                        value="plotly_white",
                                        style={"width": "100%"}
                                    ),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Button("Export Data to CSV", id="export-csv-button", n_clicks=0),
                                    dcc.Download(id="download-dataframe-csv"),
                                ], style={"padding": "10px"}),
                                html.Div([
                                    html.Button("Export Data to Excel", id="export-excel-button", n_clicks=0),
                                    dcc.Download(id="download-dataframe-excel"),
                                ], style={"padding": "10px"}),
                            ]
                        ),
                        style={"width": "100%"}
                    ),
                    width=3
                ),
                dbc.Col(
                    [
                        dcc.Store(id="filtered-data"),
                        dcc.Graph(id="cash-flow-graph", style={"width": "100%", "padding": "10px"}),
                        dcc.Graph(id="feature-comparison-graph", style={"width": "100%", "padding": "10px"}),
                        dcc.Graph(id="payment-delay-histogram", style={"width": "100%", "padding": "10px"}),
                        dcc.Graph(id="cash-flow-by-category-bar", style={"width": "100%", "padding": "10px"}),
                        dcc.Graph(id="transaction-type-pie", style={"width": "100%", "padding": "10px"}),
                        dcc.Graph(id="seasonality-heatmap", style={"width": "100%", "padding": "10px"}),
                        dcc.Graph(id="payment-delay-box", style={"width": "100%", "padding": "10px"}),
                        html.Div(id="summary-stats", style={"text-align": "center", "margin-top": "20px", "font-size": "18px"}),
                    ],
                    width=9
                ),
            ]
        ),
    ],
    fluid=True
)

# Callbacks for interactivity
@app.callback(
    [Output("category-dropdown", "options"),
     Output("region-dropdown", "options"),
     Output("transaction-type-dropdown", "options")],
    [Input("date-range-picker", "start_date"),
     Input("date-range-picker", "end_date")]
)
def update_filter_options(start_date, end_date):
    conn = sqlite3.connect('cash_flow_data.db')
    query = f"""
    SELECT DISTINCT Category, Region, "Transaction Type"
    FROM cash_flow
    WHERE Date BETWEEN ? AND ?
    """
    params = [start_date, end_date]
    filtered_data = pd.read_sql_query(query, conn, params=params)
    conn.close()

    category_options = [{"label": category, "value": category} for category in filtered_data["Category"].unique()]
    region_options = [{"label": region, "value": region} for region in filtered_data["Region"].unique()]
    transaction_type_options = [{"label": ttype, "value": ttype} for ttype in filtered_data["Transaction Type"].unique()]

    return category_options, region_options, transaction_type_options

@app.callback(
    Output("filtered-data", "data"),
    [Input("category-dropdown", "value"),
     Input("region-dropdown", "value"),
     Input("transaction-type-dropdown", "value"),
     Input("date-range-picker", "start_date"),
     Input("date-range-picker", "end_date"),
     Input("date-granularity-radio", "value"),
     Input("discount-slider", "value"),
     Input("delay-slider", "value")]
)
def filter_data(selected_categories, selected_regions, selected_transaction_types, start_date, end_date, date_granularity, discount_range, delay_range):
    conn = sqlite3.connect('cash_flow_data.db')
    query = f"""
    SELECT * FROM cash_flow
    WHERE Category IN ({','.join(['?']*len(selected_categories))})
    AND Region IN ({','.join(['?']*len(selected_regions))})
    AND "Transaction Type" IN ({','.join(['?']*len(selected_transaction_types))})
    AND Date BETWEEN ? AND ?
    AND "Discount Applied (%)" BETWEEN ? AND ?
    AND "Payment Delay (days)" BETWEEN ? AND ?
    """
    params = selected_categories + selected_regions + selected_transaction_types + [start_date, end_date] + discount_range + delay_range
    filtered_data = pd.read_sql_query(query, conn, params=params)

    if date_granularity != "D":
        filtered_data = filtered_data.resample(date_granularity, on="Date").sum().reset_index()

    conn.close()
    return filtered_data.to_dict('records')

@app.callback(
    [Output("cash-flow-graph", "figure"),
     Output("feature-comparison-graph", "figure"),
     Output("payment-delay-histogram", "figure"),
     Output("cash-flow-by-category-bar", "figure"),
     Output("transaction-type-pie", "figure"),
     Output("seasonality-heatmap", "figure"),
     Output("payment-delay-box", "figure"),
     Output("summary-stats", "children")],
    [Input("filtered-data", "data"),
     Input("theme-dropdown", "value")]
)
def update_graphs(filtered_data, theme):
    filtered_data = pd.DataFrame(filtered_data)
    template = theme

    # Cash flow graph
    cash_flow_fig = px.line(
        filtered_data,
        x="Date",
        y="Daily Cash Flow",
        title="Daily Cash Flow",
        labels={"Daily Cash Flow": "Cash Flow (USD)", "Date": "Date"},
        template=template
    )
    cash_flow_fig.update_layout(hovermode="x unified")

    # Feature comparison graph
    feature_comparison_fig = px.scatter(
        filtered_data,
        x="Seasonality Index",
        y="Daily Cash Flow",
        color="Transaction Type",
        size="Discount Applied (%)",
        title="Feature Comparison: Seasonality vs Cash Flow",
        labels={"Seasonality Index": "Seasonality Index", "Daily Cash Flow": "Cash Flow (USD)"},
        template=template
    )

    # Payment delay histogram
    payment_delay_fig = px.histogram(
        filtered_data,
        x="Payment Delay (days)",
        nbins=15,
        title="Payment Delay Distribution",
        labels={"Payment Delay (days)": "Payment Delay (days)"},
        template=template
    )
    payment_delay_fig.update_layout(bargap=0.1)

    # Bar chart for daily cash flow by category
    cash_flow_by_category_fig = px.bar(
        filtered_data,
        x="Category",
        y="Daily Cash Flow",
        title="Daily Cash Flow by Category",
        labels={"Daily Cash Flow": "Cash Flow (USD)", "Category": "Category"},
        template=template
    )

    # Pie chart for transaction types
    transaction_type_pie_fig = px.pie(
        filtered_data,
        names="Transaction Type",
        title="Transaction Type Distribution",
        template=template
    )

    # Heatmap for seasonality index vs. daily cash flow
    seasonality_heatmap_fig = px.density_heatmap(
        filtered_data,
        x="Seasonality Index",
        y="Daily Cash Flow",
        title="Heatmap: Seasonality Index vs Cash Flow",
        labels={"Seasonality Index": "Seasonality Index", "Daily Cash Flow": "Cash Flow (USD)"},
        template=template
    )

    # Box plot for payment delays
    payment_delay_box_fig = px.box(
        filtered_data,
        x="Payment Delay (days)",
        title="Payment Delay Distribution (Box Plot)",
        labels={"Payment Delay (days)": "Payment Delay (days)"},
        template=template
    )

    # Summary statistics
    total_cash_flow = filtered_data["Daily Cash Flow"].sum()
    average_payment_delay = filtered_data["Payment Delay (days)"].mean()
    summary_stats = html.Div([
        html.H3("Summary Statistics", style={"color": "#ecf0f1" if theme == "plotly_dark" else "#2c3e50"}),
        html.P(f"Total Cash Flow: ${total_cash_flow:.2f}", style={"color": "#ecf0f1" if theme == "plotly_dark" else "#2c3e50"}),
        html.P(f"Average Payment Delay: {average_payment_delay:.2f} days", style={"color": "#ecf0f1" if theme == "plotly_dark" else "#2c3e50"}),
    ], style={"background-color": "#34495e" if theme == "plotly_dark" else "#f5f5f5", "padding": "20px", "border-radius": "10px"})

    return cash_flow_fig, feature_comparison_fig, payment_delay_fig, cash_flow_by_category_fig, transaction_type_pie_fig, seasonality_heatmap_fig, payment_delay_box_fig, summary_stats

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-csv-button", "n_clicks"),
    State("filtered-data", "data")
)
def export_data_to_csv(n_clicks, filtered_data):
    if n_clicks > 0:
        filtered_data = pd.DataFrame(filtered_data)
        csv_string = filtered_data.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return dict(content=csv_string, filename="filtered_data.csv")
    return None

@app.callback(
    Output("download-dataframe-excel", "data"),
    Input("export-excel-button", "n_clicks"),
    State("filtered-data", "data")
)
def export_data_to_excel(n_clicks, filtered_data):
    if n_clicks > 0:
        filtered_data = pd.DataFrame(filtered_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            filtered_data.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)
        return dict(content=output.getvalue(), filename="filtered_data.xlsx", type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return None

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
