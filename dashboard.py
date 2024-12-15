# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:29:06 2024

@author: VEDANT SHINDE
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import pandas as pd


# Load preprocessed data
data = pd.read_csv("preprocessed_cash_flow.csv")
data["Date"] = pd.to_datetime(data["Date"])

# Sample metrics for demonstration
total_cash_flow = data["Daily Cash Flow"].sum()
overdue_payments = 5  # Replace with your calculation logic
risk_level = "Medium"  # Replace with your calculation logic

# Initialize the Dash app
app = dash.Dash(__name__)

# Dashboard layout
app.layout = html.Div([
    html.H1("SME Risk Dashboard", style={'textAlign': 'center'}),
    
    # Metrics Panel
    html.Div([
        html.Div(f"Total Cash Flow: ${total_cash_flow:.2f}", style={'margin': '10px'}),
        html.Div(f"Overdue Payments: {overdue_payments}", style={'margin': '10px'}),
        html.Div(f"Risk Level: {risk_level}", style={'margin': '10px'}),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),
    
    # Cash Flow Visualization
    dcc.Graph(
        id='cash-flow-chart',
        figure={
            'data': [
                go.Scatter(x=data["Date"], y=data["Daily Cash Flow"], mode='lines', name='Daily Cash Flow')
            ],
            'layout': {
                'title': 'Cash Flow Over Time',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Cash Flow'}
            }
        }
    ),
    
    # Overdue Payments (Placeholder)
    html.Div([
        html.H3("Overdue Payments (Demo Table):"),
        html.Table([
            html.Tr([html.Th("Invoice"), html.Th("Due Date"), html.Th("Amount"), html.Th("Risk")]),
            html.Tr([html.Td("INV001"), html.Td("2023-12-01"), html.Td("$500"), html.Td("High")]),
            html.Tr([html.Td("INV002"), html.Td("2023-11-15"), html.Td("$300"), html.Td("Medium")])
        ])
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
