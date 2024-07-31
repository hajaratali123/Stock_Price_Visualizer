import os
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
from jinja2 import Environment

app = Flask(__name__)

# Load datasets
apple_stock = pd.read_csv(r'stock_data\Apple.csv')
amazon_stock = pd.read_csv(r'stock_data\Amazon.csv')
google_stock = pd.read_csv(r'stock_data\Google.csv')
tesla_stock = pd.read_csv(r'stock_data\tesla.csv')

# Convert 'Date' column to datetime
apple_stock['Date'] = pd.to_datetime(apple_stock['Date'])
amazon_stock['Date'] = pd.to_datetime(amazon_stock['Date'])
google_stock['Date'] = pd.to_datetime(google_stock['Date'])
tesla_stock['Date'] = pd.to_datetime(tesla_stock['Date'])




# Function to generate Plotly figure
def generate_plot(df, spike_lines=True):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close'
    ))

    if spike_lines:
        fig.update_layout(
            title='Stock Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
    else:
        fig.update_layout(
            title='Stock Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode=False
        )

    return fig



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_choice = request.form['stockChoice']
        interval = request.form['interval']

        if stock_choice == 'Apple':
            df = apple_stock
        elif stock_choice == 'Amazon':
            df = amazon_stock
        elif stock_choice == 'Google':
            df = google_stock
        elif stock_choice == 'Tesla':
            df = tesla_stock
        else:
            df = pd.DataFrame()  # Empty DataFrame in case of no stock selection

        # Predict stock price for the chosen interval
        model = LinearRegression()
        X = np.array(range(len(df))).reshape(-1, 1)  # Feature: Days as integers
        y = df['Close'].values  # Target: Closing prices
        model.fit(X, y)

        interval_days = {
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }

        future_days = interval_days.get(interval, 30)  # Default to 30 days if not found
        future_date = df['Date'].iloc[-1] + timedelta(days=future_days)
        future_index = len(df) + future_days

        future_prediction = model.predict([[future_index]])  # Predict for the future date

        # Convert prediction from USD to INR
        usd_to_inr_conversion_rate = 1 # Example conversion rate
        future_prediction_inr = future_prediction[0] * usd_to_inr_conversion_rate

        # Append prediction to the DataFrame
        df_next_day = pd.DataFrame({'Date': [future_date], 'Close': [future_prediction[0]]})
        df_with_prediction = pd.concat([df, df_next_day], ignore_index=True)

        # Generate Plotly figure
        spike_lines = 'spike-lines' in request.form  # Check if spike-lines checkbox is checked
        fig = generate_plot(df_with_prediction, spike_lines)
        graphJSON = pio.to_json(fig, remove_uids=False)

        return render_template('index.html', graphJSON=graphJSON, prediction=future_prediction_inr, future_date=future_date)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
