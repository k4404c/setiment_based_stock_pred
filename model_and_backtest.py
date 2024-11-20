import pandas as pd

# import data
ticker_sentiment_data = pd.read_csv('ticker_sentiment_data.csv')

ticker_sentiment_data.drop(columns=['prob_neutral', 'prob_positive', 'prob_negative',
                                    'voted_label', 'label_distribution', 'majority_label',
                                    'sentiment_interpretation', 'day_type'], inplace=True)

# convert nans to zeros
ticker_sentiment_data.fillna(0, inplace=True)

ticker_sentiment_data.set_index('Date', inplace=True)
ticker_sentiment_data.index = pd.to_datetime(ticker_sentiment_data.index)




# Create radnom forest model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = ticker_sentiment_data.copy()


X = df[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume',
        'SMA_20', 'SMA_50', 'EMA_20', 'Volatility_20', 'RSI',
        'Momentum_10', 'existing_news', 'total_comments',
        'article_count', 'sentiment_score']]

#X = X[:-1]

y = df['Signal']#.shift(-1)  # Target variable
y.dropna(inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle= False)

# Normalize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Assess feature importances
import numpy as np
import matplotlib.pyplot as plt
feature_names = X_test.columns.tolist()
importances = rf_model.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [feature_names[i] for i in indices]

# Create plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()


# Adjust predictions based on sentiment score
from itertools import count
# Function to adjust predictions based on sentiment thresholds
def sentiment_based_adjustment(model_prediction, sentiment_score, lower_threshold, upper_threshold):
    adjusted_predictions = []
    count= 0
    for pred, sentiment in zip(model_prediction, sentiment_score):
        if sentiment < lower_threshold:
            adjusted_predictions.append(-1)  # Use model prediction
            if(pred != -1):
              count += 1
        elif sentiment > upper_threshold:
            adjusted_predictions.append(1)  # Use model prediction
            if(pred != 1):
              count += 1
        else:
            adjusted_predictions.append(pred)  # Use model prediction
    print(f"Number of changed: {count}")
    return adjusted_predictions

# Example user-defined thresholds
lower_threshold = -10
upper_threshold = 10

adjusted_predictions = sentiment_based_adjustment(y_pred, X_test['sentiment_score'], lower_threshold, upper_threshold)

# Evaluations for example
print("Adjusted Classification Report:")
print(classification_report(y_test, adjusted_predictions))



# Calculate and plot backtest performance
import numpy as np
def calculate_drawdown(equity_curve):
    """Calculate the drawdown series and maximum drawdown."""
    rolling_max = equity_curve.expanding().max()
    drawdown = equity_curve / rolling_max - 1
    return drawdown, drawdown.min()


def backtest(data):

    if data is None or len(data) < 2:
        raise ValueError("Insufficient data for backtesting.")

    df = data.copy()

    ## Logarithmic returns with safety for division by zero
    # Return is the rate of change from todays close to tommorrows close
    # Our model assumes we invest right at close, and dont make our change until the next close
    df["returns"] = np.log(df["Close"].shift(-1) / df["Close"].replace(0, np.nan)).fillna(0)
    
    
    df["strategy_returns"] = df["Predicted_Signal"] * df["returns"] 
    #df["effective_signal"] = df["Predicted_Signal"].replace(0, np.nan).fillna(method='ffill').fillna(0) -> 0 means holding previous position
    #df["strategy_returns"] = df["effective_signal"] * df["returns"]

    # Handle missing values
    df["strategy_returns"] = df["strategy_returns"].fillna(0)

    # Cumulative returns
    df["cum_returns"] = df["returns"].cumsum()
    df["cum_strategy_returns"] = df["strategy_returns"].cumsum()

    # Transform to percentage terms
    df["accum_buy_and_hold"] = df["cum_returns"].apply(np.exp)
    df["accum_strategy_returns"] = df["cum_strategy_returns"].apply(np.exp)

    # Trading costs
    trading_cost = 0.001  # 0.1%
   
    # When Signal is different from yesterdays signal, we trade, this incur costs.
    # Consecutive symbols do not incur cost
    # 1 -> 1 : 0 cost
    # 1 -> 0 : +1 trade
    # 1 -> -1: +1 trade cost (2?)
    df["trades"] = (df["Predicted_Signal"] != df["Predicted_Signal"].shift(1)).astype(int).fillna(0)
      

    # Apply trading costs
    df["strategy_returns_tc"] = df["strategy_returns"] - df["trades"] * trading_cost
    df["accum_strategy_returns_tc"] = df["strategy_returns_tc"].cumsum().apply(np.exp)

    # Calculate Win/Loss Ratio
    winning_trades = (df["strategy_returns"] > 0).sum()
    losing_trades = (df["strategy_returns"] < 0).sum()
    win_loss_ratio = winning_trades / losing_trades if losing_trades > 0 else np.inf

    # Calculate Sharpe Ratio (assuming 252 trading days per year)
    risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = df["strategy_returns"] - daily_rf
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    # Calculate Maximum Drawdown
    drawdown_series, max_drawdown = calculate_drawdown(df["accum_strategy_returns"])
    df["drawdown"] = drawdown_series

    # Store metrics in a dictionary
    metrics = {
        "win_loss_ratio": win_loss_ratio,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": winning_trades + losing_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0,
        "final_return": df["accum_strategy_returns"].iloc[-1] - 1,
        "final_return_tc": df["accum_strategy_returns_tc"].iloc[-1] - 1
    }

    # Add metrics to DataFrame attributes
    df.attrs["metrics"] = metrics

    return df




import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_backtest_performance(df, title):
    """
    Plot cumulative returns comparison between strategy and buy-and-hold with metrics
    """
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=('Cumulative Returns', 'Trading Signals', 'Drawdown'),
                        row_heights=[0.5, 0.25, 0.25])

    # Plot cumulative returns
    fig.add_trace(
        go.Scatter(x=df.index,
                  y=df['accum_buy_and_hold'],
                  name='Buy and Hold',
                  line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index,
                  y=df['accum_strategy_returns'],
                  name='Strategy (No Costs)',
                  line=dict(color='green')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index,
                  y=df['accum_strategy_returns_tc'],
                  name='Strategy (With Costs)',
                  line=dict(color='red')),
        row=1, col=1
    )

    # Plot trading signals
    fig.add_trace(
        go.Scatter(x=df.index,
                  y=df['Predicted_Signal'],
                  name='Trading Signal',
                  line=dict(color='purple')),
        row=2, col=1
    )

    # Plot drawdown
    fig.add_trace(
        go.Scatter(x=df.index,
                  y=df['drawdown'],
                  name='Drawdown',
                  fill='tozeroy',
                  line=dict(color='orange')),
        row=3, col=1
    )

    # Add metrics
    metrics = df.attrs['metrics']
    metrics_text = (
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}<br>"
        f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}<br>"
        f"Win Rate: {metrics['win_rate']:.1%}<br>"
        f"Max Drawdown: {metrics['max_drawdown']:.1%}<br>"
        f"Total Trades: {metrics['total_trades']}<br>"
        f"Return (No Costs): {metrics['final_return']:.1%}<br>"
        f"Return (With Costs): {metrics['final_return_tc']:.1%}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.13, y=0.85,
        text=metrics_text,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title= title,
        yaxis_title='Cumulative Returns',
        yaxis2_title='Position',
        yaxis3_title='Drawdown',
        showlegend=True,
        height=1000,

        margin=dict(r=200)  # Make room for metrics annotation
    )

    return fig

def plot_price_action_signals(df, title):
    """
    Create candlestick chart with trading signals and indicators
    """
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=('Price Action & Signals', 'Volume'),
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(x=df.index,
                       open=df['Open'],
                       high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       name='Price'),
        row=1, col=1
    )

    # Add Moving Averages
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index,
                      y=df['SMA_20'],
                      name='SMA 20',
                      line=dict(color='orange', dash='dash')),
            row=1, col=1
        )

    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index,
                      y=df['SMA_50'],
                      name='SMA 50',
                      line=dict(color='blue', dash='dash')),
            row=1, col=1
        )

    # Volume chart
    colors = ['red' if close < open else 'green'
              for close, open in zip(df['Close'], df['Open'])]

    fig.add_trace(
        go.Bar(x=df.index,
               y=df['Volume'],
               name='Volume',
               marker_color=colors),
        row=2, col=1
    )

    # Overlay buy/sell signals
    buy_signals = df[df['Predicted_Signal'] == 1].index
    sell_signals = df[df['Predicted_Signal'] == -1].index

    fig.add_trace(
        go.Scatter(x=buy_signals,
                  y=df.loc[buy_signals, 'Low'] * 0.99,
                  name='Buy Signal',
                  mode='markers',
                  marker=dict(symbol='triangle-up',
                            size=10,
                            color='green')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=sell_signals,
                  y=df.loc[sell_signals, 'High'] * 1.01,
                  name='Sell Signal',
                  mode='markers',
                  marker=dict(symbol='triangle-down',
                            size=10,
                            color='red')),
        row=1, col=1
    )

    # Add metrics
    metrics = df.attrs['metrics']
    metrics_text = (
        f"Winning Trades: {metrics['winning_trades']}<br>"
        f"Losing Trades: {metrics['losing_trades']}<br>"
        f"Win Rate: {metrics['win_rate']:.1%}<br>"
        f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.02, y=0.98,
        text=metrics_text,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title= title,
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800,
        margin=dict(r=200)
    )

    return fig

def analyze_and_plot_all(df, title_back= 'Backtest Performance Analysis', title_signal='Price Action with Trading Signals'):
    """
    Generate and display all backtest visualization plots and metrics
    """
    # Create performance plot
    perf_fig = plot_backtest_performance(df, title_back)
    perf_fig.show()

    # Create price action plot
    price_fig = plot_price_action_signals(df, title_signal)
    price_fig.show()

    # Print comprehensive performance metrics
    metrics = df.attrs['metrics']
    
    print("\nPerformance Metrics:")
    print("=" * 50)
    print(f"Returns:")
    print(f"  Strategy Return (No Costs): {metrics['final_return']:.2%}")
    print(f"  Strategy Return (With Costs): {metrics['final_return_tc']:.2%}")
    print(f"  Buy & Hold Return: {df['accum_buy_and_hold'].iloc[-1] - 1:.2%}")
    print(f"  Outperformance vs B&H (With Costs): {metrics['final_return_tc'] - (df['accum_buy_and_hold'].iloc[-1] - 1):.2%}")
    
    print("\nRisk Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    
    print("\nTrade Statistics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Winning Trades: {metrics['winning_trades']}")
    print(f"  Losing Trades: {metrics['losing_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")



#This is base model
df_y_pred = X_test.copy()
df_y_pred['Predicted_Signal'] = y_pred
df_y_pred.head()

title1 = 'Backtest Performance Analysis Base Model'
title2 = 'Price Action with Trading Signals Base Model'
backtest_df = backtest(df_y_pred)
analyze_and_plot_all(backtest_df, title1, title2)



#This is adjusted threshold based model
lower_threshold= -20
upper_threshold= 20
adjusted_predictions = sentiment_based_adjustment(y_pred, X_test['sentiment_score'], 
                                                  lower_threshold= lower_threshold,
                                                  upper_threshold= upper_threshold)

df_y_pred_adjusted = X_test.copy()
df_y_pred_adjusted['Predicted_Signal'] = adjusted_predictions
df_y_pred_adjusted.head()

title1 = f'Backtest Performance Analysis {lower_threshold}, {upper_threshold} thresholding'
title2 = f'Price Action with Trading Signals {lower_threshold}, {upper_threshold} thresholding'
backtest_df_adjusted = backtest(df_y_pred_adjusted)
analyze_and_plot_all(backtest_df_adjusted, title1, title2)



