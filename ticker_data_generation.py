'''
In this script, we will use the Yahoo Finance API to get historical stock price data for a specific stock.
We will use the data to generate labels for the stock price data using the triple barrier method.
We will also calculate moving window metrics to be used as features for predictive modeling.
We will save the labeled data to a CSV file.
'''

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

period = 'max'
symbol = 'INTC' #tracking intel stock

data = yf.download(symbol, period = period)



# Fix the labels
data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
data = data.reset_index()
#data.head()

data.set_index('Date', inplace=True)

data_close = data.copy()
data_close.drop(columns = ['Adj Close', 'High', 'Low', 'Open', 'Volume'],axis=1, inplace=True)

# Tripple Barrier Labeling sourced: https://www.ostirion.net/post/dataframe-ready-implementation-for-triple-barrier-labels
# Tripple Barrier Labeling allows us to get ground truth labels for our data, which can help in future for modeling
# Tripple Barrier Labeling is a method to label the data based on the price action in the future
def compute_vol(df: pd.DataFrame, span: int=100) -> pd.DataFrame:
    '''
     Compute period volatility of returns as exponentially
     weighted moving standard deviation.
     Args:
          df (pd.DataFrame): Dataframe with price series in a single
          column.
          span (int): Span for exponential weighting.
     Returns:
          pd.DataFrame: Dataframe containing volatility estimates.
    '''
    df.fillna(method='ffill', inplace=True)
    r = df.pct_change()
    return r.ewm(span=span).std()


def triple_barrier_labels(
    df: pd.DataFrame,
    t: int,
    upper: float=None,
    lower: float=None,
    devs: float=2.5,
    join: bool=False,
    span: int=100) -> pd.DataFrame:
    '''
    Compute the triple barrier label for a price time series.
    Args:
        df (pd.DataFrame): Dataframe with price series in a single
        column.
        t (int): Future periods to obtain the label for.
        upper (float): Returns for upper limit.
        lower (float): Returns for lower limit.
        join (bool): If True, the input dataframe and the labels are
        returned joined.
        span (int): Span for exponential weighting.
        dev (float): Standard deviations to set the upper and lower
        return limits when no limits are passed.
    Returns:
        pd.DataFrame: Dataframe containing labels and optionally
        (join=True) input values.
    '''

    if t < 1:
        raise ValueError("Look-ahead time invalid, t<1.")

    df.fillna(method='ffill', inplace=True)
    lims = np.array([upper, lower])
    labels = pd.DataFrame(index=df.index, columns=['Label'])
    returns = df.pct_change()

    for idx in range(0, len(df)-1-t):
        s = returns.iloc[idx:idx+t]
        minimum = s.cumsum().values.min()
        maximum = s.cumsum().values.max()

        if not all(np.isfinite(s.cumsum().values)):
            labels['Label'].iloc[idx] = np.nan
            continue

        if any(lims == None):
            vol = compute_vol(df[:idx+t], span)

        if upper is None:
            u = vol.iloc[idx].values * devs
        else:
            u = upper

        if lower is None:
            l = -vol.iloc[idx].values * devs
        else:
            l = lower

        if not (np.isfinite(u) and np.isfinite(l)):
            labels['Label'].iloc[idx] = np.nan
            continue

        if any(s.cumsum().values >= u):
            labels['Label'].iloc[idx] = 1
        elif any(s.cumsum().values <= l):
            labels['Label'].iloc[idx] = -1
        else:
            labels['Label'].iloc[idx] = 0

    if join:
        return df.join(labels)
    return labels

labels = triple_barrier_labels(
    df= data_close,
    t = 10,
    upper =None,
    lower =None,
    devs = 2.5,
    join = False,
    span =20
)

labeled_data = pd.merge(data, labels, left_index=True, right_index=True)

# We also get moving window metrics to be used for predictive modeling in future:

# moving window metrics to help with model:
# Feature engineering with moving metrics
labeled_data['SMA_20'] = labeled_data['Close'].rolling(window=20).mean()
labeled_data['SMA_50'] = labeled_data['Close'].rolling(window=50).mean()
labeled_data['EMA_20'] = labeled_data['Close'].ewm(span=20, adjust=False).mean()
labeled_data['Volatility_20'] = labeled_data['Close'].rolling(window=20).std()
labeled_data['RSI'] = 100 - (100 / (1 + labeled_data['Close'].pct_change().rolling(14).mean() / labeled_data['Close'].pct_change().rolling(14).std()))
labeled_data['Momentum_10'] = labeled_data['Close'].diff(10)

# Drop rows with NaN values resulting from rolling calculations
labeled_data.dropna(inplace=True)

# Export to csv
labeled_data.to_csv('ticker_data.csv')