'''
The goal of this script is to aggregate the news sentiment data with the stock ticker data.
We will first load the data from the CSV files, clip the data to the overlapping date range, 
and then aggregate the news sentiment data by date (averaging probabilities for the articles in a given day).
We will then convert the sentiment probabilities to a sentiment score on a scale from -100 to 100.
Finally, we will join the sentiment data with the ticker data on the date index.
The resulting DataFrame will be saved to a new CSV file.
'''

import pandas as pd
ticker_data = pd.read_csv('ticker_data.csv')
news_data = pd.read_csv('news_data_with_sentiment.csv')

# set publish date to date type
news_data.drop(columns=['Unnamed: 0'], axis = 1, inplace= True)
news_data['publish_date'] = pd.to_datetime(news_data['attributes.publishOn']).dt.date

ticker_data['publish_date'] = pd.to_datetime(ticker_data['Date']).dt.date


#find clipped bounds for dataframes:
import builtins

min_news = builtins.min(news_data['publish_date'])
max_news = builtins.max(news_data['publish_date'])
min_ticker = builtins.min(ticker_data['publish_date'])
max_ticker = builtins.max(ticker_data['publish_date'])

print(f"latest date in news_data {max_news}")
print(f"earliest date in news_data {min_news}")
print()
print(f"latest date in ticker_data {max_ticker}")
print(f"earliest date in ticker_data {min_ticker}")

clipped_lower = builtins.max(min_news, min_ticker)
clipped_upper = builtins.min(max_news, max_ticker)
print()
print(f"clipped lower {clipped_lower}")
print(f"clipped upper {clipped_upper}")

# clip each data sent to max and min
clipped_news_data = news_data.copy()
clipped_ticker_data = ticker_data.copy()

clipped_news_data = clipped_news_data[(clipped_news_data['publish_date'] >= clipped_lower) & (clipped_news_data['publish_date'] <= clipped_upper)]
clipped_ticker_data = clipped_ticker_data[(clipped_ticker_data['publish_date'] >= clipped_lower) & (clipped_ticker_data['publish_date'] <= clipped_upper)]

min_clipped_news = builtins.min(clipped_news_data['publish_date'])
max_clipped_news = builtins.max(clipped_news_data['publish_date'])
min_clipped_ticker = builtins.min(clipped_ticker_data['publish_date'])
max_clipped_ticker = builtins.max(clipped_ticker_data['publish_date'])

print(f"latest date in clipped_news_data {max_clipped_news}")
print(f"earliest date in news_data {min_clipped_news}")
print()
print(f"latest date in clipped_ticker_data {max_clipped_ticker}")
print(f"earliest date in ticker_data {min_clipped_ticker}")

#remove index
clipped_ticker_data.reset_index(drop=True, inplace=True)


# Aggregate news data by date (combine all news articles for a given date)
from collections import Counter

def get_voted_label(series):
    #Find the label with the most votes.
    
    counts = Counter(series)
    max_count = builtins.max(counts.values())
    max_labels = [label for label, count in counts.items() if count == max_count]

    if len(max_labels) == 1:
        return max_labels[0]
    else:
        return f"tie:{','.join(sorted(max_labels))}"

def get_probability_label(row):
    #Determine the label based on highest probability.
    probs = {
        'neutral': row['prob_neutral'],
        'positive': row['prob_positive'],
        'negative': row['prob_negative']
    }
    return builtins.max(probs, key=probs.get)

def aggregate_daily_sentiment(data):
    #Aggregate sentiment data by date (publish_date).
    
    df = data.copy()
    # Count number of articles per date
    article_counts = df.groupby('publish_date').size()

    # Calculate raw sums for each sentiment
    agg_dict = {
        'neutral': 'sum',
        'positive': 'sum',
        'negative': 'sum',
        'attributes.commentCount': 'sum'
    }

    # Group by date and aggregate
    daily_agg = df.groupby('publish_date').agg(agg_dict)

    # Normalize sentiment probabilities
    sentiment_columns = ['neutral', 'positive', 'negative']
    for date in daily_agg.index:
        total = daily_agg.loc[date, sentiment_columns].sum()
        daily_agg.loc[date, sentiment_columns] = daily_agg.loc[date, sentiment_columns] / total

    # Get voted label (based on most frequent label in original data)
    voted_labels = df.groupby('publish_date')['label'].agg(get_voted_label)

    # Add voted labels and article counts
    daily_agg['voted_label'] = voted_labels
    daily_agg['article_count'] = article_counts

    # Add label distribution
    label_counts = df.groupby('publish_date')['label'].agg(lambda x: dict(Counter(x)))
    daily_agg['label_distribution'] = label_counts

    # Round sentiment scores for readability
    daily_agg[sentiment_columns] = daily_agg[sentiment_columns].round(3)

    # Rename columns for clarity
    daily_agg.columns = [
        'prob_neutral',
        'prob_positive',
        'prob_negative',
        'total_comments',
        'voted_label',
        'article_count',
        'label_distribution'
    ]

    # Add majority label based on probabilities
    daily_agg['majority_label'] = daily_agg.apply(get_probability_label, axis=1)

    return daily_agg

aggregated_daily_sentiment = aggregate_daily_sentiment(clipped_news_data)


def convert_to_sentiment_scale(data):
    #Convert sentiment probabilities to a scale from -100 to 100.

    df = data.copy()

    df['sentiment_score'] = (
        # Difference between positive and negative
        (df['prob_positive'] - df['prob_negative']) *
        # Scale by non-neutral probability to dampen when highly neutral
        (1 - df['prob_neutral']) *
        # Scale to -100 to 100
        100
    ).round(1)

    # Add a function to get illustrative representation
    def get_sentiment_breakdown(row):
        score = row['sentiment_score']
        if score == 0:
            return "100% Neutral"

        if abs(score) == 100:
            return f"100% {'Positive' if score > 0 else 'Negative'}"

        main_sentiment = "Positive" if score > 0 else "Negative"
        abs_score = abs(score)

        return f"{abs_score}% {main_sentiment}, {100-abs_score}% Neutral"

    df['sentiment_interpretation'] = df.apply(get_sentiment_breakdown, axis=1)

    return df

aggregated_daily_sentiment_scaled = convert_to_sentiment_scale(aggregated_daily_sentiment)


# Join sentiment data (aggregated and scaled) with ticker data
def join_sentiment_ticker(sentiment_df, ticker_df):
    #Join daily sentiment data with ticker data, handling missing dates and setting appropriate defaults.

    # Ensure both dataframes have datetime index
    if not isinstance(sentiment_df.index, pd.DatetimeIndex):
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
    if not isinstance(ticker_df.index, pd.DatetimeIndex):
        ticker_df.index = pd.to_datetime(ticker_df.index)

    # Create copy of ticker_df to avoid modifying original
    result_df = ticker_df.copy()

    # Create the existing_news column first (before join)
    result_df['existing_news'] = result_df.index.isin(sentiment_df.index)

    # Perform left join using date index
    result_df = result_df.join(sentiment_df, how='left')

    # Fill sentiment_score with 0 for days without news
    result_df['sentiment_score'] = result_df['sentiment_score'].fillna(0)

    # Optional: Add columns to show what kind of day it is
    result_df['day_type'] = result_df.apply(
        lambda row: 'News Day' if row['existing_news'] else 'No News Day',
        axis=1
    )

    return result_df



clipped_ticker_data.set_index('publish_date', inplace=True)
ticker_sentiment_data = join_sentiment_ticker(aggregated_daily_sentiment_scaled, clipped_ticker_data)

ticker_sentiment_data.to_csv('ticker_sentiment_data.csv')

