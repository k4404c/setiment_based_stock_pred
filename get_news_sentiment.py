'''
In this script, we will use the FinBERT model to get sentiment scores for each news article in the dataset.
The FinBERT model gives us probabilities for each of the following labels: positive, negative, and neutral.
We save these probabilities and the label with the highest probability to the DataFrame.
We then export the DataFrame to a new CSV file.
'''

import pandas as pd

data = pd.read_csv('news_data.csv')
#drop non important columns
data.drop(columns= ['Unnamed: 0', 'relationships.secondaryTickers.data', 'relationships.author.data.id', 'relationships.primaryTickers.data', 'relationships.secondaryTickers.data', 'type', 'attributes.isLockedPro', 'attributes.gettyImageUrl', 'attributes.videoPreviewUrl','attributes.videoDuration','relationships.author.data.type', 'relationships.otherTags.data' ], axis=1, inplace=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import scipy

# load model nad tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment_scores(text):
  inputs = tokenizer(text, return_tensors="pt")
  with torch.no_grad():
    logits = model(**inputs).logits
  scores = {k: v for k, v in zip(model.config.id2label.values(), scipy.special.softmax(logits.numpy().squeeze()))}
  return scores

def add_sentiment_scores(row):
    scores = get_sentiment_scores(row['attributes.title'])
    # Get the label with the highest score
    label = max(scores, key=scores.get)
    # Update row with scores and label
    row['positive'] = scores['positive']
    row['negative'] = scores['negative']
    row['neutral'] = scores['neutral']
    row['label'] = label
    return row

# Apply the function to each row of the DataFrame
data = data.apply(add_sentiment_scores, axis=1)

# Export to CSV
data.to_csv('news_data_with_sentiment.csv')