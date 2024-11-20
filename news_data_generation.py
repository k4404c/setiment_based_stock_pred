'''
In this script, we leverage the Seeking Alpha API to get news articles for a specific stock over a date range. 
We loop through the pages of the API response and concatenate the data into a single DataFrame.
We then save the DataFrame to a CSV file.
'''

import requests
import pandas as pd
import time
import os
#get requests for pages in range
url = "https://seeking-alpha.p.rapidapi.com/news/v2/list-by-symbol"
df_out = pd.DataFrame()

api_key= os.environ.get('API_KEY')
api_host= os.environ.get('API_HOST')

headers = {
	"x-rapidapi-key": api_key,
	"x-rapidapi-host": api_host
}
for i in range(1,150):
  querystring = {
    "until":"1731481200",
    "since":"1415862000",
    "size":"40",
    "number":str(i),
    "id":"intc"}
  response = requests.get(url, headers=headers, params=querystring)
  response.raise_for_status()
  data = response.json()
  df = pd.json_normalize(data['data'])
  # Append to main DataFrame
  df_out = pd.concat([df_out, df], ignore_index=True)

  # Add delay to avoid hitting rate limits
  time.sleep(1)

# Save to CSV
df_out.to_csv('news_data.csv')