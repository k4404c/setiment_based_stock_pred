import unittest
import pandas as pd
import numpy as np

class TestTickerSentimentData(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the generated CSV file
        cls.result_df = pd.read_csv('ticker_sentiment_data.csv')
        cls.result_df['publish_date'] = pd.to_datetime(cls.result_df['publish_date'])
    
    def test_columns_exist(self):
        # Check if the expected columns are present
        expected_columns = [
            'publish_date', 'existing_news', 'sentiment_score', 
            'day_type', 'prob_neutral', 'prob_positive', 
            'prob_negative', 'total_comments', 'voted_label',
            'article_count', 'label_distribution', 'majority_label'
        ]
        for column in expected_columns:
            self.assertIn(column, self.result_df.columns, f"Missing column: {column}")
    
    def test_sentiment_score_range(self):
        # Test if sentiment_score is within the valid range [-100, 100]
        min_score = self.result_df['sentiment_score'].min()
        max_score = self.result_df['sentiment_score'].max()
        self.assertGreaterEqual(min_score, -100, "Sentiment score below -100")
        self.assertLessEqual(max_score, 100, "Sentiment score above 100")
    
    def test_existing_news_column(self):
        # Ensure `existing_news` is boolean
        self.assertTrue(
            self.result_df['existing_news'].isin([True, False]).all(),
            "existing_news column contains non-boolean values"
        )
    
    def test_day_type_values(self):
        # Ensure `day_type` only contains expected values
        valid_day_types = {'News Day', 'No News Day'}
        unique_day_types = set(self.result_df['day_type'].unique())
        self.assertTrue(
            unique_day_types.issubset(valid_day_types),
            f"Unexpected day_type values: {unique_day_types - valid_day_types}"
        )
    
    def test_no_null_publish_date(self):
        # Ensure there are no null values in the `publish_date` column
        self.assertFalse(
            self.result_df['publish_date'].isnull().any(),
            "publish_date column contains null values"
        )
    
    def test_article_count_non_negative(self):
        # Ensure article_count is non-negative
        self.assertTrue(
            (self.result_df['article_count'].fillna(0) >= 0).all(),
            "article_count contains negative values"
        )
    
    def test_sentiment_columns_nan_or_sum_to_one(self):
        # Test if sentiment probabilities are either NaN or sum to 1 (approximately)
        sentiment_columns = ['prob_neutral', 'prob_positive', 'prob_negative']
        
        def is_valid_row(row):
            if row.isna().all():
                return True  # All NaN is valid
            return abs(row.sum() - 1) < 0.01  # Sum approximately equals 1
        
        self.assertTrue(
            self.result_df[sentiment_columns].apply(is_valid_row, axis=1).all(),
            "Sentiment probabilities are invalid (neither NaN nor summing to 1)"
        )
    
    def test_label_distribution_format(self):
        # Check if label_distribution is valid even with missing data
        for value in self.result_df['label_distribution'].dropna():
            try:
                eval_dict = eval(value)
                self.assertIsInstance(eval_dict, dict, "label_distribution is not a dictionary")
            except Exception as e:
                self.fail(f"label_distribution parsing failed: {e}")
    
    def test_sentiment_score_for_no_news(self):
        # For days without news, sentiment_score should be 0
        no_news_df = self.result_df[self.result_df['existing_news'] == False]
        self.assertTrue(
            (no_news_df['sentiment_score'] == 0).all(),
            "Sentiment score is not 0 for days without news"
        )


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)