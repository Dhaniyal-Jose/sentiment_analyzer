import pandas as pd

# Load the dataset
df = pd.read_csv('C:/Users/dhani/Downloads/sentiment_analyzer/data/sentiment_analysis.csv')

# Print class distribution
print(df['sentiment'].value_counts())
