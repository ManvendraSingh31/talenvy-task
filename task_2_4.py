# -*- coding: utf-8 -*-
"""task_2_4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1P48tpSRTcZ4pVVQsRsZpWRuEqYFF_VmE
"""

# 1. Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')

# 2. Sample Dataset
data = {
    "review": [
        "This product is fantastic! Highly recommended.",
        "Terrible experience. Waste of money.",
        "It's okay, not great but not bad either.",
        "Absolutely love it! Five stars.",
        "Do not buy this. It broke on the first use."
    ]
}
df = pd.DataFrame(data)

# 3. VADER Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df['vader_score'] = df['review'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['vader_sentiment'] = df['vader_score'].apply(
    lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
)

# 4. TextBlob Polarity (Optional for comparison)
df['textblob_polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['textblob_sentiment'] = df['textblob_polarity'].apply(
    lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
)

# 5. Insights Summary
print("VADER Sentiment Counts:")
print(df['vader_sentiment'].value_counts())

# 6. Visualization
sns.countplot(data=df, x='vader_sentiment', palette='coolwarm')
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# 7. Output Table
df[['review', 'vader_score', 'vader_sentiment', 'textblob_polarity', 'textblob_sentiment']]