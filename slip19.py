import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


df = pd.read_csv('movie_review.csv')


df['Sentiment'] = df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)


pos_df = df[df['Sentiment'] > 0.2]


wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=STOPWORDS,
                      min_font_size=10).generate(' '.join(pos_df['Review']))


plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

