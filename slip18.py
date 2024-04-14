
!pip install nltk matplotlib wordcloud


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud


nltk.download('stopwords')


text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed tristique ante et velit vestibulum, vel pharetra orci iaculis. Nullam mattis risus quis augue tincidunt rhoncus. Morbi varius, arcu vitae scelerisque laoreet, magna est imperdiet quam, sit amet ultrices lectus justo id enim. Sed dictum suscipit commodo. Sed maximus consequat risus, nec pharetra nibh interdum quis. Etiam eget quam vel augue dictum dignissim sit amet nec elit. Nunc at sapien dolor. Nulla vitae iaculis lorem. Suspendisse potenti. Sed non ante turpis. Morbi consectetur, arcu a vestibulum suscipit, mauris eros convallis nibh, nec feugiat orci enim sit amet enim. Aliquam erat volutpat. Etiam vel nisi id neque viverra dapibus non non lectus."


words = word_tokenize(text.lower())
sentences = sent_tokenize(text)


stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.casefold() not in stop_words]


fdist = FreqDist(filtered_words)
fdist.plot(30, cumulative=False)
plt.show()


wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stop_words,
                      min_font_size=10).generate(text)
# Plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

