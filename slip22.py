import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


text = "Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."


words = word_tokenize(text)


stop_words = set(stopwords.words('english'))


filtered_words = [word for word in words if word.casefold() not in stop_words]


filtered_sentence = ' '.join(filtered_words)

print(filtered_sentence)

