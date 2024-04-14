import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Text paragraph
text = "Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."

# Tokenize the text
tokens = nltk.word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if not word.lower() in stop_words]

# Print the filtered tokens
print(filtered_tokens)

