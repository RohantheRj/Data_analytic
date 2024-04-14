import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


text = "Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."


preprocessed_text = re.sub(r'[^a-zA-Z\s]', '', text)


sentences = sent_tokenize(preprocessed_text)


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)
similarity_matrix = cosine_similarity(tfidf_matrix)


N = 2
top_sentences = sorted(range(len(similarity_matrix[-1])), key=lambda i: similarity_matrix[-1][i])[-N:]


summary = ''
for i in top_sentences:
    summary += sentences[i] + ' '

print(summary)

