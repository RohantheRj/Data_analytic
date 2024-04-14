import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from heapq import nlargest

# Sample text paragraph you can write any text
text = "Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human languages, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. The history of natural language processing generally started in the 1950s, although work can be found from earlier periods."


text = re.sub('[^a-zA-Z]', ' ', text)


sentences = sent_tokenize(text)


stop_words = set(stopwords.words('english'))
words = []
for sentence in sentences:
    words.extend(word_tokenize(sentence))
words = [word.lower() for word in words if word.lower() not in stop_words]


word_freq = nltk.FreqDist(words)


sentence_scores = {}
for sentence in sentences:
    for word in word_tokenize(sentence.lower()):
        if word in word_freq:
            if len(sentence.split(' ')) < 30:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]


summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)

print(summary)

