import re
from nltk.tokenize import sent_tokenize


text = "So, keep working. Keep striving. Never give up. Fall down seven times, get up eight. Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than hardship. So, keep moving, keep growing, keep learning. See you at work."


text = re.sub('[^A-Za-z]+', ' ', text)


sentences = sent_tokenize(text)


scores = {}
for sentence in sentences:
    words = sentence.split()
    score = len(words)
    scores[sentence] = score


sorted_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)


summary_sentences = [sentence[0] for sentence in sorted_sentences[:2]]
summary = " ".join(summary_sentences)


print(summary)

