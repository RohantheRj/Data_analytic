import re

text = "Hello, #world123! This is a sample text paragraph. It contains special characters and 5 digits."


processed_text = re.sub(r'[^a-zA-Z\s]', '', text)

print(processed_text)

