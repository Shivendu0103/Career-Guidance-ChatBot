from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

text = "Test sentence for NLTK."
print(word_tokenize(text))
