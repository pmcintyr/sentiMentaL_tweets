import nltk
import ssl

# Download the stopwords resource
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
