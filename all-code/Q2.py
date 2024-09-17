#################
#Q2
#################

####################
#Data Preprocessing
####################

#pip install pdfminer.six
#pip install pdfplumber
import os
import pdfminer.high_level
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import pdfplumber
import re
import os
from pdfplumber import open as pdf_open

# Define the file paths
honda_2008_path = (r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2008_honda_accord.pdf')
honda_2009_path = (r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_honda_accord.pdf')
hyundai_2009_path = (r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_hyundai_sonata.pdf')
toyota_2009_path = (r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_toyota_corolla.pdf')

# Define a list of words to remove
words_to_remove = ['<DOC>','</DOC>','<DOCNO>','<TEXT>','<AUTHOR>','<DATE>','</TEXT>',
                   '</DATE>','</DOCNO>','</AUTHOR>','</FAVORITE>','<FAVORITE>']

# Function to extract text from a PDF file and remove specified words
def extract_text_from_pdf(pdf_path, words_to_remove):
    with pdf_open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()

    # Remove the specified words
    for word in words_to_remove:
        text = text.replace(word, '')

    return text.strip()

# Extract text from each PDF file and remove specified words
honda_2008_text = extract_text_from_pdf(honda_2008_path, words_to_remove)
honda_2009_text = extract_text_from_pdf(honda_2009_path, words_to_remove)
hyundai_2009_text = extract_text_from_pdf(hyundai_2009_path, words_to_remove)
toyota_2009_text = extract_text_from_pdf(toyota_2009_path, words_to_remove)

# Combine the text from all PDF files
all_text = ' '.join([honda_2008_text, honda_2009_text, hyundai_2009_text, toyota_2009_text])

# Print the extracted text
print("Honda 2008 text:")
print(honda_2008_text)
print("\nHonda 2009 text:")
print(honda_2009_text)
print("\nHyundai 2009 text:")
print(hyundai_2009_text)
print("\nToyota 2009 text:")
print(toyota_2009_text)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Plot the word cloud
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud')
plt.show()

###############################
#sensitivity analysis section 1
###############################

import matplotlib.pyplot as plt
from textblob import TextBlob

#function for sensitivity analysis and returning positive score
def analyze_sentiment(text):
  analysis = TextBlob(text)
  return analysis.sentiment.polarity

from textblob import TextBlob

#computing sensitivity score for each car
honda_2009_sentiment = analyze_sentiment(honda_2008_text)
hyundai_2009_sentiment = analyze_sentiment(hyundai_2009_text)
toyota_2009_sentiment = analyze_sentiment(toyota_2009_text)

#scores comparison and determining the car with the highest customer satisfaction
car_sentiments = {
honda_2009_text: honda_2009_sentiment,
hyundai_2009_text : hyundai_2009_sentiment,
toyota_2009_text: toyota_2009_sentiment
}

best_car = max(car_sentiments, key=car_sentiments.get)
print(f'The car with the highest customer satisfaction in 2009 is: {best_car}')

texts = {
honda_2009_text: honda_2009_text,
hyundai_2009_text : hyundai_2009_text,
toyota_2009_text :  toyota_2009_text
}

#computing sensitivity scores for each car
sentiments = {car: analyze_sentiment(text) for car, text in texts.items()}

#display the results in a bar chart
plt.bar(sentiments.keys(), sentiments.values(), color=['blue', 'green', 'red'])
plt.xlabel('Car Model')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of Car Reviews in 2009')
plt.show()

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(best_car)

# Plot the word cloud
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud')
plt.show()

##################################
#count_comments
##################################


# تابعی برای شمارش تعداد نظرات بر اساس نشانههای پایان جمله
def count_comments(text):
# فرض میکنیم که هر نظر با یک نقطه پایان مییابد
  comments = re.split(r'[  .!"؟) + ]', text)
# حذف فضاهای خالی و جملات خالی
  comments = [comment.strip() for comment in comments if comment.strip() != '']
  return len(comments)



# شمارش تعداد نظرات برای هر بررسی

honda_2008_comments_count = count_comments(honda_2008_text)
honda_2009_comments_count = count_comments(honda_2009_text)
hyundai_2009_comments_count = count_comments(hyundai_2009_text)
toyota_2009_comments_count = count_comments(toyota_2009_text)



# چاپ تعداد نظرات
print(f"Honda 2008 Comments Count: {honda_2008_comments_count}")
print(f"Honda 2009 Comments Count: {honda_2009_comments_count}")
print(f"Hyundai 2009 Comments Count: {hyundai_2009_comments_count}")
print(f"Toyota 2009 Comments Count: {toyota_2009_comments_count}")