###############################
#sensitivity analysis section 2
###############################

#function for sensitivity analysis and returning positive score
def analyze_sentiment(text):
  analysis = TextBlob(text)
  return analysis.sentiment.polarity

#compare text file of 2008 & 2009 honda
texts = {
honda_2008_text: honda_2008_text,
honda_2009_text : honda_2009_text
}

#computing sensitivity scores for 2008 & 2009 honda
sentiments = {year: analyze_sentiment(text) for year, text in texts.items()}

#computing sensitivity scores
improvement = sentiments[honda_2009_text] - sentiments[honda_2008_text]
is_significant_improvement = abs(improvement) > 0.1  # تعیین اینکه آیا بهبود قابل توجهی وجود دارد

print(f"Sentiment score for 2008 Honda Accord: {sentiments[honda_2008_text]}")
print(f"Sentiment score for 2009 Honda Accord: {sentiments[honda_2009_text]}")
print(f"Improvement in sentiment from 2008 to 2009: {improvement}")
print(f"Is the improvement significant? {'Yes' if is_significant_improvement else 'No'}")

#computing sensitivity scores for each car
sentiments = {car: analyze_sentiment(text) for car, text in texts.items()}

#display the results in a bar chart
plt.bar(sentiments.keys(), sentiments.values(), color = ['blue', 'red'])
plt.xlabel('car model')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of Car Reviews in 2009')
plt.show()

Honda = ' '.join([honda_2008_text, honda_2009_text])
# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(Honda)

# Plot the word cloud
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud')
plt.show()

###################
#clustring for 2008
###################

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import re

data = honda_2008_text
print(data)

def remove_numbers(data):
  pattern = r"[0-9]"
  return re.sub(pattern, "", data)

text1_cleaned = remove_numbers(data)
print(f"{text1_cleaned}")

from sklearn.feature_extraction.text import TfidfVectorizer

cleaned_text = [text1_cleaned]
vectorizer = TfidfVectorizer()
vectorizer.fit(cleaned_text)

cleaned_documents = [text1_cleaned]  # List of cleaned text documents

# Split the text data into separate comments
comments1 = cleaned_documents[0].split('.')
comments1 = [comment.strip() for comment in comments1 if comment.strip()]

comments1

# Assign each comment to a cluster
cluster_labels = kmeans.labels_

# Print the results
for i, comment in enumerate(comments1):
    if i < len(cluster_labels):
        print(f'Comment: "{comment}"')
        print(f'Cluster: {cluster_labels[i]}')
        print()
    else:
        print(f'Comment: "{comment}"')
        print('Cluster: (No cluster assigned)')
        print()

# Vectorize the comments using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(comments1)

X

print(X)

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=7, max_iter=300 )

kmeans.fit(X)

kmeans.inertia_

from sklearn.metrics import silhouette_score

silhouette_score(X, kmeans.labels_)

############################
#two-cluster for 2008 & 2009
############################

cleaned_documents1 = [text2_cleaned]  # List of cleaned text documents

# Split the text data into separate comments
comments = cleaned_documents1[0].split('.')
comments = [comment.strip() for comment in comments if comment.strip()]

comments

# Assign each comment to a cluster
cluster_labels = kmeans.labels_

# Print the results
for i, comment in enumerate(comments):
    if i < len(cluster_labels):
        print(f'Comment: "{comment}"')
        print(f'Cluster: {cluster_labels[i]}')
        print()
    else:
        print(f'Comment: "{comment}"')
        print('Cluster: (No cluster assigned)')
        print()

# Vectorize the comments using TF-IDF
vectorizer = TfidfVectorizer()
X1 = vectorizer.fit_transform(comments)

X1

print(X1)

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=7, max_iter=300 )

kmeans.fit(X1)

kmeans.inertia_

from sklearn.metrics import silhouette_score

silhouette_score(X1, kmeans.labels_)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

honda_2008_text = extract_text_from_pdf(honda_2008_path, words_to_remove)

# Assuming extract_text_from_pdf returns a list of text reviews
honda_reviews = honda_2008_text

# Assuming honda_reviews is a long string containing all reviews separated by a delimiter (e.g., newline)
individual_reviews = honda_reviews.split("\n")  # Split by newline character
print(individual_reviews )
honda_2008_text = extract_text_from_pdf(honda_2008_path, words_to_remove)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(individual_reviews)
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores to identify the elbow point
import matplotlib.pyplot as plt
plt.plot(range_n_clusters, silhouette_scores)
plt.show()
