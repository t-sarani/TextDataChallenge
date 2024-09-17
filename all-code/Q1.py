######################
#Q1
######################

#pip install wordcloud
#pip install numpy
#pip install matplotlib
#pip install PyPDF2
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PyPDF2
import re
from PyPDF2 import PdfReader

pdf_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q1_Khayyam Question\A_Conversation_with_Kanti_Mardia.pdf'

# Initialize the extracted text variable.
text = ''

# Read the PDF file.
with open(pdf_path, 'rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)

    # Extract text from each page.
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()  # Extract text from page.
        if extracted_text:  # Check if text was extracted successfully.
            text += extracted_text  # Concatenate the text.

# Print part of the extracted text to inspect it.
print("Extracted Text Preview:")
print(text[:2000])  # Print the first 2000 characters for inspection.

# Clean up and normalize the text.
cleaned_text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces.

# Define the search pattern, allowing for flexibility in matching.
search_pattern = r'Omar\s*Khayyam'  # Allows for multiple spaces/newlines.

# Search for all matches using the defined regex pattern.
matches = re.findall(search_pattern, cleaned_text, re.IGNORECASE)

# Count the occurrences and print them.
count = len(matches)
print(f"The term 'Omar Khayyam' appears {count} times in the paper.")

# Check for specific occurrences and display them.
if count > 0:
    print("\nOccurrences found:")
    for match in matches:
        print(match)  # Print each match found.
else:
    print("No occurrences found.")

###############

with open(pdf_path, 'rb') as f:
  pdf_reader = PyPDF2.PdfReader(f)
  text = ''
  for page in pdf_reader.pages:
    text += page.extract_text()

text = text.lower()
text = re.sub(r'[^\w\s]', '', text)
words = text.split()

word_counts = {}
for word in words:
  if word in word_counts:
    word_counts[word] += 1
  else:
    word_counts[word] = 1

wordcloud = WordCloud(width=800, height=600).generate_from_frequencies(word_counts)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#pip install nltk

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


with open(r'C:\Users\USER\Desktop\Neyshabur competition\data\Q1_Khayyam Question\A_Conversation_with_Kanti_Mardia.pdf', 'rb') as f:
  pdf_reader = PyPDF2.PdfReader(f)
  text = ''
  for page in pdf_reader.pages:
    text += page.extract_text()

text = text.lower()
text = re.sub(r'[^\w\s]', '', text)
words = text.split()

filtered_words = [word for word in words if not word in stop_words]

word_counts = {}
for word in filtered_words:
  if word in word_counts:
    word_counts[word] += 1
  else:
    word_counts[word] = 1

wordcloud = WordCloud(width=800, height=600).generate_from_frequencies(word_counts)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


mask_image = np.array(Image.open(r'C:\Users\USER\Desktop\Neyshabur competition\images .jpg'))

wordcloud = WordCloud(width=800, height=600, background_color='white', mask=mask_image).generate_from_frequencies(word_counts)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
