import pandas as pd
import matplotlib.pyplot as plt

# Define the file paths
honda_2008_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2008_honda_accord.pdf'
honda_2009_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_honda_accord.pdf'
hyundai_2009_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_hyundai_sonata.pdf'
toyota_2009_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_toyota_corolla.pdf'

car_labels = ['Toyota Corolla 2009', 'Hyundai Sonata 2009', 'Honda Accord 2009', 'Honda Accord 2008']

# Sample DataFrame definition (replace with your actual data)
data = {
'Model': ['toyota_corolla', 'hyundai_sonata', 'honda_accord', 'honda_accord', 'toyota_corolla'],
'Year': [2009, 2009, 2009, 2008, 2009],
'Comments': ['Good car', 'Nice ride', 'Very reliable', 'Old but gold', 'Affordable']
}
df = pd.DataFrame(data)

comment_counts = [
len(df[df['Model'] == 'toyota_corolla']),
len(df[df['Model'] == 'hyundai_sonata']),
len(df[(df['Model'] == 'honda_accord') & (df['Year'] == 2009)]),
len(df[(df['Model'] == 'honda_accord') & (df['Year'] == 2008)])
]

print('Number of comments given for each car')
for label, count in zip(car_labels, comment_counts):
    print(f'Number of comments for {label}: {count}')

colors = ['red', 'yellow', 'purple', 'orange']
plt.figure(figsize=(10, 6))
plt.bar(car_labels, comment_counts, color=colors)
plt.ylabel('Number of Comments')
plt.title('Number of Comments for Each Car')
plt.show()

import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import PyPDF2

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)
sentiment_analyser = SentimentIntensityAnalyzer()

# Define the file paths (these paths are still necessary if you plan to extract comments from PDFs)
honda_2008_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2008_honda_accord.pdf'
honda_2009_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_honda_accord.pdf'
hyundai_2009_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_hyundai_sonata.pdf'
toyota_2009_path = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_toyota_corolla.pdf'

# Sample DataFrame definition (replace with your actual data)
data = {
    'Model': ['toyota_corolla', 'hyundai_sonata', 'honda_accord', 'honda_accord', 'toyota_corolla'],
    'Year': [2009, 2009, 2009, 2008, 2009],
    'Comments': ['Good car', 'Nice ride', 'Very reliable', 'Old but gold', 'Affordable']
}
df = pd.DataFrame(data)

# Print DataFrame to check its structure
print("DataFrame structure before sentiment analysis:")
print(df)

def get_sentiment_score(text):
    scores = sentiment_analyser.polarity_scores(text)
    return scores["compound"]

def get_sentiment_class(score):
    if score > 0:
        return "Positive"
    elif score == 0:
        return "Neutral"
    else:
        return "Negative"

# Change 'Comment' to 'Comments' to match your DataFrame
scores = df["Comments"].apply(get_sentiment_score)

# Create a new column 'SentimentScore' in the DataFrame
df["SentimentScore"] = scores

# Classify sentiment based on the scores
sentiment_class = df["SentimentScore"].apply(get_sentiment_class)
df["SentimentCategory"] = sentiment_class

# Print the resulting DataFrame to view results
print("DataFrame after sentiment analysis:")
print(df[['Model', 'Year', 'Comments', 'SentimentScore', 'SentimentCategory']])

# Optional: Plotting the sentiment categories
df["SentimentCategory"].value_counts().plot(kind='bar')
plt.title('Sentiment Analysis of Comments')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Comments')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.show()


######################نمودار چگالی

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# فرض کنید که داده‌های شما در قالب یک DataFrame به نام df_2009 موجود است
# صدای ایجاد داده‌های تصادفی برای تولید آزمایشی
data = {
    'Model': ['honda_accord'] * 100 + ['hyundai_sonata'] * 100 + ['toyota_corolla'] * 100,
    'SentimentScore': [0.8] * 50 + [0.9] * 50 + [0.1] * 50 + [0.3] * 50 + [0.7] * 50 + [0.6] * 50
}
df_2009 = pd.DataFrame(data)

# فیلتر کردن داده‌ها
models = ['honda_accord', 'hyundai_sonata', 'toyota_corolla']

# تعریف رنگ‌ها و برچسب‌ها
colors = ['red', 'blue', 'purple']
labels = ['Honda Accord', 'Hyundai Sonata', 'Toyota Corolla']

# رسم نمودار چگالی
plt.figure(figsize=(10, 6))
for model, color, label in zip(models, colors, labels):
    sns.kdeplot(data=df_2009[df_2009['Model'] == model]['SentimentScore'], 
                 color=color, label=label, fill=True)

plt.title('Density Distribution of Sentiment Scores by Car Model')
plt.xlabel('Sentiment Score')
plt.ylabel('Density')
plt.xlim([-1, 2])
plt.legend()
plt.show()

# ایجاد زیرنمودارها برای هر مدل خودرو
fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(15, 5))
for ax, model, color, label in zip(axes, models, colors, labels):
    sns.histplot(data=df_2009[df_2009['Model'] == model], 
                 x='SentimentScore', 
                 bins=20, 
                 kde=True, 
                 color=color, 
                 ax=ax, 
                 label=label)
    ax.set_title(label)
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()

###############################هیستوگرام
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure NLTK VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzer
sentiment_analyser = SentimentIntensityAnalyzer()

# Sample DataFrame definition (make sure the lengths of lists match)
data = {
    'Model': ['toyota_corolla', 'hyundai_sonata', 'honda_accord', 'honda_accord', 'toyota_corolla'],
    'Year': [2009, 2009, 2009, 2008, 2009],
    'Comments': ['Good car', 'Nice ride', 'Very reliable', 'Old but gold', 'Affordable']
}

# Create the DataFrame
df_2009 = pd.DataFrame(data)

# Assuming you would calculate sentiment scores
df_2009["SentimentScore"] = df_2009["Comments"].apply(lambda x: sentiment_analyser.polarity_scores(x)["compound"])

# Ensure the data for plotting sentiment scores is available
# Creating histograms for each car model
data_to_plot = [
    df_2009[df_2009["Model"] == "honda_accord"]["SentimentScore"],
    df_2009[df_2009["Model"] == "hyundai_sonata"]["SentimentScore"],
    df_2009[df_2009["Model"] == "toyota_corolla"]["SentimentScore"]
]

colors = ["red", "blue", "purple"]
labels = ["Honda Accord", "Hyundai Sonata", "Toyota Corolla"]

# Histogram for combined sentiment scores
plt.hist(data_to_plot, alpha=0.5, color=colors, label=labels, bins=10, density=True)
plt.title("Density Distribution of Sentiment Scores by Car Model")
plt.xlabel("Sentiment Score")
plt.ylabel("Density")
plt.legend()
plt.show()

# Create individual histograms for each car model
models = ["honda_accord", "hyundai_sonata", "toyota_corolla"]
fig, axes = plt.subplots(nrows=len(models), figsize=(10, 5))

for ax, model, color in zip(axes, models, colors):
    data = df_2009[df_2009["Model"] == model]["SentimentScore"]
    ax.hist(data, bins=20, alpha=0.5, color=color, density=True)
    ax.set_title(f"Sentiment Scores for {model.replace('_', ' ').title()}")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Density")

plt.tight_layout()
plt.show()

# Calculate and print skewness
skewness_values = []
for model in models:
    data = df_2009[df_2009["Model"] == model]["SentimentScore"]
    skewness = stats.skew(data)
    skewness_values.append(skewness)

# Sort and print skewness rankings
sorted_models = sorted(zip(models, skewness_values), key=lambda x: x[1])
print("Skewness Ranking:")
for rank, (model, skew) in enumerate(sorted_models, start=1):
    print(f"{rank}. {model.replace('_', ' ').title()}: {skew:.2f}")

import pandas as pd
from scipy import stats
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import PyPDF2

# Ensure NLTK VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Initialize the sentiment analyser
sentiment_analyser = SentimentIntensityAnalyzer()

# Define the file paths (these paths will be used to extract comments from PDFs)
honda_2008_data = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2008_honda_accord.pdf'
honda_2009_data = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_honda_accord.pdf'
hyundai_2009_data = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_hyundai_sonata.pdf'
toyota_2009_data = r'C:\Users\USER\Desktop\Neyshabur competition\data\Q2_Main Question\2009_toyota_corolla.pdf'

def extract_comments_from_pdf(file_path):
    comments = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            comments.append(page.extract_text())
    return comments

# Extract comments from PDFs (assumes each PDF contains comments under a specific structure)
honda_2009_comments = extract_comments_from_pdf(honda_2009_data)
hyundai_2009_comments = extract_comments_from_pdf(hyundai_2009_data)
toyota_2009_comments = extract_comments_from_pdf(toyota_2009_data)

# Combine comments into a DataFrame
data = {
    'Model': ['honda_accord'] * len(honda_2009_comments) + \
             ['hyundai_sonata'] * len(hyundai_2009_comments) + \
             ['toyota_corolla'] * len(toyota_2009_comments),
    'Comments': honda_2009_comments + hyundai_2009_comments + toyota_2009_comments
}
df = pd.DataFrame(data)

# Calculate sentiment scores
df['SentimentScore'] = df['Comments'].apply(lambda comment: sentiment_analyser.polarity_scores(comment)['compound'])

# Separate the DataFrame by models for t-tests
honda_2009_scores = df[df['Model'] == 'honda_accord']['SentimentScore']
hyundai_2009_scores = df[df['Model'] == 'hyundai_sonata']['SentimentScore']
toyota_2009_scores = df[df['Model'] == 'toyota_corolla']['SentimentScore']

# Perform t-tests:
# Comparing Honda and Hyundai
# One-sided
print("Testing if the mean of Honda is less than that of Hyundai")
print(stats.ttest_ind(honda_2009_scores, hyundai_2009_scores, alternative="less"))

# Two-sided
print("Testing if the mean of Honda is unequal to that of Hyundai")
print(stats.ttest_ind(honda_2009_scores, hyundai_2009_scores, alternative="two-sided"))

print("------------------------------")

# Comparing Honda and Toyota
# One-sided
print("Testing if the mean of Honda is less than that of Toyota")
print(stats.ttest_ind(honda_2009_scores, toyota_2009_scores, alternative="less"))

# Two-sided
print("Testing if the mean of Honda is unequal to that of Toyota")
print(stats.ttest_ind(honda_2009_scores, toyota_2009_scores, alternative="two-sided"))

print("------------------------------")

# Comparing Toyota and Hyundai
# One-sided
print("Testing if the mean of Toyota is less than that of Hyundai")
print(stats.ttest_ind(toyota_2009_scores, hyundai_2009_scores, alternative="less"))

# Two-sided
print("Testing if the mean of Toyota is unequal to that of Hyundai")
print(stats.ttest_ind(toyota_2009_scores, hyundai_2009_scores, alternative="two-sided"))

##################################آمار توصیفی

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from scipy import stats
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer



# Sample DataFrame definition (replace with your actual data)
data = {
'Model': ['toyota_corolla', 'hyundai_sonata', 'honda_accord', 'honda_accord', 'toyota_corolla'],
'Year': [2009, 2009, 2009, 2008, 2009],
'Comments': ['Good car', 'Nice ride', 'Very reliable', 'Old but gold', 'Affordable']
}
df = pd.DataFrame(data)

# Initialize the sentiment analyzer
sentiment_analyser = SentimentIntensityAnalyzer()


# Function to get sentiment score
def get_sentiment_score(text):
    scores = sentiment_analyser.polarity_scores(text)
    return scores["compound"]

# Apply the sentiment score function to the 'Comments' column
df["SentimentScore"] = df["Comments"].apply(get_sentiment_score)


# Filter the DataFrame for the year 2009
df_2009 = df[df["Year"] == 2009]

# Descriptive statistics of sentiment scores by model
descrip_stats = df_2009.groupby("Model")["SentimentScore"].agg(["min", "max", "mean", "median", "var", "std"])
print("Descriptive statistics of sentiment scores by model:\n", descrip_stats, "\n")


