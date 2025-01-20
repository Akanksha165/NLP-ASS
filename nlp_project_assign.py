import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Ensure the necessary resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Use tkinter to open a file dialog and select the CSV file
root = Tk()
root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
file_path = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

# Check if a file was selected
if not file_path:
    print("No file selected. Exiting...")
    exit()

# Load dataset from the selected CSV file
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Check if the expected columns are present
if 'review' not in df.columns or 'sentiment' not in df.columns:
    print("CSV file does not contain the required columns 'review' and 'sentiment'. Exiting...")
    exit()

# Preprocess the reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# Debugging: Print unique labels in y_test and y_pred
print("Unique labels in y_test:", y_test.unique())
print("Unique labels in y_pred:", pd.Series(y_pred).unique())

# Debugging: Print some predictions and actual labels
print("\nSample predictions vs actual labels:")
for i in range(10):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")

# Confusion Matrix
labels = ['positive', 'negative', 'neutral']
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()