import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Path to the folder containing text files
folder_path = "./transcriptions"  # Replace with your folder path

# Step 1: Read all text files and combine their content
combined_transcript = ""

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Check for text files
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            combined_transcript += file.read() + " "  # Add a space between files

#print("Combined Transcript:\n", combined_transcript)

# Step 2: Preprocess the Combined Transcript
tokens = word_tokenize(combined_transcript.lower())  # Tokenize and convert to lowercase
filtered_words = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]

# Step 3: Calculate Vocabulary Metrics
unique_words = set(filtered_words)
vocab_size = len(unique_words)
lexical_diversity = vocab_size / len(filtered_words)

# Step 4: Check Against Predefined Vocabulary List
professional_vocab = {"self-introduction", "background", "experience", "skills", "strengths", "strengths", "skills", "abilities", "competencies", "qualities",  "weaknesses", "areas of improvement", "challenges", "growth areas", "self-awareness",  "future goals", "career aspirations", "long-term vision", "career path", "professional development",  "qualifications", "fit for the role", "skills", "experience", "value", "contribution"}
used_vocab = unique_words.intersection(professional_vocab)
missed_vocab = professional_vocab - used_vocab

# Step 5: Sentiment Analysis
blob = TextBlob(combined_transcript)
sentiment_score = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

# Step 6: Provide Feedback
print("\n--- Vocabulary Analysis Feedback ---")
print(f"Total Words Analyzed: {len(filtered_words)}")
print(f"Vocabulary Size: {vocab_size}")
print(f"Lexical Diversity: {lexical_diversity:.2f}")
print(f"Sentiment: {sentiment}")
print(f"Used Professional Vocabulary: {used_vocab}")
print(f"Missed Professional Vocabulary: {missed_vocab}")

if len(missed_vocab) > 0:
    print("\nSuggestion: Try incorporating these words for a stronger impact:")
    print(", ".join(missed_vocab))
