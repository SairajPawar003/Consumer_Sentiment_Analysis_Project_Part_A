from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
training_data = [
    "I love this movie", 
    "This film was amazing", 
    "Best experience ever", 
    "I hated this movie", 
    "Worst film of all time", 
    "Absolutely terrible"
]
training_labels = [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative

# Step 1: Initialize and fit the CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_data)

# Step 2: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, training_labels)

# Step 3: Test with new reviews
test_reviews = ["I love this", "This was terrible", "Amazing experience", "Worst movie ever"]
X_test = vectorizer.transform(test_reviews)

# Step 4: Predict sentiment
predictions = model.predict(X_test)

# Step 5: Print predictions
for review, sentiment in zip(test_reviews, predictions):
    label = "Positive" if sentiment == 1 else "Negative"
    print(f"Review: '{review}' -> Sentiment: {label}")
