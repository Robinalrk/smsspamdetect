import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
# Load the processed dataset
data = pd.read_csv("processed_dataset.csv")
print("Loaded processed data:")
print(data.head())

# Fill missing values and ensure strings
data['Message'] = data['Message'].fillna('').astype(str)

# Verify there are no NaN values and all are strings
assert data['Message'].isnull().sum() == 0, "Missing values still present in 'Message'"
assert data['Message'].apply(lambda x: isinstance(x, str)).all(), "Non-string values found in 'Message'"

# Split into features and labels
X = data['Message']
y = data['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()
joblib.dump(tfidf,"tfidfVectorizer.joblib")
print("model saved as")

# Models and Evaluation
models = {
    # "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    # "SVM": SVC(kernel='linear'),
    # "Naive Bayes": MultinomialNB()
}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train_tfidf, y_train)
    
    # Save the trained model
    model_file = f"{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_file)
    print(f"{model_name} model saved as '{model_file}'")
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    
    
    # Evaluate
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    

# After training
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('static/confusion_matrix.png')  # Save as static file

