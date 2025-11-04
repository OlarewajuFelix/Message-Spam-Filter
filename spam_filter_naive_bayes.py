# spam_filter_naive_bayes.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    # 1️⃣ Load dataset
    print("Loading dataset...")
    df = pd.read_csv("spambase_csv.csv")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head(), "\n")

    # 2️⃣ Split into features (X) and target (y)
    X = df.drop(columns=['class'])  # features
    y = df['class']                 # labels

    # 3️⃣ Split into training and testing sets
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}\n")

    # 4️⃣ Train Naive Bayes model
    print("Training Naive Bayes model...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Model training complete!\n")

    # 5️⃣ Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 6️⃣ Save the trained model
    joblib.dump(model, "spam_filter_naive_bayes.pkl")
    print("\n💾 Model saved as 'spam_filter_naive_bayes.pkl'")

if __name__ == "__main__":
    main()
