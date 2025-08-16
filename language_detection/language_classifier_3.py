import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os


def train_and_save_model(data_path, model_path):
    """
    Loads data, trains a language detection model, and saves it.

    Args:
        data_path (str): Path to the cleaned CSV data.
        model_path (str): Path to save the trained model file.

    Returns:
        float: The accuracy of the trained model on the test set.
    """
    try:
        # Step 1: Load the data
        data = pd.read_csv(data_path)
        if "Text" not in data.columns or "Language" not in data.columns:
            print("Error: Data must contain 'Text' and 'Language' columns.")
            return None

        # Step 2: Define features (X) and labels (y)
        X = data["Text"]
        y = data["Language"]

        # Step 3: Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Step 4: Create a pipeline (TF-IDF + Logistic Regression)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 5), max_features=20000)),
            ('clf', LogisticRegression(max_iter=200, solver='lbfgs', n_jobs=-1))
        ])

        # Step 5: Train the model
        pipeline.fit(X_train, y_train)

        # Step 6: Evaluate the model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Step 7: Save the trained model
        joblib.dump(pipeline, model_path)

        # Step 8: Return accuracy
        return accuracy

    except Exception as e:
        print(f"Error during training: {e}")
        return None


# --- Main execution block ---
if __name__ == "__main__":
    # Define file paths
    CLEAN_DATA_FILE = 'Language_Detection_Cleaned.csv'
    MODEL_OUTPUT_FILE = 'language_detection_model.joblib'

    model_accuracy = train_and_save_model(CLEAN_DATA_FILE, MODEL_OUTPUT_FILE)

    if model_accuracy is not None:
        print("\n--- Training Complete ---")
        print(f"Model saved to: {MODEL_OUTPUT_FILE}")
        print(f"Model Accuracy on Test Data: {model_accuracy:.4f}")
    else:
        print("\nModel training failed. Please check the error messages above.")

    print("\n--- Testing the saved model with sample inputs ---")
    if os.path.exists(MODEL_OUTPUT_FILE):
        loaded_model = joblib.load(MODEL_OUTPUT_FILE)

        sample_inputs = [
            "This is a test of the language detection system.",  # English
            "Ceci est un test du système de détection de langue.",  # French
            "Este es un sistema de prueba de detección de idioma.",  # Spanish
            "Это тест системы определения языка.",  # Russian
            "هذا اختبار لنظام كشف اللغة",  # Arabic
            "Questo è un test del sistema di rilevamento della lingua.",  # Italian
            "Dit is een test van het taaldetectiesysteem.",  # Dutch
            "Bu, İsveç dilinde yazılmış bir cümledir.",  # Turkish
            "Dette er en sætning skrevet på dansk."  # Danish
        ]

        # Make predictions
        predictions = loaded_model.predict(sample_inputs)

        # Print results
        for i, text in enumerate(sample_inputs):
            print(f"Input: '{text}'")
            print(f"--> Predicted Language: {predictions[i]}\n")
    else:
        print(f"Error: Model file '{MODEL_OUTPUT_FILE}' not found. Cannot run demonstration.")
