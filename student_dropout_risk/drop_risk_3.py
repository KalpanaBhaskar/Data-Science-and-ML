import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Ensemble Learning & Model Optimization Functions ---

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """
    Train a GradientBoostingClassifier.
    """
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    return gb_model


def apply_k_fold_cross_validation(model, X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int = 5) -> float:
    """
    Apply K-fold cross-validation and return the average accuracy.
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
    return cv_scores.mean()


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Evaluate the model using accuracy, precision, recall, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, cm

# --- Feature Selection, Handling Missing Values, Encoding Categorical Features, and Data Splitting ---

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features for the analysis.
    """
    features = ['Age at enrollment', 'Gender', 'Debtor', 'Tuition fees up to date', 'Curricular units 1st sem (approved)']
    X = df[features]
    return X


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    """
    X = X.copy()

    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X.loc[:, categorical_cols] = X[categorical_cols].fillna('Unknown')

    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    X.loc[:, numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

    return X


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features into numeric format using one-hot encoding.
    """
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# --- Main Execution Block ---
if __name__ == '__main__':
     
    file_path = 'dataset.csv'
    df = pd.read_csv(file_path)
    
    # Prepare features
    features = select_features(df)
    features = handle_missing_values(features)
    features = encode_categorical_features(features)

    # Target column (ensure your dataset has this)
    y = df['Target']  

    X_train, X_test, y_train, y_test = split_data(features, y)

    # --- Random Forest ---
    rf_model = train_random_forest(X_train, y_train)
    rf_cv_score = apply_k_fold_cross_validation(rf_model, X_train, y_train)
    print("Random Forest Cross-validation Accuracy:", rf_cv_score)

    rf_accuracy, rf_precision, rf_recall, rf_cm = evaluate_model(rf_model, X_test, y_test)
    print("\nRandom Forest Model Accuracy:", rf_accuracy)
    print("Random Forest Model Precision:", rf_precision)
    print("Random Forest Model Recall:", rf_recall)
    print("Random Forest Confusion Matrix:\n", rf_cm)

    # --- Gradient Boosting ---
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_cv_score = apply_k_fold_cross_validation(gb_model, X_train, y_train)
    print("\nGradient Boosting Cross-validation Accuracy:", gb_cv_score)

    gb_accuracy, gb_precision, gb_recall, gb_cm = evaluate_model(gb_model, X_test, y_test)
    print("\nGradient Boosting Model Accuracy:", gb_accuracy)
    print("Gradient Boosting Model Precision:", gb_precision)
    print("Gradient Boosting Model Recall:", gb_recall)
    print("Gradient Boosting Confusion Matrix:\n", gb_cm)