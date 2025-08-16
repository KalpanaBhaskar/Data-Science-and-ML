import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features for the analysis.
    
    Args:
        df (pd.DataFrame): The input DataFrame with student data.
        
    Returns:
        pd.DataFrame: The feature matrix X.
    """
    # Select recommended columns
    features = [
        'Age at enrollment',
        'Gender',
        'Debtor',
        'Tuition fees up to date',
        'Curricular units 1st sem (approved)'
    ]
    X = df[features]
    return X


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        X (pd.DataFrame): The input feature matrix.
    
    Returns:
        pd.DataFrame: The feature matrix with missing values handled.
    """
    X = X.copy()

    # Categorical columns
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown')

    # Numerical columns
    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    for col in numerical_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    return X


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features into numeric format using one-hot encoding.
    
    Args:
        X (pd.DataFrame): The input feature matrix.
        
    Returns:
        pd.DataFrame: The feature matrix with one-hot encoding applied.
    """
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    """
    Split the dataset into training and testing sets.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        test_size (float): Proportion of data to use for testing.
        
    Returns:
        tuple: The train-test split of X and y.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Train a DecisionTreeClassifier on the training data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        
    Returns:
        DecisionTreeClassifier: The trained decision tree model.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Evaluate the model using accuracy and confusion matrix.
    
    Args:
        model (DecisionTreeClassifier): The trained model.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing labels.
        
    Returns:
        tuple: Model accuracy and confusion matrix.
    """
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm


def apply_l2_regularization(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> LogisticRegression:
    """
    Apply L2 regularization (Ridge) using Logistic Regression.
    
    Returns:
        LogisticRegression: The trained L2 regularized model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg_model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    reg_model_l2.fit(X_train_scaled, y_train)

    return reg_model_l2


def apply_l1_regularization(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> LogisticRegression:
    """
    Apply L1 regularization (Lasso) using Logistic Regression.
    
    Returns:
        LogisticRegression: The trained L1 regularized model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg_model_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
    reg_model_l1.fit(X_train_scaled, y_train)

    return reg_model_l1


# --- Main Execution Block (for testing/demo purposes) ---
if __name__ == '__main__':
    file_path = 'dataset.csv'
    
    df = pd.read_csv(file_path)
    
    X = select_features(df)
    X = handle_missing_values(X)
    X = encode_categorical_features(X)
    y = df['Target']   # Make sure your dataset has a 'Target' column
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_decision_tree(X_train, y_train)
    accuracy, cm = evaluate_model(model, X_test, y_test)
    print("\nDecision Tree Model Accuracy: ", accuracy)
    print("\nConfusion Matrix:\n", cm)
    
    reg_model_l2 = apply_l2_regularization(X_train, y_train, X_test, y_test)
    accuracy_l2 = reg_model_l2.score(StandardScaler().fit_transform(X_test), y_test)
    print("\nL2 Regularized Model Accuracy (Ridge): ", accuracy_l2)
    
    reg_model_l1 = apply_l1_regularization(X_train, y_train, X_test, y_test)
    accuracy_l1 = reg_model_l1.score(StandardScaler().fit_transform(X_test), y_test)
    print("\nL1 Regularized Model Accuracy (Lasso): ", accuracy_l1)
