import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def explore_data(df: pd.DataFrame):
    """
    Perform basic data exploration, printing key details about the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to explore.
    """
    # Print dataset shape
    print("Dataset Shape:", df.shape)

    # Print data types of each column
    print("\nData Types:\n", df.dtypes)

    # Print number of missing values
    print("\nMissing Values:\n", df.isnull().sum())

    # Print value counts for key categorical columns
    for col in ['Gender', 'Course', 'Tuition fees up to date', 'Debtor']:
        if col in df.columns:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts())


def create_approval_rate(df: pd.DataFrame):
    """
    Create the 'approval_rate' feature based on 'Curricular units 1st sem (approved)' 
    and 'Curricular units 1st sem (enrolled)' columns.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with the 'approval_rate' column.
    """
    df['approval_rate'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)']
    df['approval_rate'] = df['approval_rate'].fillna(0)  # handle division by zero
    return df


def create_performance_score(df: pd.DataFrame):
    """
    Create the 'performance_score' feature based on 'Curricular units 1st sem (approved)' 
    and 'Curricular units 1st sem (evaluations)' columns.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with the 'performance_score' column.
    """
    df['performance_score'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (evaluations)']
    df['performance_score'] = df['performance_score'].fillna(0)  # handle division by zero
    return df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features needed for the analysis.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Add engineered features
    df = create_approval_rate(df)
    df = create_performance_score(df)
    
    # Print the first few rows of the DataFrame to verify
    print("\nData with new columns: ")
    print(df[['Course', 'Gender', 'approval_rate', 'performance_score']].head())
    
    return df


# --- Main Execution Block ---
if __name__ == '__main__':
    # Path to the dataset
    file_path = 'dataset.csv'
    
    df = load_data(file_path)
    explore_data(df)
    df = create_engineered_features(df)
