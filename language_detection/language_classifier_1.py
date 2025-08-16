import pandas as pd

def load_data(file_path):
    """
    Loads data from a CSV file.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if file not found.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def get_data_head(df, n=10):
    """
    Gets the first n rows of the DataFrame.
    Args:
        df (pandas.DataFrame): The input DataFrame.
        n (int): The number of rows.
    Returns:
        pandas.DataFrame: The first n rows.
    """
    return df.head(n)

def check_missing_values(df):
    """
    Checks for missing (null) values in the DataFrame.
    Args:
        df (pandas.DataFrame): The input DataFrame.
    Returns:
        pandas.Series: A Series with the count of missing values for each column.
    """
    return df.isnull().sum()

def check_for_empty_strings(df):
    """
    Checks for rows where the 'Text' column is empty or just whitespace.
    Args:
        df (pandas.DataFrame): The input DataFrame.
    Returns:
        int: The count of rows with empty text.
    """
    df['Text'] = df['Text'].astype(str)
    empty_text_rows = df[df['Text'].str.strip() == '']
    return len(empty_text_rows)

def get_language_distribution(df):
    """
    Calculates the distribution of languages in the dataset.
    Args:
        df (pandas.DataFrame): The input DataFrame.
    Returns:
        pandas.Series: A Series with languages as index and their counts as values.
    """
    return df["Language"].value_counts()

# --- Main execution block ---
if __name__ == "__main__":
    DATA_FILE_PATH = 'Language Detection.csv'
    
    # 1. Load the data
    data = load_data(DATA_FILE_PATH)
    
    if data is not None:
        # 2. Get and display the head
        print("--- First 10 Rows of the Dataset ---")
        head_df = get_data_head(data, n=10)
        print(head_df)
        print("\n" + "="*50 + "\n")

        # 3. Check for missing values
        print("--- Missing Values Check ---")
        missing_values = check_missing_values(data)
        print(missing_values)
        if missing_values.sum() == 0:
            print("--> Verdict: No missing (null) values found.")
        print("\n" + "="*50 + "\n")

        # 4. Check for empty strings in the 'Text' column
        print("--- Empty String Check in 'Text' Column ---")
        empty_count = check_for_empty_strings(data)
        print(f"Number of rows with empty or whitespace-only text: {empty_count}")
        if empty_count == 0:
            print("--> Verdict: No empty text entries found.")
        print("\n" + "="*50 + "\n")

        # 5. Get and display language distribution
        print("--- Language Distribution (Top 10) ---")
        lang_dist = get_language_distribution(data)
        print(lang_dist.head(10)) # Print top 10 for brevity
        print(f"\nTotal unique languages: {len(lang_dist)}")
        print("\n" + "="*50 + "\n")
