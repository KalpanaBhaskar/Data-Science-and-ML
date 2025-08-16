import pandas as pd
import re

def preprocess_text(text):
    """
    Cleans a single text string by lowercasing, removing numbers, 
    punctuation, and extra whitespace.
    
    Args:
        text (str): The input string to clean.
        
    Returns:
        str: The cleaned string.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 3. Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --- Main execution block ---
if __name__ == "__main__":
    # Define file paths
    INPUT_FILE = 'Language Detection.csv'
    OUTPUT_FILE = 'Language_Detection_Cleaned.csv'
    
    print("Starting data cleaning process...")
    
    # 1. Load the data
    try:
        data = pd.read_csv(INPUT_FILE)
        print(f"Successfully loaded {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please place it in the correct directory.")
        exit() # Exit the script if the file isn't there

    # 2. Handle missing values: Drop rows with any null values
    data.dropna(inplace=True)

    # 3. Apply the cleaning function to the 'Text' column
    print("Cleaning text data...")
    data['Text'] = data['Text'].astype(str).apply(preprocess_text)

    # 4. Save the cleaned data
    data.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Data cleaning complete. Cleaned data saved to {OUTPUT_FILE}")
