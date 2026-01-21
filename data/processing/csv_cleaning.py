import os
import pandas as pd

# Define the input and output directories
input_dir = 'm:/Python Projects/TCC (new)/data/_raw_data/daily/'
output_dir = 'm:/Python Projects/TCC (new)/data/_raw_data/cleaned/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def clean_and_save_csv(file_name):
    try:
        # Construct full file paths
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # Read the CSV file with multi-header handling
        df = pd.read_csv(input_path, delimiter=';', decimal=',', na_values=['#N/D'], header=[0, 1])

        # Combine multi-level column headers into a single header
        df.columns = [' '.join(col).strip() for col in df.columns.values]

        # Extract only the ticker names from the column headers
        df.columns = [col.split()[0] for col in df.columns]

        # Set the DATES column as the index
        df = df.rename(columns={df.columns[0]: 'DATES'})
        df.set_index('DATES', inplace=True)

        # Remove rows where all stock data is NaN (weekends)
        df = df.dropna(how='all')

        # Perform cross-sectional mean imputation for missing values between valid numbers
        df = df.apply(lambda col: col.fillna(col.interpolate(method='linear', limit_area='inside')))

        # Backward fill for starting missing values
        df = df.bfill()

        # Normalize the data (min-max normalization)
        df = (df - df.min()) / (df.max() - df.min())

        # Save the cleaned and normalized DataFrame to the output directory
        df.to_csv(output_path, sep=';', decimal=',')
        print(f"Cleaned, normalized, and saved: {file_name}")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")

# Process all files in the daily folder
if __name__ == "__main__":
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            clean_and_save_csv(file)