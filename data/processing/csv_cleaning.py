import os
from pathlib import Path
import pandas as pd

# Define the input and output directories
input_dir = '/home/diego/Documents/Python Projects/TCC/data/_raw_data/daily/'
output_dir = '/home/diego/Documents/Python Projects/TCC/data/cleaned/'

# Define the input and output directories for quarterly data
quarterly_input_dir = '/home/diego/Documents/Python Projects/TCC/data/_raw_data/quarterly/'
quarterly_output_dir = '/home/diego/Documents/Python Projects/TCC/data/cleaned/'

# Ensure the output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(quarterly_output_dir, exist_ok=True)

# Update the path for the dropped rows file
dropped_rows_file = '/home/diego/Documents/Python Projects/TCC/data/cleaned/dropped_rows_daily.csv'

# Ensure the updated output directory for dropped rows exists
os.makedirs(os.path.dirname(dropped_rows_file), exist_ok=True)

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

        # Identify and save rows with all NaN values (weekends)
        dropped_rows = df[df.isnull().all(axis=1)].index
        dropped_rows.to_series().to_csv(dropped_rows_file, index=False, header=['DATES'])

        # Remove rows where all stock data is NaN (weekends)
        df = df.dropna(how='all')

        # Fill missing values
        df = df.apply(lambda col: col.fillna(col.interpolate(method='linear', limit_area='inside')))
        df = df.bfill()

        # Normalize the data (min-max normalization)
        df = (df - df.min()) / (df.max() - df.min())

        # Save the cleaned and normalized DataFrame to the output directory
        df.to_csv(output_path, sep=';', decimal=',')
        print(f"Cleaned, forward/backward filled, and normalized: {file_name}")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")

def clean_and_save_quarterly_csv(file_name):
    try:
        # Construct full file paths
        input_path = os.path.join(quarterly_input_dir, file_name)
        output_path = os.path.join(quarterly_output_dir, file_name)

        # Read the CSV file with multi-header handling
        df = pd.read_csv(input_path, delimiter=';', decimal=',', na_values=['#N/D'], header=[0, 1])

        # Combine multi-level column headers into a single header
        df.columns = [' '.join(col).strip() for col in df.columns.values]

        # Extract only the ticker names from the column headers
        df.columns = [col.split()[0] for col in df.columns]

        # Set the DATES column as the index
        df = df.rename(columns={df.columns[0]: 'DATES'})
        df.set_index('DATES', inplace=True)

        # Load dropped rows from daily data
        dropped_rows = pd.read_csv(dropped_rows_file)['DATES']

        # Forward fill for missing values before dropping rows
        df = df.ffill()

        # Drop the same rows as in daily data
        df = df[~df.index.isin(dropped_rows)]

        # Backward fill for remaining missing values
        df = df.bfill()

        # Normalize the data (min-max normalization)
        df = (df - df.min()) / (df.max() - df.min())

        # Save the cleaned and normalized DataFrame to the output directory
        df.to_csv(output_path, sep=';', decimal=',')
        print(f"Cleaned, forward/backward filled, normalized, and dropped specific rows: {file_name}")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")

# =============================================================================
# Economatica index xlsx cleaning (IBX, Ibovespa)
# =============================================================================

# Resolve paths relative to this script so the function works on any machine
_THIS_DIR   = Path(__file__).resolve().parent          # data/processing/
_RAW_DAILY  = _THIS_DIR.parent / "_raw_data" / "daily"
_CLEANED    = _THIS_DIR.parent / "cleaned"


def clean_economatica_xlsx(filename: str, sheet_name: str, output_name: str) -> None:
    """Clean a single-series Economatica xlsx (IBX, Ibovespa layout).

    Expected layout
    ---------------
    Rows 1–3 : metadata / blank  →  skipped via header=3
    Row 4    : column headers    →  renamed to 'date' and 'price'
    Data rows: date  |  closing price (string; '-' on non-trading days)

    Output
    ------
    CSV saved to data/cleaned/<output_name> with columns: date, return
    (daily pct_change of price; first row dropped because return is NaN).
    """
    input_path  = _RAW_DAILY / filename
    output_path = _CLEANED   / output_name

    if not input_path.exists():
        print(f"Skipping {filename}: file not found at {input_path}")
        return

    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name, header=3)

        # Keep only the first two columns and rename them
        df = df.iloc[:, :2].copy()
        df.columns = ["date", "price"]

        # Drop non-trading day rows (price stored as the string '-')
        df = df[df["price"] != "-"].copy()

        # Cast and sort
        df["price"] = df["price"].astype(float)
        df["date"]  = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Daily returns
        df["return"] = df["price"].pct_change()
        df = df.dropna(subset=["return"]).reset_index(drop=True)
        df = df[["date", "return"]]

        _CLEANED.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, sep=";", decimal=",", index=False)
        print(f"Cleaned {filename} → {output_path}  ({len(df)} trading days)")

    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")


# Process all files in the daily and quarterly folders
if __name__ == "__main__":
    # Process daily files
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            clean_and_save_csv(file)

    # Process quarterly files
    for file in os.listdir(quarterly_input_dir):
        if file.endswith('.csv'):
            clean_and_save_quarterly_csv(file)

    # Process Economatica index xlsx files
    clean_economatica_xlsx("ibx.xlsx",       sheet_name="IBXX", output_name="ibx.csv")
    clean_economatica_xlsx("ibovespa.xlsx",   sheet_name="IBOV", output_name="ibovespa.csv")