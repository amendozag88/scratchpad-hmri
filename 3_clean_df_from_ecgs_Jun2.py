import os
import re
import argparse
from dateutil import parser
import pandas as pd
import numpy as np

def extract_age_from_birthday(birthday):
    if pd.isna(birthday):
        return None, birthday
    # match patterns like "1970-01-01 (76 yr)" or "76 yr"
    match = re.search(r'\(?\s*(\d{1,3})\s*yr\)?\s*$', str(birthday))
    if match:
        age = int(match.group(1))
        cleaned_birthday = re.sub(r'\s*\(?\s*\d{1,3}\s*yr\)?\s*$', '', str(birthday)).strip()
        if cleaned_birthday == '':
            cleaned_birthday = None
        return age, cleaned_birthday
    return None, birthday

def parse_birthday(birthday):
    if pd.isna(birthday):
        return None
    try:
        parsed_date = parser.parse(str(birthday), dayfirst=False, fuzzy=True)
        return parsed_date.strftime('%Y-%m-%d')
    except (ValueError, OverflowError):
        return None

def parse_datetime(datetime_str):
    if pd.isna(datetime_str):
        return None
    try:
        parsed_date = parser.parse(str(datetime_str), dayfirst=False, fuzzy=True)
        return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OverflowError):
        return None

def clean_dataframe_from_ecgs(metadata_df, filename=None):
    # Ensure columns exist
    if 'age' not in metadata_df.columns:
        metadata_df['age'] = pd.NA
    if 'birthday' not in metadata_df.columns:
        metadata_df['birthday'] = pd.NA
    if 'datetime' not in metadata_df.columns:
        metadata_df['datetime'] = pd.NA

    # Preserve source file name column so downstream code can use it
    if 'source_file' not in metadata_df.columns:
        metadata_df['source_file'] = filename if filename is not None else pd.NA

    # Extract age from birthday text when present
    metadata_df[['extracted_age', 'cleaned_birthday']] = metadata_df['birthday'].apply(
        lambda x: pd.Series(extract_age_from_birthday(x))
    )

    # Fill age column when missing with extracted_age
    metadata_df['age'] = metadata_df.apply(
        lambda row: row['extracted_age'] if (pd.isna(row['age']) or row['age'] in ['', None]) and row['extracted_age'] is not None else row['age'],
        axis=1
    )

    # Replace birthday with cleaned_birthday then parse to ISO date
    metadata_df['birthday'] = metadata_df['cleaned_birthday'].apply(parse_birthday)

    # Parse datetime column to uniform format
    metadata_df['datetime'] = metadata_df['datetime'].apply(parse_datetime)

    # Drop helper columns
    metadata_df = metadata_df.drop(columns=['extracted_age', 'cleaned_birthday'])

    # Normalize age column: remove trailing 'Y' and coerce to numeric, round and convert to nullable Int64
    metadata_df['age'] = metadata_df['age'].astype(str).str.replace(r'Y$', '', regex=True).str.strip()
    metadata_df['age'] = pd.to_numeric(metadata_df['age'], errors='coerce')
    # Round fractional ages to nearest integer
    metadata_df['age'] = metadata_df['age'].round().astype('Int64')

    return metadata_df

def process_all_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    csv_files.sort()
    if not csv_files:
        print("No CSV files found in input directory:", input_dir)
        return

    cleaned_filenames = []
    for f in csv_files:
        print(f'Processing file: {f}')
        try:
            df = pd.read_csv(os.path.join(input_dir, f), low_memory=False)
        except Exception as e:
            print(f"  Failed to read {f}: {e}")
            continue

        cleaned_df = clean_dataframe_from_ecgs(df, filename=f)
        # Determine cleaned filename: keep prefix before 'combined' if present, else append _cleaned
        if 'combined' in f:
            cleaned_name = f.split('combined')[0] + 'cleaned.csv'
        else:
            base, _ = os.path.splitext(f)
            cleaned_name = base + '_cleaned.csv'

        cleaned_path = os.path.join(output_dir, cleaned_name)
        cleaned_df.to_csv(cleaned_path, index=False)
        cleaned_filenames.append(cleaned_name)
        print(f'  Saved cleaned file: {cleaned_name} shape: {cleaned_df.shape}')

    # Concatenate all cleaned files into combined_cleaned.csv
    combined_list = []
    for cf in cleaned_filenames:
        path_cf = os.path.join(output_dir, cf)
        try:
            tmp = pd.read_csv(path_cf, low_memory=False)
            combined_list.append(tmp)
        except Exception as e:
            print(f"  Failed to read cleaned file {cf}: {e}")

    if combined_list:
        combined_df = pd.concat(combined_list, ignore_index=True)
        combined_out_path = os.path.join(output_dir, 'combined_cleaned.csv')
        combined_df.to_csv(combined_out_path, index=False)
        print(f'Combined cleaned dataframe saved to {combined_out_path}, shape: {combined_df.shape}')

        # Create age_labels.csv with age and source_file columns
        if 'source_file' not in combined_df.columns:
            combined_df['source_file'] = pd.NA
        age_labels = combined_df[['age', 'source_file']].copy()
        age_labels_path = os.path.join(output_dir, 'age_labels.csv')
        age_labels.to_csv(age_labels_path, index=False)
        print(f'Age labels saved to {age_labels_path}. Missing ages: {int(age_labels["age"].isna().sum())}')
    else:
        print("No cleaned files to combine.")

def main():
    parser = argparse.ArgumentParser(description="Clean ECG metadata CSVs and produce combined cleaned CSVs and age labels.")
    parser.add_argument("--input_dir", type=str,
                        default="/home/hmaiaxg21/alkindilab/hmaiaxg21/ecg_pdf_df_output",
                        help="Folder containing raw CSVs to clean")
    parser.add_argument("--output_dir", type=str,
                        default="/home/hmaiaxg21/alkindilab/hmaiaxg21/ecg_pdf_df_output",
                        help="Folder where cleaned CSVs and combined outputs will be written")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    process_all_files(input_dir, output_dir)

if __name__ == "__main__":
    main()