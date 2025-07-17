import numpy as np
import pandas as pd
# import mne
import zipfile
import os


zipdatapath = r"D:\Inje University Work\Publications Inje Univ\EEG Workload and Not Workload\EEG Data.zip"
def unzip_folder(zip_file_path, extract_to_folder):

    try:
        # Check if the zip file exists
        if not os.path.exists(zip_file_path):
            print(f"The file '{zip_file_path}' does not exist.")
            return

        # Create the target folder if it doesn't exist
        if not os.path.exists(extract_to_folder):
            os.makedirs(extract_to_folder)

        # Open the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all the contents
            zip_ref.extractall(extract_to_folder)
            print(f"Extracted contents to '{extract_to_folder}' successfully.")

    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' is not a valid zip file.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
zip_file = "path/to/your/file.zip"
extract_to = "path/to/extract/folder"

unzip_folder(zip_file, extract_to)