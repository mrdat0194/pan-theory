import os
import opendatasets as od

def main():
    print("--- Demo: opendatasets library ---")
    
    # URL for the Titanic dataset
    dataset_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    print(f"Downloading dataset from: {dataset_url}")
    # opendatasets download function
    od.download(dataset_url)
    
    # Check if the download succeeded and verify where it was saved
    # For a direct URL, opendatasets typically saves it in a folder with the filename (titanic.csv)
    # or directly as a file.
    expected_path_folder = os.path.join(".", "titanic.csv")
    expected_path_file = os.path.join(expected_path_folder, "titanic.csv")
    
    if os.path.exists(expected_path_file):
        print(f"Success! Dataset downloaded and saved to: {os.path.abspath(expected_path_file)}")
    elif os.path.exists(expected_path_folder) and os.path.isfile(expected_path_folder):
        print(f"Success! Dataset downloaded and saved directly to: {os.path.abspath(expected_path_folder)}")
    else:
        print("Dataset downloaded, checking directory structure...")
        for root, dirs, files in os.walk("."):
            if "titanic.csv" in files:
                print(f"Found titanic.csv at: {os.path.join(root, 'titanic.csv')}")
                break

if __name__ == "__main__":
    main()
