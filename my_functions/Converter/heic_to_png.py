import os
from PIL import Image
import pillow_heif as pyheif

# Folder containing .heic files
input_folder = r'C:\Users\mrdat\Desktop\archive\Iphone'
output_folder = r'C:\Users\mrdat\Desktop\archive\Iphone\png'

# # CreS

def delete_all_heic_files(folder):
    """
    Delete all .heic files in the specified folder.
    """
    for filename in os.listdir(folder):
        if filename.lower().endswith(".heic"):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")

# Example usage:
delete_all_heic_files(input_folder)