from os.path import join
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import requests
import zipfile
import shutil
import random
import os


class Prc:
    
    def download_dataset(self, url:str, destination:str):
    
        # Send a GET request to the URL
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check if the request was successful
        except requests.RequestException as e:
            print(f"Error downloading the file: {e}")
            return None

        # Get the total file size from the response headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Open a file to write the data
        with open(destination, 'wb') as file:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    pbar.update(len(data))

        return destination


    def extract_zip(self, zip_file_path:str, extraction_path:str):
        # Create the extraction directory if it doesn't exist
        os.makedirs(extraction_path, exist_ok=True)
        
        # Open the specified zip file in read mode
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all contents of the zip file into the specified extraction path
            zip_ref.extractall(extraction_path)
        
        
    def is_image_file(self, files_path:list[str]):
        
        images = []  # List to hold valid image file paths
        invalid_files = []  # List to hold invalid image file paths
        
        # Iterate through each file path in the provided list
        for file in files_path:  
            try:
                # Attempt to open the file as an image
                with Image.open(file) as img:  
                    # Verify that the opened image is valid
                    img.verify()  
                    
                # If no exception is raised, add to valid images
                images.append(file)  
                
            # Catch any exception that occurs during the image opening or verification
            except Exception as e:  
                # Print the error message for the invalid file
                print(f"File {file} is not a valid image. Error: {e}")  
                
                # Add the invalid file path to the list of invalid files
                invalid_files.append(file)  
                
                
        # Return the lists of valid and invalid files
        return images, invalid_files  

        
    def image_verification(self, base_directory_path:str):
        # Create a list of specific file paths to remove 
        # These include known unwanted files for the 'Cat' and 'Dog' folders
        rm = [
            join(base_directory_path, 'Cat', 'Thumbs.db'), 
            join(base_directory_path, 'Cat', '666.jpg'),    
            join(base_directory_path, 'Dog', '11702.jpg'),   
            join(base_directory_path, 'Dog', 'Thumbs.db'),   
        ]
        
        # Convert the directory path string into a Path object for easier manipulation
        base_directory_path = Path(base_directory_path)

        # Recursively gather all file paths from the directory, storing them in a list
        all_files = list(base_directory_path.rglob('*'))

        # Use the is_image_file method to identify valid image files
        _, invalid_files = self.is_image_file(all_files)
        
        # Extend the removal list with the invalid files found
        rm.extend(invalid_files)

        # Loop through the unique paths in the removal list
        for path in set(rm):
            try:
                # Attempt to remove each file
                os.remove(path)
            except FileNotFoundError:
                # If the file is not found, continue to the next file without raising an error
                pass

        
    def split_dataset(self, data_dir:str, output_dir:str):

        train_dir = join(output_dir, 'train')
        test_dir = join(output_dir, 'test')

        # Create directories for training and testing
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Define the desired split ratio
        split_ratio = 0.8  # 80% train, 20% test

        # Loop through each class directory (Cat/Dog)
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)

            # Check if the path is a directory
            if os.path.isdir(class_path):
                # Create class directories for train and test
                os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

                # Get all image files in the class directory
                images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

                # Shuffle the images
                random.shuffle(images)

                # Determine the split index
                split_index = int(len(images) * split_ratio)

                # Split the images
                train_images = images[:split_index]
                test_images = images[split_index:]

                # Move images to respective directories
                for img in train_images:
                    shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

                for img in test_images:
                    shutil.move(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

        print("Dataset has been split into training and testing sets.")