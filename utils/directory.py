#code to make root directories
import os 

def make_directories(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
