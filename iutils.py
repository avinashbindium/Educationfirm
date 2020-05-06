import subprocess as s
import os

# Run textcleaner

def run_textcleaner(filename, img_id):

    cleaned_file = "cleaned"  + str(img_id) + ".jpg"

    s.call(["./textcleaner", "-g", "-e", "stretch", "-f", str(25), "-o", str(10), "-s", str(1), str(filename), str(cleaned_file)]) #good

    return cleaned_file

