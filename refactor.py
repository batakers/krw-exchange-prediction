import os
import shutil

print("=== Starting Project Restructuring ===")

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)
print("1. Folders 'data/' and 'notebooks/' checked/created.")

# Move CSV
csv_file = 'XAU BTC Silver SP500 dataset.csv'
if os.path.exists(csv_file):
    shutil.move(csv_file, os.path.join('data', csv_file))
    print(f"2. Moved {csv_file} -> data/")
else:
    print(f"2. {csv_file} not found in root, might already be moved.")

# Move Notebook
nb_file = 'Project_Code.ipynb'
if os.path.exists(nb_file):
    shutil.move(nb_file, os.path.join('notebooks', nb_file))
    print(f"3. Moved {nb_file} -> notebooks/")
else:
    print(f"3. {nb_file} not found in root, might already be moved.")

# Remove temp file
tmp_file = 'tmp_inspect_nb.py'
if os.path.exists(tmp_file):
    os.remove(tmp_file)
    print(f"4. Succeeded in removing temporary file: {tmp_file}")
else:
    print(f"4. {tmp_file} not found, already removed.")

print("=== Restructuring Complete! ===")
print("Please run these git commands next:")
print("  git add .")
print("  git commit -m \"Apply industrial repo restructurization\"")
print("  git push")
