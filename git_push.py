import subprocess
import os
import sys

# Set the project directory
project_dir = r"C:\Users\Timber's Pad\OneDrive\Desktop\Assignment\596 D\Llama_FineTune"

# Change to project directory
os.chdir(project_dir)
print(f"Changed directory to: {os.getcwd()}")

# Initialize git repository if not exists
if not os.path.exists(".git"):
    result = subprocess.run(["git", "init"], capture_output=True, text=True)
    print("Git init:", result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)
else:
    print("Git repository already initialized")

# Add remote if not exists
result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
if "origin" not in result.stdout:
    result = subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/Timber-Gu/Llama_Project.git"],
        capture_output=True, text=True
    )
    print("Remote added:", result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)
else:
    print("Remote origin already exists")

# Add all files
result = subprocess.run(["git", "add", "."], capture_output=True, text=True)
print("Files added:", result.stdout if result.stdout else "Success")
if result.returncode != 0:
    print("Error:", result.stderr)

# Commit changes
result = subprocess.run(
    ["git", "commit", "-m", "Add fine-tuned model checkpoint files"],
    capture_output=True, text=True
)
print("Commit:", result.stdout)
if result.returncode != 0:
    print("Note:", result.stderr)

# Push to GitHub
result = subprocess.run(["git", "push", "-u", "origin", "main"], capture_output=True, text=True)
print("Push output:", result.stdout)
if result.returncode != 0:
    # If main doesn't exist, try master
    print("Trying master branch...")
    result = subprocess.run(["git", "branch", "-M", "main"], capture_output=True, text=True)
    result = subprocess.run(["git", "push", "-u", "origin", "main"], capture_output=True, text=True)
    print("Push output:", result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)

print("\nDone! Check your GitHub repository.")

