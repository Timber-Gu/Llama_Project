@echo off
cd /d "C:\Users\Timber's Pad\OneDrive\Desktop\Assignment\596 D\Llama_FineTune"

echo Initializing git repository...
git init

echo Adding remote repository...
git remote add origin https://github.com/Timber-Gu/Llama_Project.git

echo Adding files...
git add .

echo Committing changes...
git commit -m "Add fine-tuned model checkpoint files"

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause

