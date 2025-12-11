# Navigate to the project directory
$projectPath = "C:\Users\Timber's Pad\OneDrive\Desktop\Assignment\596 D\Llama_FineTune"
Set-Location $projectPath

# Initialize git repository if not exists
if (-not (Test-Path ".git")) {
    git init
    Write-Host "Git repository initialized"
}

# Add remote if not exists
$remoteUrl = "https://github.com/Timber-Gu/Llama_Project.git"
$remotes = git remote -v
if ($remotes -notmatch "origin") {
    git remote add origin $remoteUrl
    Write-Host "Remote origin added"
}

# Add all files
git add .
Write-Host "Files added to staging"

# Commit changes
git commit -m "Add fine-tuned model checkpoint files"
Write-Host "Changes committed"

# Push to GitHub (main branch)
git push -u origin main
Write-Host "Pushed to GitHub successfully"

