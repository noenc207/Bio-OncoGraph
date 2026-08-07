$ErrorActionPreference = "Stop"

Write-Host "Installing Bio-OncoGraph dependencies..."
pip install -r requirements.txt
Write-Host "Installation complete."
