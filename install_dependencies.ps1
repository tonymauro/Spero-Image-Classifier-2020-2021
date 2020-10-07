# Works in Powershell on Windows computers
# Installs the dependencies from the requirements.txt file into the python virtual environment

foreach($line in Get-Content .\requirements.txt)
{
    pip install $line
}

Write-Host "All dependencies installed."