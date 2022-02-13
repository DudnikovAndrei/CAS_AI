Get-childItem venv -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
python.exe -m venv venv
venv\scripts\activate
python -m pip install --upgrade pip setuptools