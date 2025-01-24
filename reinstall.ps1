# Clean everything
pip uninstall market_analyzer -y
Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "dist" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "src\*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "src\market_analyzer\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "tests\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".pytest_cache" -Recurse -Force -ErrorAction SilentlyContinue

# Reinstall the package
pip install -e .

# Set PYTHONPATH (temporary for this session)
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"

# Run tests with verbose output and debug information
python -m pytest tests/test_analyzer.py -v --tb=short