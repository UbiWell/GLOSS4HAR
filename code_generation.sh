# filename: code_generation.sh

# Ensure we're using venv (built-in with Python) to create a clean environment
python3 -m venv data_fetch_env

# Activate the new virtual environment
source data_fetch_env/bin/activate

# Install compatible packages for the desired functionality
pip install -qqq pydantic==2.7.4 openai==1.58.1 pillow==11.0.0

# Confirm the installation and dependency acceptance
pip show pydantic openai pillow

# (Optional) If other known dependencies are required for specific tasks, include them similarly
# pip install some_other_package==version

echo "Setup complete. You can now run your Python script in this isolated environment."
