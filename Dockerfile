# Image the coding agent uses to execute the Python code it generates.
# Build with:  docker build -t sensemaking-code .
# At run time the repository root is mounted at /workspace, so the generated code
# can import data_processing / data_streams and read the CSVs in data/.
FROM python:3.11-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
