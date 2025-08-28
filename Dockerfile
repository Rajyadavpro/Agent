# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt, if it exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Run langgraph2.py when the container launches
CMD ["python", "langgraph2.py"]
