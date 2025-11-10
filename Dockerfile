# Dockerfile

# Use a specific Python base image that is known to work well
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Gunicorn will run on
EXPOSE 10000

# Define the command to run the application using Gunicorn
# The port must be pulled from the environment variable provided by Render
CMD gunicorn --bind 0.0.0.0:$PORT app:app