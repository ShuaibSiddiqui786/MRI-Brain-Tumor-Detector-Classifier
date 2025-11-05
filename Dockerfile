# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files to the container
COPY . /app

# Upgrade pip and install required build tools
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app with the correct host and port
CMD ["python", "app.py"]
