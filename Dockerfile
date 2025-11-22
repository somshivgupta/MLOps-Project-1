# Use an official Python 3.10 image
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project (includes .project-root)
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose the port your app will run on
EXPOSE 5000

# Run the FastAPI app
CMD ["python3", "main.py"]