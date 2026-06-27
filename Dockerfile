FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000 to comply with Hugging Face Space security guidelines
RUN useradd -m -u 1000 user

# Switch to the new user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements.txt and install python packages
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r $HOME/app/requirements.txt

# Copy the rest of the application files
COPY --chown=user . $HOME/app

# Expose the Hugging Face Space default port (7860)
EXPOSE 7860

# Run the Streamlit application on port 7860
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
