# Use a Miniconda base image
FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /app

# Copy environment.yml first to leverage Docker build caching
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the conda environment
# Make sure it is activated for all RUN, CMD, etc.
ENV PATH /opt/conda/envs/fuel-forecaster/bin:$PATH

# Copy the rest of your code into the container
COPY . .

# (Optional) Make your main script executable
RUN chmod +x src/main.py

# Run your main script using the environment
CMD ["python", "src/main.py", "--process-data"]