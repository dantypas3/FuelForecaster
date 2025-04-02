FROM continuumio/miniconda3

WORKDIR /app

# Copy environment.yml into container
COPY environment.yml .

# Create conda environment
RUN conda config --set channel_priority strict && \
    conda env create -f environment.yml

# Set env path to make conda environment default
ENV PATH=/opt/conda/envs/fuel-forecaster/bin:$PATH
ENV PYTHONPATH=/app

# Copy the rest of your code
COPY . .

RUN chmod +x src/main.py

RUN mkdir -p /app/models


CMD ["python", "src/main.py", "--process-data"]
