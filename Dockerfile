# Multi-stage build for EchoLoc-NN development and production

# Development stage
FROM python:3.10-slim as development

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    make \
    curl \
    gcc \
    g++ \
    libasound2-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev]"

# Expose ports for development server
EXPOSE 8080 8888

# Development command
CMD ["python", "-m", "echoloc_nn.visualization.server", "--port", "8080"]

# Production stage
FROM python:3.10-slim as production

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libasound2 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash echoloc
USER echoloc
WORKDIR /home/echoloc

# Copy and install package
COPY --chown=echoloc:echoloc pyproject.toml README.md LICENSE ./
COPY --chown=echoloc:echoloc echoloc_nn/ ./echoloc_nn/

# Install production dependencies only
RUN pip install --user --no-cache-dir -e .

# Add user local bin to PATH
ENV PATH="/home/echoloc/.local/bin:$PATH"

# Production command
CMD ["python", "-m", "echoloc_nn.inference.server"]

# Hardware testing stage (for CI/CD with hardware)
FROM development as hardware-test

# Install additional hardware testing dependencies
RUN apt-get update && apt-get install -y \
    arduino-cli \
    minicom \
    udev \
    && rm -rf /var/lib/apt/lists/*

# Add dialout group for serial communication
RUN usermod -a -G dialout root

# Copy hardware test scripts
COPY tests/hardware/ ./tests/hardware/
COPY firmware/ ./firmware/

CMD ["python", "-m", "pytest", "tests/", "-m", "hardware", "-v"]