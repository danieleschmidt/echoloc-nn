# EchoLoc-NN Quantum Planning Deployment Guide

## Overview

This guide covers deployment strategies for the EchoLoc-NN Quantum Planning system across different environments, from edge devices to cloud infrastructure.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Deployment Architectures](#deployment-architectures)
5. [Container Deployment](#container-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Edge Deployment](#edge-deployment)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Security Hardening](#security-hardening)
10. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores, 2.0 GHz (ARM64 or x86_64)
- **Memory**: 2 GB RAM
- **Storage**: 1 GB available space
- **OS**: Linux (Ubuntu 20+, RHEL 8+), macOS 11+, Windows 10+
- **Python**: 3.8+
- **Network**: 100 Mbps for distributed deployments

### Recommended Requirements

- **CPU**: 4+ cores, 3.0 GHz with SIMD support
- **Memory**: 8 GB RAM
- **Storage**: 10 GB SSD with fast I/O
- **GPU**: CUDA-compatible for acceleration (optional)
- **Network**: 1 Gbps with low latency

### Edge Device Requirements

- **Raspberry Pi 4B**: 4GB+ RAM model
- **NVIDIA Jetson Nano/Xavier**: For GPU acceleration
- **Intel NUC**: For x86 edge deployment
- **Arduino/ESP32**: For sensor integration

## Installation Methods

### Python Package Installation

```bash
# Install from PyPI (when published)
pip install echoloc-nn

# Or install from source
git clone https://github.com/your-org/echoloc-nn.git
cd echoloc-nn
pip install -e .

# Install with quantum planning extras
pip install echoloc-nn[quantum,optimization]
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/your-org/echoloc-nn.git
cd echoloc-nn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Verify installation
python -c "import echoloc_nn; print('Installation successful')"
```

### System Package Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install echoloc-nn

# RHEL/CentOS
sudo yum install echoloc-nn

# Arch Linux
sudo pacman -S echoloc-nn
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Core configuration
ECHOLOC_CONFIG_FILE=/etc/echoloc/config.json
ECHOLOC_LOG_LEVEL=INFO
ECHOLOC_ENVIRONMENT=production

# Database configuration
DATABASE_URL=postgresql://user:password@localhost:5432/echoloc
REDIS_URL=redis://localhost:6379/0

# Security configuration
SECRET_KEY=your-secure-secret-key-here
JWT_SECRET=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here

# Quantum planning configuration
QUANTUM_MAX_ITERATIONS=1000
QUANTUM_INITIAL_TEMP=100.0
QUANTUM_FINAL_TEMP=0.01
QUANTUM_STRATEGY=quantum_annealing

# Resource configuration
MAX_WORKERS=8
RESOURCE_POOL_SIZE=16
AUTO_SCALING_ENABLED=true

# Hardware configuration
SENSOR_PORT=/dev/ttyUSB0
ARDUINO_PORT=/dev/ttyACM0
GPIO_PINS=18,19,20,21

# Performance configuration
ENABLE_GPU_ACCELERATION=true
ENABLE_CACHING=true
CACHE_SIZE=1000
```

### Configuration File

Create `/etc/echoloc/config.json`:

```json
{
  "system": {
    "environment": "production",
    "debug": false,
    "log_level": "INFO"
  },
  "quantum_planning": {
    "default_strategy": "quantum_annealing",
    "max_iterations": 1000,
    "initial_temperature": 100.0,
    "final_temperature": 0.01,
    "parallel_threads": 4,
    "enable_gpu_acceleration": true
  },
  "resource_management": {
    "auto_scaling": {
      "enabled": true,
      "min_workers": 1,
      "max_workers": 8,
      "cpu_threshold_up": 70.0,
      "cpu_threshold_down": 30.0
    },
    "resource_pool": {
      "initial_size": 4,
      "max_size": 16,
      "load_balancing": "round_robin"
    }
  },
  "security": {
    "enable_authentication": true,
    "session_timeout": 1800,
    "max_login_attempts": 5,
    "require_https": true,
    "enable_cors": false
  },
  "hardware": {
    "ultrasonic_array": {
      "sensors": 4,
      "frequency": 40000,
      "sample_rate": 250000,
      "chirp_duration": 0.005
    },
    "gpio": {
      "trigger_pins": [18, 19, 20, 21],
      "echo_pins": [22, 23, 24, 25]
    }
  },
  "monitoring": {
    "enable_metrics": true,
    "metrics_port": 9090,
    "health_check_interval": 30,
    "log_file": "/var/log/echoloc/application.log"
  }
}
```

## Deployment Architectures

### Single Node Deployment

```
┌─────────────────────────────────────────┐
│           Single Node                   │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐   │
│  │   Web API   │  │  Quantum        │   │
│  │   Server    │  │  Planning       │   │
│  │             │  │  Engine         │   │
│  └─────────────┘  └─────────────────┘   │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ Hardware    │  │  Database       │   │
│  │ Interface   │  │  (SQLite)       │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────┘
```

**Pros**: Simple setup, low latency, minimal dependencies
**Cons**: Single point of failure, limited scalability
**Best for**: Development, small deployments, edge devices

### Distributed Deployment

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Web Tier  │    │ Application │    │  Data Tier  │
│             │    │    Tier     │    │             │
│ Load        │    │ Quantum     │    │ PostgreSQL  │
│ Balancer    │───▶│ Planning    │───▶│ Redis       │
│ (HAProxy)   │    │ Workers     │    │ InfluxDB    │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Pros**: High availability, horizontal scaling, fault tolerance
**Cons**: Complex setup, network latency, data consistency
**Best for**: Production environments, high-load systems

### Microservices Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Gateway   │  │  Planning   │  │  Resource   │
│   Service   │──│   Service   │──│  Service    │
└─────────────┘  └─────────────┘  └─────────────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Auth        │  │ Validation  │  │ Monitoring  │
│ Service     │  │ Service     │  │ Service     │
└─────────────┘  └─────────────┘  └─────────────┘
```

**Components**:
- **Gateway Service**: API routing and authentication
- **Planning Service**: Core quantum optimization
- **Resource Service**: Resource pool management
- **Validation Service**: Input validation and security
- **Monitoring Service**: Metrics and health checking

## Container Deployment

### Docker

#### Single Container

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package
RUN pip install -e .

# Create non-root user
RUN useradd -r -s /bin/false echoloc
USER echoloc

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "echoloc_nn.api.server"]
```

#### Build and Run

```bash
# Build image
docker build -t echoloc-nn:latest .

# Run container
docker run -d \
  --name echoloc-nn \
  -p 8000:8000 \
  -v /etc/echoloc:/etc/echoloc:ro \
  -v /var/log/echoloc:/var/log/echoloc \
  --env-file .env \
  echoloc-nn:latest

# Check logs
docker logs echoloc-nn

# Check health
docker exec echoloc-nn curl http://localhost:8000/health
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  echoloc-nn:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://echoloc:password@postgres:5432/echoloc
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./config:/etc/echoloc:ro
      - ./logs:/var/log/echoloc
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=echoloc
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=echoloc
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs:ro
    depends_on:
      - echoloc-nn
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: echoloc-nn
  labels:
    app: echoloc-nn
spec:
  replicas: 3
  selector:
    matchLabels:
      app: echoloc-nn
  template:
    metadata:
      labels:
        app: echoloc-nn
    spec:
      containers:
      - name: echoloc-nn
        image: echoloc-nn:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: echoloc-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: echoloc-secrets
              key: secret-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /etc/echoloc
          readOnly: true
        - name: logs
          mountPath: /var/log/echoloc
      volumes:
      - name: config
        configMap:
          name: echoloc-config
      - name: logs
        persistentVolumeClaim:
          claimName: echoloc-logs-pvc
```

#### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: echoloc-nn-service
  labels:
    app: echoloc-nn
spec:
  selector:
    app: echoloc-nn
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: echoloc-nn-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.echoloc.example.com
    secretName: echoloc-tls
  rules:
  - host: api.echoloc.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: echoloc-nn-service
            port:
              number: 80
```

## Cloud Deployment

### AWS

#### Elastic Container Service (ECS)

```json
{
  "family": "echoloc-nn",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "echoloc-nn",
      "image": "your-account.dkr.ecr.region.amazonaws.com/echoloc-nn:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:echoloc/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/echoloc-nn",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### Application Load Balancer

```bash
# Create target group
aws elbv2 create-target-group \
  --name echoloc-nn-targets \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-12345678 \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3

# Create application load balancer
aws elbv2 create-load-balancer \
  --name echoloc-nn-alb \
  --subnets subnet-12345678 subnet-87654321 \
  --security-groups sg-12345678
```

### Google Cloud Platform

#### Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: echoloc-nn
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/memory: "2Gi"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/echoloc-nn:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: echoloc-secrets
              key: database-url
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
```

Deploy:

```bash
# Build and push image
gcloud builds submit --tag gcr.io/project-id/echoloc-nn

# Deploy to Cloud Run
gcloud run deploy echoloc-nn \
  --image gcr.io/project-id/echoloc-nn \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances 10 \
  --memory 2Gi \
  --cpu 2
```

### Azure

#### Container Instances

```json
{
  "location": "East US",
  "properties": {
    "containers": [
      {
        "name": "echoloc-nn",
        "properties": {
          "image": "your-registry.azurecr.io/echoloc-nn:latest",
          "resources": {
            "requests": {
              "cpu": 1,
              "memoryInGB": 2
            }
          },
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ],
          "environmentVariables": [
            {
              "name": "ENVIRONMENT",
              "value": "production"
            }
          ]
        }
      }
    ],
    "osType": "Linux",
    "ipAddress": {
      "type": "Public",
      "ports": [
        {
          "protocol": "TCP",
          "port": 8000
        }
      ]
    },
    "restartPolicy": "Always"
  }
}
```

## Edge Deployment

### Raspberry Pi

#### Setup Script

```bash
#!/bin/bash
# setup-raspberry-pi.sh

set -e

echo "Setting up EchoLoc-NN on Raspberry Pi..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
  python3-pip \
  python3-venv \
  git \
  curl \
  build-essential \
  libffi-dev \
  libssl-dev

# Install GPIO libraries
sudo apt install -y \
  python3-rpi.gpio \
  wiringpi

# Create application user
sudo useradd -r -s /bin/false -m echoloc

# Create application directory
sudo mkdir -p /opt/echoloc-nn
sudo chown echoloc:echoloc /opt/echoloc-nn

# Switch to application user
sudo -u echoloc bash << 'EOF'
cd /opt/echoloc-nn

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install EchoLoc-NN
pip install echoloc-nn[raspberry-pi]

# Create configuration
mkdir -p /opt/echoloc-nn/config
cat > /opt/echoloc-nn/config/config.json << 'CONFIG'
{
  "system": {
    "environment": "edge",
    "debug": false,
    "log_level": "INFO"
  },
  "hardware": {
    "platform": "raspberry_pi",
    "gpio_pins": {
      "trigger": [18, 19, 20, 21],
      "echo": [22, 23, 24, 25]
    },
    "spi": {
      "enabled": true,
      "device": "/dev/spidev0.0"
    }
  },
  "quantum_planning": {
    "max_iterations": 500,
    "parallel_threads": 2
  }
}
CONFIG
EOF

# Create systemd service
sudo tee /etc/systemd/system/echoloc-nn.service > /dev/null << 'SERVICE'
[Unit]
Description=EchoLoc-NN Quantum Planning Service
After=network.target

[Service]
Type=simple
User=echoloc
Group=echoloc
WorkingDirectory=/opt/echoloc-nn
Environment=PATH=/opt/echoloc-nn/venv/bin
Environment=ECHOLOC_CONFIG_FILE=/opt/echoloc-nn/config/config.json
ExecStart=/opt/echoloc-nn/venv/bin/python -m echoloc_nn.api.server
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable echoloc-nn
sudo systemctl start echoloc-nn

echo "EchoLoc-NN setup complete!"
echo "Service status:"
sudo systemctl status echoloc-nn
```

### NVIDIA Jetson

```bash
#!/bin/bash
# setup-jetson.sh

# Install JetPack SDK
sudo apt update
sudo apt install -y nvidia-jetpack

# Install Python dependencies with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install EchoLoc-NN with GPU support
pip3 install echoloc-nn[gpu,jetson]

# Configure for GPU acceleration
cat > /etc/echoloc/jetson-config.json << 'EOF'
{
  "optimization": {
    "enable_gpu_acceleration": true,
    "cuda_device": 0,
    "tensorrt_optimization": true
  },
  "quantum_planning": {
    "parallel_threads": 6,
    "use_gpu_kernels": true
  }
}
EOF
```

## Monitoring and Maintenance

### Health Checks

```python
# health_check.py
from echoloc_nn.utils.monitoring import HealthChecker
from echoloc_nn.quantum_planning import QuantumTaskPlanner

def check_system_health():
    checker = HealthChecker()
    
    # Check core components
    results = checker.check_all_components()
    
    # Check quantum planning
    planner = QuantumTaskPlanner()
    planning_health = checker.check_quantum_planning(planner)
    
    # Generate health report
    overall_health = all(r.status == 'healthy' for r in results.values())
    
    return {
        'status': 'healthy' if overall_health else 'unhealthy',
        'components': results,
        'quantum_planning': planning_health,
        'timestamp': time.time()
    }
```

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
planning_requests = Counter('echoloc_planning_requests_total', 'Total planning requests')
planning_duration = Histogram('echoloc_planning_duration_seconds', 'Planning duration')
active_tasks = Gauge('echoloc_active_tasks', 'Number of active tasks')
resource_utilization = Gauge('echoloc_resource_utilization', 'Resource utilization', ['resource_type'])

# Start metrics server
start_http_server(9090)
```

### Log Aggregation

```yaml
# logstash.conf
input {
  file {
    path => "/var/log/echoloc/*.log"
    type => "echoloc"
    codec => "json"
  }
}

filter {
  if [type] == "echoloc" {
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    mutate {
      add_field => { "environment" => "${ENVIRONMENT:development}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "echoloc-%{+YYYY.MM.dd}"
  }
}
```

## Security Hardening

### Network Security

```bash
# firewall-setup.sh
#!/bin/bash

# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port as needed)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow application port
sudo ufw allow 8000/tcp

# Enable firewall
sudo ufw enable
```

### SSL/TLS Configuration

```nginx
# nginx-ssl.conf
server {
    listen 443 ssl http2;
    server_name api.echoloc.example.com;

    ssl_certificate /etc/ssl/certs/echoloc.crt;
    ssl_certificate_key /etc/ssl/private/echoloc.key;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python -m echoloc_nn.api.server

# Optimize configuration
export ECHOLOC_MAX_WORKERS=2
export ECHOLOC_CACHE_SIZE=100
```

#### Slow Planning Performance

```bash
# Enable performance profiling
export ECHOLOC_ENABLE_PROFILING=true

# Check CPU usage
top -p $(pgrep -f echoloc)

# Optimize quantum planning
cat > config/performance.json << 'EOF'
{
  "quantum_planning": {
    "max_iterations": 500,
    "parallel_threads": 4,
    "enable_gpu_acceleration": true,
    "use_fast_annealing": true
  }
}
EOF
```

#### Hardware Connection Issues

```bash
# Check USB/serial connections
lsusb
ls -la /dev/tty*

# Test GPIO (Raspberry Pi)
gpio readall

# Check permissions
sudo usermod -a -G dialout $USER
sudo usermod -a -G gpio $USER
```

### Debug Mode

```bash
# Enable debug logging
export ECHOLOC_LOG_LEVEL=DEBUG
export ECHOLOC_DEBUG=true

# Start with debug output
python -m echoloc_nn.api.server --debug

# Check component status
python -c "
from echoloc_nn.utils.monitoring import HealthChecker
checker = HealthChecker()
print(checker.check_all_components())
"
```

### Performance Testing

```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 http://localhost:8000/api/v1/health

# Stress testing
python scripts/stress_test.py --requests 10000 --concurrent 50

# Memory profiling
python -m memory_profiler scripts/memory_test.py
```

This deployment guide provides comprehensive coverage for deploying EchoLoc-NN Quantum Planning across various environments. Choose the deployment method that best fits your infrastructure and requirements.