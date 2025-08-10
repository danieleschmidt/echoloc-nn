# EchoLoc-NN Deployment Guide

## 🚀 Production Deployment Guide

This guide covers deploying EchoLoc-NN from development to production environments with comprehensive configuration, security, and optimization considerations.

## 📋 System Status

### ✅ Implemented Components
- **Core Functionality**: Neural models, signal processing, inference engine
- **Robustness**: Error handling, fault tolerance, monitoring infrastructure  
- **Optimization**: Performance framework, caching, concurrent processing
- **Containerization**: Multi-stage Docker build with development/production targets
- **Infrastructure**: Redis caching, InfluxDB metrics, Grafana monitoring
- **Health Checks**: HTTP endpoint monitoring and container health validation

### ⚠️ Production Requirements
- **Security Hardening**: TLS/SSL, API authentication, input sanitization
- **Performance Optimization**: Model quantization, auto-tuning, edge configuration
- **Monitoring**: Alert configuration, log aggregation, performance baselines
- **CI/CD**: Automated testing, deployment pipelines, rollback procedures

## 🏗️ Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│              EchoLoc-NN Services                           │
├─────────────────────┼───────────────────────────────────────┤
│  ┌─────────────────┐ │ ┌─────────────────┐ ┌───────────────┐ │
│  │ Inference API   │ │ │ Model Storage   │ │ Hardware I/F  │ │
│  │ (8080)         │ │ │ (Volumes)       │ │ (Serial/GPIO) │ │
│  └─────────────────┘ │ └─────────────────┘ └───────────────┘ │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│              Support Services                              │
├─────────────────────┼───────────────────────────────────────┤
│  ┌─────────────────┐ │ ┌─────────────────┐ ┌───────────────┐ │
│  │ Redis Cache     │ │ │ InfluxDB        │ │ Grafana       │ │
│  │ (6379)         │ │ │ (8086)          │ │ (3000)        │ │
│  └─────────────────┘ │ └─────────────────┘ └───────────────┘ │
└─────────────────────┴───────────────────────────────────────┘
```

## 🐳 Container Deployment

### Development Environment
```bash
# Start development environment
docker compose up echoloc-dev

# Access Jupyter notebook
http://localhost:8888

# Access visualization server  
http://localhost:8080
```

### Production Deployment
```bash
# Build production image
docker build -t echoloc-nn:production --target production .

# Start production services
docker compose up -d echoloc-prod redis influxdb grafana

# Health check
curl http://localhost:8080/health
```

### Hardware Testing
```bash
# Hardware-in-the-loop testing
docker compose up echoloc-hardware

# Connect Arduino to /dev/ttyUSB0
# Run hardware validation tests
```

## ⚙️ Configuration Management

### Environment Variables

#### Core Application
```bash
# Model configuration
MODEL_PATH=/models/echoloc-indoor-v2.pt
DEVICE=cpu  # or cuda
INFERENCE_BATCH_SIZE=1

# Performance settings
OPTIMIZATION_LEVEL=balanced  # fast, balanced, accurate
TARGET_LATENCY_MS=50
ENABLE_CACHING=true

# Hardware interface
ARDUINO_PORT=/dev/ttyUSB0
SENSOR_ARRAY_CONFIG=square_4sensor
```

#### Infrastructure Services
```bash
# Redis caching
REDIS_HOST=redis
REDIS_PORT=6379
CACHE_TTL=3600

# Monitoring
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=your_token_here
METRICS_INTERVAL=10

# Security
API_KEY_REQUIRED=true
TLS_ENABLED=false  # Enable for production
CORS_ORIGINS=*
```

### Configuration Files

#### Production pyproject.toml
```toml
[project]
dependencies = [
    "torch>=2.0.0,<3.0",
    "numpy>=1.21.0,<2.0",
    "scipy>=1.7.0,<2.0",
    "PyYAML>=6.0,<7.0"
]

[project.optional-dependencies]
production = [
    "gunicorn>=20.1.0",
    "redis>=4.0.0",
    "influxdb-client>=1.27.0"
]
```

## 🛡️ Security Configuration

### Required Security Hardening

#### 1. TLS/SSL Configuration
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Update docker-compose.yml
environment:
  - TLS_CERT_PATH=/certs/cert.pem
  - TLS_KEY_PATH=/certs/key.pem
  - TLS_ENABLED=true
```

#### 2. API Authentication
```python
# Add API key validation
from echoloc_nn.security import APIKeyValidator

validator = APIKeyValidator()
@app.before_request
def validate_api_key():
    if not validator.validate(request.headers.get('X-API-Key')):
        abort(401)
```

#### 3. Input Sanitization
```python
# Enhanced input validation
from echoloc_nn.utils.validation import EchoDataValidator

validator = EchoDataValidator(
    strict_mode=True,
    max_frequency=50000,
    min_snr_db=-20
)
```

### Security Checklist
- [ ] TLS/SSL certificates configured
- [ ] API authentication implemented  
- [ ] Input validation hardened
- [ ] Container security scanning enabled
- [ ] Network segmentation configured
- [ ] Secrets management implemented
- [ ] Rate limiting enabled
- [ ] Security headers configured

## ⚡ Performance Optimization

### Model Optimization Pipeline

#### 1. Quantization
```python
from echoloc_nn.optimization import ModelOptimizer

optimizer = ModelOptimizer()
# INT8 quantization for production
quantized_model = optimizer.quantize(
    model, method='dynamic', bits=8
)
```

#### 2. Hardware-Specific Optimization
```python
# CPU optimization
optimizer.optimize_for_cpu(
    enable_mkldnn=True,
    num_threads=4
)

# GPU optimization (if available)
optimizer.optimize_for_gpu(
    enable_tensorrt=True,
    precision='fp16'
)
```

#### 3. Caching Configuration
```python
# Multi-level caching
cache_config = {
    'echo_data_ttl': 300,      # 5 minutes
    'model_results_ttl': 1800,  # 30 minutes
    'geometry_ttl': 3600,       # 1 hour
    'compression_level': 6       # LZ4 compression
}
```

### Performance Monitoring

#### Key Metrics
```yaml
performance_metrics:
  - inference_latency_ms
  - throughput_requests_per_second
  - model_accuracy_percentage
  - cache_hit_rate
  - memory_usage_mb
  - cpu_utilization_percentage
  - gpu_utilization_percentage (if applicable)
```

#### Alerting Thresholds
```yaml
alerts:
  critical:
    - inference_latency_ms > 100
    - accuracy_drop > 10%
    - memory_usage > 80%
  warning:
    - inference_latency_ms > 75
    - cache_hit_rate < 60%
    - cpu_utilization > 70%
```

## 📊 Monitoring & Observability

### Grafana Dashboards

#### 1. System Performance
- Inference latency trends
- Throughput metrics
- Resource utilization
- Cache performance

#### 2. Model Health
- Prediction accuracy
- Confidence score distribution
- Error rate trends
- Model drift detection

#### 3. Hardware Monitoring
- Sensor array status
- Arduino communication health
- Serial port connectivity
- Hardware failure detection

### Log Configuration
```python
logging_config = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'json'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': '/var/log/echoloc-nn.log',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        'echoloc_nn': {
            'handlers': ['console', 'file'],
            'level': 'INFO'
        }
    }
}
```

## 🔧 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: EchoLoc-NN CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          docker build --target development .
          docker run --rm test-image pytest tests/
      
  security-scan:
    runs-on: ubuntu-latest  
    steps:
      - name: Security scan
        run: python scripts/security_scan.py

  deploy:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Build and deploy
        run: |
          docker build -t echoloc-nn:latest .
          docker push registry/echoloc-nn:latest
```

### Deployment Checklist
- [ ] Automated testing passes
- [ ] Security scan clean
- [ ] Performance benchmarks met
- [ ] Health checks configured
- [ ] Monitoring dashboards ready
- [ ] Rollback procedure tested
- [ ] Load testing completed

## 🌐 Scaling & Load Balancing

### Horizontal Scaling
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: echoloc-nn-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: echoloc-nn-api
  template:
    spec:
      containers:
      - name: echoloc-nn
        image: echoloc-nn:production
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi" 
            cpu: "1000m"
```

### Auto-scaling Configuration
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: echoloc-nn-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: echoloc-nn-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📈 Performance Benchmarks

### Target Performance Metrics
```
Inference Latency:
  - Target: <50ms (95th percentile)
  - Current: 88-150ms (needs optimization)

Throughput:
  - Target: >20 requests/second
  - Current: ~6-10 requests/second

Accuracy:
  - Target: >90% confidence on test set
  - Current: 17-75% (needs model improvement)

Resource Usage:
  - Memory: <1GB per instance
  - CPU: <2 cores per instance
  - GPU: Optional acceleration
```

### Load Testing
```bash
# Use Apache Bench for load testing
ab -n 1000 -c 10 -H "Content-Type: application/json" \
   -p test_data.json http://localhost:8080/localize

# Monitor with:
docker stats echoloc-nn-prod
curl http://localhost:8080/metrics
```

## 🚨 Troubleshooting

### Common Issues

#### High Latency
```bash
# Check CPU usage
docker exec echoloc-nn-prod top

# Check memory usage  
docker exec echoloc-nn-prod free -h

# Optimize model
python -m echoloc_nn.optimization.model_optimizer \
  --model-path models/current.pt \
  --optimize-for cpu \
  --quantize int8
```

#### Hardware Communication Errors
```bash
# Check serial port
ls -la /dev/ttyUSB*

# Test Arduino connection
docker exec echoloc-hardware \
  python -c "from echoloc_nn.hardware import UltrasonicArray; \
             array = UltrasonicArray(); array.test_connection()"
```

#### Model Accuracy Issues
```bash
# Run model validation
python -m echoloc_nn.validation.model_validator \
  --model-path models/current.pt \
  --test-data data/validation.npz \
  --threshold 0.9
```

### Monitoring Commands
```bash
# System health
curl http://localhost:8080/health

# Performance metrics
curl http://localhost:8080/metrics

# Cache statistics
docker exec redis redis-cli info stats

# Application logs
docker logs echoloc-nn-prod --tail 100 -f
```

## 📚 Additional Resources

- **Architecture Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: [docs/api/](docs/api/)
- **Hardware Setup**: [docs/hardware/](docs/hardware/)  
- **Development Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)
- **Security Guide**: [SECURITY.md](SECURITY.md)

## 🆘 Support & Maintenance

### Regular Maintenance Tasks
- [ ] Update dependencies monthly
- [ ] Review security scans weekly
- [ ] Performance benchmarks daily
- [ ] Backup models and data weekly
- [ ] Monitor resource usage continuously
- [ ] Update documentation as needed

### Contact Information
- **Issues**: GitHub Issues tracker
- **Security**: security@terragon.ai
- **Support**: support@terragon.ai

---

**Status**: Production deployment infrastructure ready with security and performance hardening required for full production deployment.