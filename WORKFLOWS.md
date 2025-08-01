# CI/CD Workflows Documentation

## Overview

EchoLoc-NN uses a comprehensive CI/CD pipeline designed for machine learning projects with hardware components. The workflows are built for GitHub Actions but can be adapted for other CI/CD platforms.

## Workflow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │   Integration   │    │   Deployment    │
│                 │    │                 │    │                 │
│ • Code Quality  │    │ • Full Testing  │    │ • Package Build │
│ • Unit Tests    │ ━━▶│ • Hardware Test │ ━━▶│ • Docker Images │
│ • Security Scan │    │ • Performance   │    │ • Documentation │
│ • Type Check    │    │ • Cross-platform│    │ • Release       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Workflow Files

### 1. Continuous Integration (`ci.yml`)

**Triggers**: Push to any branch, Pull requests
**Purpose**: Code quality, testing, security validation

```yaml
name: Continuous Integration
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Code formatting check
      run: |
        black --check echoloc_nn tests
        isort --check-only echoloc_nn tests
    
    - name: Linting
      run: flake8 echoloc_nn tests
    
    - name: Type checking
      run: mypy echoloc_nn
    
    - name: Security scanning
      run: |
        bandit -r echoloc_nn
        safety check
    
    - name: Unit tests
      run: |
        pytest tests/ -v --cov=echoloc_nn --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### 2. Hardware Testing (`hardware-test.yml`)

**Triggers**: Scheduled (daily), Manual dispatch
**Purpose**: Hardware-in-the-loop testing with real sensors

```yaml
name: Hardware Testing
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:

jobs:
  hardware-test:
    runs-on: [self-hosted, hardware]  # Custom runner with hardware
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup hardware environment
      run: |
        docker-compose up -d echoloc-hardware
    
    - name: Run hardware tests
      run: |
        docker-compose exec -T echoloc-hardware \
          python -m pytest tests/ -m hardware -v --junit-xml=hardware-results.xml
    
    - name: Cleanup hardware
      if: always()
      run: |
        docker-compose down
```

### 3. Performance Benchmarking (`benchmark.yml`)

**Triggers**: Push to main, Release tags
**Purpose**: Performance regression testing and benchmarking

```yaml
name: Performance Benchmarking
on:
  push:
    branches: [ main ]
  release:
    types: [ published ]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]  # GPU-enabled runner
    
    steps:
    - uses: actions/checkout@v4
      with:
        lfs: true  # For model files
    
    - name: Setup environment
      run: |
        pip install -e ".[dev]"
    
    - name: Run inference benchmarks
      run: |
        python benchmarks/inference_speed.py --output benchmark-results.json
    
    - name: Run training benchmarks
      run: |
        python benchmarks/training_performance.py --epochs 5
    
    - name: Compare with baseline
      run: |
        python benchmarks/compare_baseline.py benchmark-results.json
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark-results.json
```

### 4. Security Scanning (`security.yml`)

**Triggers**: Scheduled (weekly), Push to main
**Purpose**: Comprehensive security vulnerability scanning

```yaml
name: Security Scanning
on:
  schedule:
    - cron: '0 3 * * 1'  # Weekly on Monday at 3 AM
  push:
    branches: [ main ]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Dependency vulnerability scan
      run: |
        pip install pip-audit
        pip-audit --format=json --output=vulnerability-report.json
    
    - name: Secret scanning
      run: |
        pip install detect-secrets
        detect-secrets scan --baseline .secrets.baseline
    
    - name: Container security scan
      run: |
        docker build -t echoloc-nn-security .
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy image echoloc-nn-security
    
    - name: Code security analysis
      uses: github/codeql-action/analyze@v2
      with:
        languages: python
```

### 5. Release Pipeline (`release.yml`)

**Triggers**: Git tags (v*)
**Purpose**: Automated package building and distribution

```yaml
name: Release Pipeline
on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Validate release tag
      run: |
        echo "Building release for tag: ${{ github.ref_name }}"
    
    - name: Build Python package
      run: |
        pip install build
        python -m build
    
    - name: Build Docker images
      run: |
        docker build -t echoloc-nn:${{ github.ref_name }} .
        docker build -t echoloc-nn:latest .
    
    - name: Push to Docker Hub
      env:
        DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
        DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
      run: |
        echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
        docker push echoloc-nn:${{ github.ref_name }}
        docker push echoloc-nn:latest
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: |
        pip install twine
        twine upload dist/*
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
```

### 6. Documentation (`docs.yml`)

**Triggers**: Push to main, Pull requests affecting docs/
**Purpose**: Build and deploy documentation

```yaml
name: Documentation
on:
  push:
    branches: [ main ]
    paths: [ 'docs/**', '**.md', 'echoloc_nn/**/*.py' ]
  pull_request:
    paths: [ 'docs/**', '**.md' ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        pip install sphinx sphinx-rtd-theme
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

### 7. Terragon Value Discovery (`terragon.yml`)

**Triggers**: Scheduled (daily), Push to main
**Purpose**: Autonomous value discovery and backlog management

```yaml
name: Terragon Value Discovery
on:
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM UTC
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  value-discovery:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Need full history for analysis
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install pyyaml
    
    - name: Run value discovery
      run: |
        python .terragon/value_discovery.py
    
    - name: Create issues from discoveries
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        python .terragon/create_github_issues.py
    
    - name: Update project board
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        python .terragon/update_project_board.py
```

## Runner Configuration

### Standard Runners
- **ubuntu-latest**: Primary CI/CD runner for most jobs
- **windows-latest**: Windows compatibility testing
- **macos-latest**: macOS compatibility testing

### Self-Hosted Runners

#### Hardware Testing Runner
```yaml
# Labels: [self-hosted, hardware]
# Requirements:
# - Ubuntu 20.04 LTS
# - Docker and docker-compose
# - Physical ultrasonic sensors connected
# - Serial port access (dialout group)
# - Arduino CLI installed
```

#### Performance Testing Runner
```yaml
# Labels: [self-hosted, gpu]
# Requirements:
# - Ubuntu 20.04 LTS with NVIDIA drivers
# - CUDA 11.8+
# - Docker with nvidia-container-runtime
# - 16GB+ RAM, 100GB+ storage
# - GPU with 8GB+ VRAM
```

## Secrets Configuration

Required repository secrets:

```yaml
DOCKER_USERNAME: Docker Hub username
DOCKER_PASSWORD: Docker Hub access token
PYPI_TOKEN: PyPI publishing token
CODECOV_TOKEN: Codecov upload token
SLACK_WEBHOOK: Slack notification webhook (optional)
```

## Branch Protection Rules

### Main Branch
- Require pull request reviews (2 reviewers)
- Require status checks:
  - `quality-checks`
  - `security-scan`
  - `build-docs`
- Require branches to be up to date
- Include administrators in restrictions

### Develop Branch
- Require pull request reviews (1 reviewer)
- Require status checks:
  - `quality-checks`
- Allow force pushes for maintainers

## Workflow Best Practices

### 1. Fail Fast Strategy
- Run fastest checks first (linting, formatting)
- Run expensive tests (hardware, GPU) only after basic checks pass
- Use matrix strategies for parallel execution

### 2. Caching Strategy
```yaml
- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}

- name: Cache Docker layers
  uses: actions/cache@v3
  with:
    path: /tmp/.buildx-cache
    key: ${{ runner.os }}-buildx-${{ github.sha }}
```

### 3. Conditional Execution
```yaml
- name: Run hardware tests
  if: contains(github.event.head_commit.message, '[test-hardware]') || 
      github.event_name == 'schedule'
  run: make test-hardware
```

### 4. Artifact Management
```yaml
- name: Upload test results
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: test-results-${{ matrix.python-version }}
    path: test-results.xml
    retention-days: 30
```

## Integration with External Services

### Codecov Integration
```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    flags: unittests
    name: codecov-umbrella
```

### Slack Notifications
```yaml
- name: Notify Slack on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#echoloc-nn-dev'
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

## Monitoring and Observability

### Workflow Analytics
- Track build times and success rates
- Monitor test flakiness and reliability
- Analyze deployment frequency and lead time

### Performance Metrics
- Inference latency benchmarks
- Model accuracy on validation sets
- Memory usage and GPU utilization
- Docker image size and build time

## Troubleshooting Common Issues

### 1. Hardware Test Failures
```bash
# Check hardware runner status
docker ps
docker logs echoloc-nn-hardware

# Verify sensor connections
python -m echoloc_nn.hardware.test_sensors
```

### 2. Memory Issues in GPU Jobs
```yaml
# Add memory management
env:
  PYTORCH_CUDA_ALLOC_CONF: max_split_size_mb:128
```

### 3. Cache Corruption
```bash
# Clear GitHub Actions cache
gh api repos/:owner/:repo/actions/caches --method DELETE
```

## Future Enhancements

### Planned Improvements
1. **Multi-environment testing**: Test across different acoustic environments
2. **Automated model validation**: Compare new models against benchmarks
3. **Security compliance**: SOC 2, GDPR compliance checks
4. **Performance regression detection**: Automated performance alerts
5. **Cross-platform mobile builds**: iOS and Android deployment

### Advanced Features
1. **Canary deployments**: Gradual rollout with automatic rollback
2. **A/B testing framework**: Compare model variants in production
3. **Federated learning workflows**: Distributed training coordination
4. **Edge deployment automation**: Raspberry Pi fleet management

## Contact and Support

For workflow issues or questions:
- Create an issue with label `ci/cd`
- Contact the DevOps team: devops@echoloc-nn.org
- Check the troubleshooting guide in docs/