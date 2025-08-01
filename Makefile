.PHONY: help install install-dev clean test lint format type-check security-check all-checks build

help:
	@echo "Available commands:"
	@echo "  install      Install package for production"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  test         Run test suite"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo "  security-check Run security vulnerability scans"
	@echo "  all-checks   Run all quality checks"
	@echo "  build        Build package"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

test:
	pytest tests/ -v --cov=echoloc_nn --cov-report=html --cov-report=term

lint:
	flake8 echoloc_nn tests
	black --check echoloc_nn tests
	isort --check-only echoloc_nn tests

format:
	black echoloc_nn tests
	isort echoloc_nn tests

type-check:
	mypy echoloc_nn

security-check:
	detect-secrets scan --baseline .secrets.baseline
	pip-audit

all-checks: lint type-check security-check test

build: clean
	python -m build

# Development server for real-time visualization
serve-viz:
	python -m echoloc_nn.visualization.server --port 8080

# Hardware testing utilities
test-hardware:
	python -m pytest tests/ -m hardware -v

# Generate documentation
docs:
	cd docs && make html

# Clean and rebuild documentation  
docs-clean:
	cd docs && make clean && make html