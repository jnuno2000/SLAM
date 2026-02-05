# SLAM Project Makefile
# Provides convenient commands for development and testing

.PHONY: help install install-dev test test-verbose lint format clean run

# Default target
help:
	@echo "SLAM - Simultaneous Localization and Mapping"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make install-dev  Install dev dependencies (includes testing)"
	@echo "  make test         Run unit tests"
	@echo "  make test-verbose Run tests with verbose output"
	@echo "  make lint         Run code linting"
	@echo "  make format       Format code with black"
	@echo "  make clean        Remove build artifacts"
	@echo "  make run          Run SLAM on sample data"
	@echo ""

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Run tests
test:
	python -m pytest tests/ -v

# Run tests with coverage
test-cov:
	python -m pytest tests/ -v --cov=slam --cov-report=term-missing

# Run tests verbose
test-verbose:
	python -m pytest tests/ -v -s

# Lint code
lint:
	flake8 slam/ tests/ --max-line-length=100 --ignore=E501
	mypy slam/ --ignore-missing-imports

# Format code
format:
	black slam/ tests/ main.py --line-length=100

# Clean build artifacts
clean:
	rm -rf __pycache__
	rm -rf slam/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf *.egg-info
	rm -rf dist/
	rm -rf build/
	rm -rf .coverage
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Run SLAM on data file
run:
	python main.py data.csv --visualize

# Run SLAM and save output
run-save:
	python main.py data.csv --output results.png

# Type checking
typecheck:
	mypy slam/ --ignore-missing-imports --strict
