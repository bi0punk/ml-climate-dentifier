.PHONY: run train test docker-build docker-run clean

# Run inference
run:
	python app.py

# Run with recording
run-record:
	python app.py --record

# Run training
train:
	python -m training.train

train-mobilenetv2:
	python -m training.train --backbone mobilenetv2

train-evaluate:
	python -m training.train --evaluate

# Build dataset
dataset:
	python scripts/build_dataset.py

dataset-analyze:
	python scripts/build_dataset.py --analyze

label-captures:
	python scripts/label_captures.py

# Test
test:
	python -m pytest tests/ -v

# Docker
docker-build:
	docker build -t ml-climate-dentifier .

docker-run:
	docker run --rm --device=/dev/video0:/dev/video0 ml-climate-dentifier

# Utils
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
