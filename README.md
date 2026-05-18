# AI-Based Face Recognition Attendance System

A real-time face recognition backend system built with **Flask** and **InsightFace**. The system extracts face embeddings, filters them based on face angles, and utilizes **PostgreSQL (pgvector)** for fast facial matching. The entire system is fully containerized using **Docker**.

## Features
- **Real-time Optimization**: Low-latency face processing.
- **Frontal Face Filtering**: Registers and recognizes only proper frontal faces.
- **Vector Database Matching**: Uses PostgreSQL with `pgvector` for accurate cosine similarity search.
- **Dockerized Environment**: Runs seamlessly via Docker Compose.

## Tech Stack
- **Backend Framework**: Flask
- **AI/ML Libraries**: InsightFace, OpenCV, PyTorch, ONNX Runtime
- **Database**: PostgreSQL (with pgvector extension)
- **DevOps**: Docker, Docker Compose

## Quick Start with Docker
1. Clone the repository and navigate to the folder.
2. Build and run containers:
   ```bash
   docker-compose up --build

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
