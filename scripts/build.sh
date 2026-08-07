#!/bin/bash
set -e

echo "Building Docker images..."
docker-compose -f docker/docker-compose.yml build
echo "Build complete."
