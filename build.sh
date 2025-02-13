#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
LOCAL_BASE_IMAGE="remote-client-server-base:latest"
REGISTRY_BASE_IMAGE="docker.semoss.org/genai/remote-client-server-base:latest"

# Check if base image needs to be built
build_base_image() {
    echo -e "${YELLOW}Checking if base image needs to be built...${NC}"
    
    if docker images | grep -q "^remote-client-server-base.*latest"; then
        echo -e "${GREEN}Base image already exists locally${NC}"
        return 0
    else
        echo -e "${YELLOW}Base image not found locally, building...${NC}"
        docker build -f Dockerfile.base -t $LOCAL_BASE_IMAGE .
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Base image build failed!${NC}"
            exit 1
        fi
        echo -e "${GREEN}Base image built successfully!${NC}"
        return 0
    fi
}

# Build server image
build_server_image() {
    local base_image=$1
    echo -e "${GREEN}Building server image with base: $base_image${NC}"
    docker build --build-arg BASE_IMAGE=$base_image -t remote-client-server:latest .
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Server image build failed!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Server image built successfully!${NC}"
}

# option to use registry image
USE_REGISTRY=false
while getopts "r" opt; do
  case $opt in
    r) USE_REGISTRY=true ;;
  esac
done

if [ "$USE_REGISTRY" = true ]; then
    echo -e "${YELLOW}Using registry base image...${NC}"
    build_server_image $REGISTRY_BASE_IMAGE
else
    echo -e "${YELLOW}Using local base image...${NC}"
    build_base_image
    build_server_image $LOCAL_BASE_IMAGE
fi