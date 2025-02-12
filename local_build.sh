#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
LOCAL_BASE_IMAGE="remote-client-server-base:latest"

# Check if base image needs to be built
build_base_image() {
    echo -e "${YELLOW}Checking if base image needs to be built...${NC}"
    if ! docker image inspect $LOCAL_BASE_IMAGE >/dev/null 2>&1; then
        echo -e "${YELLOW}Building base image (this will take a while)...${NC}"
        docker build -f Dockerfile.base -t $LOCAL_BASE_IMAGE .
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Base image build failed!${NC}"
            exit 1
        fi
        echo -e "${GREEN}Base image built successfully!${NC}"
        return 0
    fi
    echo -e "${GREEN}Base image already exists${NC}"
    return 0
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

# Main build process
echo -e "${YELLOW}Building for local development...${NC}"
build_base_image
build_server_image $LOCAL_BASE_IMAGE