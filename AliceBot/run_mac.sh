#!/bin/bash

echo "=== Starting Project Alice on macOS ==="

# 检查Anaconda是否安装
if ! command -v conda &> /dev/null; then
    echo "Anaconda could not be found. Please install Anaconda first."
    exit 1
fi

# 检查虚拟环境是否存在
ENV_NAME="alicebot_env"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Virtual environment $ENV_NAME not found. Please run ./setup_mac.sh first."
    exit 1
fi

# 激活虚拟环境
echo "Activating conda environment: $ENV_NAME"
source activate $ENV_NAME

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file from .env.example and fill in the required API keys."
    exit 1
fi

# 启动AliceBot
echo "Starting AliceBot..."
python3 qq_server.py

