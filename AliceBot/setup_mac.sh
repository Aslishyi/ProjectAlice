#!/bin/bash

echo "=== Project Alice macOS Setup ==="

# 检查Anaconda是否安装
if ! command -v conda &> /dev/null; then
    echo "Anaconda could not be found. Please install Anaconda first."
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python version: $PYTHON_VERSION"

# 创建conda虚拟环境
ENV_NAME="alicebot_env"
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.11 -y

# 激活虚拟环境
echo "Activating conda environment: $ENV_NAME"
source activate $ENV_NAME

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 移除Windows特定依赖
pip uninstall -y mss

# 配置文件
if [ ! -f ".env" ]; then
    echo "Creating .env from example..."
    cp .env.example .env
    echo "Please edit .env with your API keys."
fi

echo ""
echo "=== Setup Complete! ==="
echo "To activate the environment: conda activate $ENV_NAME"
echo "To start AliceBot: ./run_mac.sh"
echo ""
