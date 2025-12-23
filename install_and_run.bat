@echo off
chcp 65001
title Project Alice - Auto Setup & Launcher

echo ===================================================
echo        Project Alice 一键部署脚本
echo ===================================================

REM 1. 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未检测到 Python，请先安装 Python 3.11 或以上版本并添加到环境变量。
    pause
    exit /b
)

REM 2. 创建虚拟环境
if not exist "venv" (
    echo [INFO] 正在创建虚拟环境 (venv)...
    python -m venv venv
) else (
    echo [INFO] 虚拟环境已存在。
)

REM 3. 激活环境并安装依赖
echo [INFO] 正在激活环境并检查依赖...
call venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 4. 检查配置文件
if not exist ".env" (
    echo [WARN] 未检测到 .env 文件！
    echo [INFO] 正在从模板创建 .env 文件，请稍后手动填入 API Key。
    copy .env.example .env
    echo [INFO] .env 文件已创建。
    echo.
    echo ---------------------------------------------------
    echo 请现在打开项目目录下的 .env 文件，填入你的 API KEY。
    echo 配置完成后，按任意键继续启动服务...
    echo ---------------------------------------------------
    pause
)

REM 5. 启动服务
echo [INFO] 正在启动 Project Alice 服务...
python qq_server.py

pause