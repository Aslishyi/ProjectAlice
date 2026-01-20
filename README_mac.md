# Project Alice macOS 移植指南

本指南将帮助您将 Project Alice 从 Windows 移植到 macOS 系统。

## 前置要求

1. **macOS 系统**：推荐使用 macOS 12.0 或更高版本
2. **Anaconda**：已安装并配置好环境变量
3. **OneBot v11 客户端**：如 [NapCatQQ](https://github.com/NapNeko/NapCatQQ) 或 [Lagrange.Core](https://github.com/LagrangeDev/Lagrange.Core)

## 移植步骤

### 1. 准备项目文件

将整个 ProjectAlice 目录复制到 macOS 上的合适位置，例如：
```bash
cp -r /path/to/windows/ProjectAlice ~/Documents/ProjectAlice
```

### 2. 配置环境变量

编辑 `.env` 文件，确保所有必要的 API 密钥和配置项都已正确填写：

```bash
cd ~/Documents/ProjectAlice/AliceBot
cp .env.example .env
nano .env
```

主要配置项：
- `SILICON_API_KEY`：硅基流动 API Key
- `LLM_MODEL_NAME`：主模型名称
- `TAVILY_API_KEY`：Tavily 搜索 API Key
- `VECTOR_DB_PATH`：向量数据库存储路径（macOS 上建议使用绝对路径）

### 3. 安装依赖

使用提供的 macOS 专用脚本安装依赖：

```bash
cd ~/Documents/ProjectAlice/AliceBot
chmod +x setup_mac.sh run_mac.sh
./setup_mac.sh
```

该脚本将：
- 创建名为 `alicebot_env` 的 conda 虚拟环境
- 安装所有必要的依赖包
- 移除 Windows 特定的 `mss` 库
- 从 `.env.example` 创建 `.env` 文件（如果不存在）

### 4. 启动 OneBot 客户端

在 macOS 上启动 OneBot 客户端（如 NapCatQQ），并配置反向 WebSocket 地址为：
```
ws://127.0.0.1:6199/ws
```

### 5. 启动 AliceBot

使用提供的启动脚本启动 AliceBot：

```bash
cd ~/Documents/ProjectAlice/AliceBot
./run_mac.sh
```

## 手动安装指南（可选）

如果自动脚本出现问题，您可以按照以下步骤手动安装：

### 1. 创建并激活 conda 虚拟环境

```bash
conda create -n alicebot_env python=3.11 -y
conda activate alicebot_env
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 移除 Windows 特定依赖

```bash
pip uninstall -y mss
```

### 4. 启动 AliceBot

```bash
python qq_server.py
```

## 常见问题与解决方案

### 1. Anaconda 未找到

确保 Anaconda 已正确安装并添加到环境变量中。您可以通过以下命令验证：
```bash
conda --version
```

如果未找到，请尝试：
```bash
export PATH="~/opt/anaconda3/bin:$PATH"
source ~/.bash_profile
```

### 2. 端口冲突

如果 6199 端口已被占用，您可以修改 `qq_server.py` 文件中的端口配置：

```python
# 在 qq_server.py 中查找以下行并修改端口
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6199)
```

### 3. 向量数据库问题

在 macOS 上，建议使用绝对路径存储向量数据库：

```bash
# 在 .env 文件中设置
VECTOR_DB_PATH=~/Documents/ProjectAlice/AliceBot/data/chroma_db
```

### 4. 权限问题

如果脚本无法执行，请确保已添加执行权限：
```bash
chmod +x setup_mac.sh run_mac.sh
```

## 项目结构

```
ProjectAlice/
├── AliceBot/
│   ├── app/                # 应用代码
│   │   ├── background/     # 后台任务
│   │   ├── core/           # 核心配置
│   │   ├── graph/          # LangGraph 节点
│   │   ├── memory/         # 记忆模块
│   │   ├── plugins/        # 插件系统
│   │   ├── tools/          # 工具集
│   │   └── utils/          # 辅助工具
│   ├── .env.example        # 环境变量示例
│   ├── requirements.txt    # 依赖列表
│   ├── qq_server.py        # 主入口
│   ├── setup_mac.sh        # macOS 安装脚本
│   └── run_mac.sh          # macOS 启动脚本
├── README.md               # 项目说明
└── README_mac.md           # macOS 移植指南
```

## 更新说明

- 移除了 Windows 特定的 `mss` 库（屏幕截图功能）
- 保留了所有其他核心功能
- 提供了 macOS 专用的安装和启动脚本

## 联系方式

如果您在移植过程中遇到问题，请参考项目文档或提交 Issue。

---

**享受使用 AliceBot 吧！** 🤖✨
