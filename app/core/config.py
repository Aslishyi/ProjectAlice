import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Config:
    # --- LLM Settings ---

    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

    MIMO_API_KEY = os.getenv("MIMO_API_KEY")
    MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    MIMO_MODEL = os.getenv("MIMO_MODEL")

    # 推荐使用支持 Function Calling 和强逻辑能力的模型
    MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    SMALL_LLM_MODEL_NAME = os.getenv("SMALL_LLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
    TEMPERATURE = 0.7

    # --- Vector DB Settings ---
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    COLLECTION_NAME = "anima_memories"

    # --- Tool Settings ---
    MAX_SEARCH_RESULTS = 3

    # --- Emotion & Personality Settings ---
    # 初始情绪状态
    DEFAULT_VALENCE = 0.1  # 略微积极
    DEFAULT_AROUSAL = 0.5  # 平静且专注

    # --- System Paths ---
    LOG_DIR = "./logs"

    # --- Safety Thresholds ---
    # 如果检测到敏感词的阈值 (0-1)
    SENSITIVE_THRESHOLD = 0.8


config = Config()
