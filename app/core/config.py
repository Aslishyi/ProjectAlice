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

    AIZEX_API_KEY = os.getenv("AIZEX_API_KEY")
    AIZEX_URL = os.getenv("AIZEX_URL", "https://a1.aizex.me/v1")
    AIZEX_MODEL = os.getenv("AIZEX_MODEL")

    # 推荐使用支持 Function Calling 和强逻辑能力的模型
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    SMALL_LLM_MODEL_NAME = os.getenv("SMALL_LLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
    TEMPERATURE = 0.7

    # vision_router context_filter memory_saver psychology summarizer
    SMALL_MODEL_API_KEY = MIMO_API_KEY
    SMALL_MODEL_URL = MIMO_BASE_URL
    SMALL_MODEL = MIMO_MODEL

    # dream proactive_agent unified_agent
    MODEL_API_KEY = SILICONFLOW_API_KEY
    MODEL_URL = SILICONFLOW_BASE_URL
    MODEL_NAME = LLM_MODEL_NAME

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



config = Config()
