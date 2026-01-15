import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 获取AliceBot根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config:
    # --- LLM Settings ---

    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("SILICON_API_KEY")
    SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1") or os.getenv("SILICON_URL")

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
    SMALL_PROVIDER = "siliconflow"
    PROVIDER = "siliconflow"
    # mimo
    # aizex
    if SMALL_PROVIDER == "aizex":
        SMALL_MODEL_API_KEY = AIZEX_API_KEY
        SMALL_MODEL_URL = AIZEX_URL
        SMALL_MODEL = AIZEX_MODEL
    elif SMALL_PROVIDER == "siliconflow":
        SMALL_MODEL_API_KEY = SILICONFLOW_API_KEY
        SMALL_MODEL_URL = SILICONFLOW_BASE_URL
        SMALL_MODEL = SMALL_LLM_MODEL_NAME
    elif SMALL_PROVIDER == "mimo":
        SMALL_MODEL_API_KEY = MIMO_API_KEY
        SMALL_MODEL_URL = MIMO_BASE_URL
        SMALL_MODEL = MIMO_MODEL

    # dream proactive_agent unified_agent
    if PROVIDER == "aizex":
        MODEL_API_KEY = AIZEX_API_KEY
        MODEL_URL = AIZEX_URL
        MODEL_NAME = AIZEX_MODEL
    elif PROVIDER == "siliconflow":
        MODEL_API_KEY = SILICONFLOW_API_KEY
        MODEL_URL = SILICONFLOW_BASE_URL
        MODEL_NAME = LLM_MODEL_NAME
    elif PROVIDER == "mimo":
        MODEL_API_KEY = MIMO_API_KEY
        MODEL_URL = MIMO_BASE_URL
        MODEL_NAME = MIMO_MODEL

    # --- Vector DB Settings ---
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", os.path.join(BASE_DIR, "data", "chroma_db"))
    COLLECTION_NAME = "anima_memories"

    # --- Tool Settings ---
    MAX_SEARCH_RESULTS = 3

    # --- Emotion & Personality Settings ---
    # 初始情绪状态
    DEFAULT_VALENCE = 0.1  # 略微积极
    DEFAULT_AROUSAL = 0.5  # 平静且专注

    # --- System Paths ---
    LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "log"))



config = Config()
