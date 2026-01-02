from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str

    MONGO_URI: str = "mongodb+srv://admin:admin123@cluster0.sqtrcby.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    MONGO_DB: str = "test"

    CHAT_MODEL: str = "gpt-4o-mini"
    EMB_MODEL: str = "text-embedding-3-small"

    DEFAULT_SCORE_THRESHOLD: float = 0.25

settings = Settings()
