from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str

    MONGO_URI: str
    MONGO_DB: str

    CHAT_MODEL: str 
    EMB_MODEL: str 

    DEFAULT_SCORE_THRESHOLD: float

settings = Settings()
