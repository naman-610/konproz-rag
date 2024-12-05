import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.environ.get(
        "OPENAI_API_KEY",
        "",
    )
    TEXT_LLM_MODEL: str = os.environ.get("TEXT_LLM_MODEL", "gpt-4o")


config_settings = Settings()
