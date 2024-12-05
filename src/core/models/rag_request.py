from pydantic import BaseModel


class Query(BaseModel):
    query: str
    top_k: int = 5
