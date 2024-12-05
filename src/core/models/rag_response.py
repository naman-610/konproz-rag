from pydantic import BaseModel


class ResponseModel(BaseModel):
    query: str
    response: str
