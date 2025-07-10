from pydantic import BaseModel
from typing import Optional

class DiaryRequest(BaseModel):
    content: str
    model: Optional[str] = "gpt-4o"

class DiaryResponse(BaseModel):
    emotion: str
    feedback: str
    therapy: str
