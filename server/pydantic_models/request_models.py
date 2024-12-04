from pydantic import BaseModel, field_validator
from typing import Optional, List
from enum import Enum


class ImageRequest(BaseModel):
    prompt: str
    model: str
    consistency_decoder: bool = False
    negative_prompt: str = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    height: int = 512
    width: int = 512
    seed: int = None
    file_name: str = "client_x.jpg"


class TextRequest(BaseModel):
    model: str
    task: str
    operation: str
    temp: Optional[float] = 0.1
    prob: Optional[float] = 0.2
    max_tokens: Optional[int] = 2048

    @field_validator("temp", "prob", "max_tokens", mode="before")
    def set_defaults(cls, v, info):
        if v is not None:
            return v
        defaults = {"temp": 0.1, "prob": 0.2, "max_tokens": 1024}
        return defaults[info.field_name]


class TextJSONRequest(BaseModel):
    model: str
    prompt: str
    json_schema: str
    operation: str
    context: Optional[str] = None


class NERRequest(BaseModel):
    model: str
    text: str
    entities: List[str]
    mask_entities: Optional[List[str]] = []


# ----------------- Chat Completion -----------------
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 0.5
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None
