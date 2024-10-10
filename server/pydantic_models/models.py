from pydantic import BaseModel, field_validator
from typing import Optional


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


class InstructionRequest(BaseModel):
    model: str
    task: str
    temp: Optional[float] = 0.1
    prob: Optional[float] = 0.2
    max_tokens: Optional[int] = 2048

    @field_validator("temp", "prob", "max_tokens", mode="before")
    def set_defaults(cls, v, info):
        if v is not None:
            return v
        defaults = {"temp": 0.1, "prob": 0.2, "max_tokens": 1024}
        return defaults[info.field_name]
