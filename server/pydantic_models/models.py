from pydantic import BaseModel
from typing import Optional


class ImageRequest(BaseModel):
    prompt: str
    model: str
    consistency_decoder: bool = False
    negative_prompt: Optional[str] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    file_name: str = "client_x.jpg"
