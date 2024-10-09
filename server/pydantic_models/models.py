from pydantic import BaseModel


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
    temp: float = 0.1
    prob: float = 0.2
    max_tokens: int = 1024
