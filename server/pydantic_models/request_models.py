from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union, Dict
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


class NERRequest(BaseModel):
    model: Optional[str] = None
    text: str
    entities: List[str]
    mask_entities: Optional[List[str]] = []


# ----------------- Chat Completion -----------------
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.2
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None


# ----------------- End Chat Completion -----------------

# ----------------- Embeddings -----------------


class ImageUrl(BaseModel):
    url: str


class ImageInput(BaseModel):
    type: str = Field(..., pattern="^image_url$")
    image_url: ImageUrl


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], ImageInput, List[ImageInput]]
    model: Optional[str] = None
    encoding_format: str = "float"

    def is_image_request(self) -> bool:
        """Check if this is an image embedding request."""
        if isinstance(self.input, (ImageInput, list)):
            if isinstance(self.input, list):
                return len(self.input) > 0 and isinstance(self.input[0], ImageInput)
            return True
        return False

    def get_image_urls(self) -> List[str]:
        """Extract image URLs from the request."""
        if not self.is_image_request():
            raise ValueError("Not an image embedding request")

        if isinstance(self.input, ImageInput):
            return [self.input.image_url.url]
        return [item.image_url.url for item in self.input]


# ----------------- End Embeddings -----------------
