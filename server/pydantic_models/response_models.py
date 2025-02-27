from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class NERResponse(BaseModel):
    status: str = Field(..., description="Status of the response")
    output: str = Field(..., description="List of detected entities")
    raw_output: List[Dict[str, Any]] = Field(
        ..., description="Raw output from the model"
    )
    mask_values: Dict[str, Any] = Field(..., description="Masked values")
    input: str = Field(..., description="Input text")
    entities: List[str] = Field(..., description="List of entities")


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict]
    usage: Dict[str, int]


class SentimentGenResponse(BaseModel):
    id: str = Field(..., description="Unique Id of the response")
    object: str
    created: int = Field(..., description="Created time of the response")
    model: str = Field(..., description="Used model")
    sentiment_gen_report: List[Dict]  = Field(..., description="List of generated sentiments with the input texts")
