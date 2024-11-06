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
