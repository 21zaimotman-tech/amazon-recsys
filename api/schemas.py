from pydantic import BaseModel
from typing import Optional, List

class Item(BaseModel):
    item_id: str
    title: Optional[str] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None

class RecResponse(BaseModel):
    model_label: str                 # which model produced these (shown in UI)
    items: List[Item]
    latency_ms: dict                 # per-component timings
