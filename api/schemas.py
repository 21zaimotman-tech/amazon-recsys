from pydantic import BaseModel
from typing import Optional, List

class Item(BaseModel):
    item_id: str
    title: Optional[str] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    avg_rating: Optional[float] = None

class RecResponse(BaseModel):
    model_label: str                 # which model produced these (shown in UI)
    items: List[Item]
    latency_ms: dict                 # per-component timings


class RegisterRequest(BaseModel):
    user_id: str
    password: str


class LoginRequest(BaseModel):
    user_id: str
    password: str


class AuthResponse(BaseModel):
    user_id: str
    n_interactions: int


class CartResponse(BaseModel):
    items: List[Item]


class OrderItem(Item):
    purchased_at: Optional[str] = None


class OrdersResponse(BaseModel):
    items: List[OrderItem]
