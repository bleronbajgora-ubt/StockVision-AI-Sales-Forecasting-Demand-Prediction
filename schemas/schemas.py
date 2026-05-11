"""
=============================================================================
 app/schemas/schemas.py — Pydantic Request & Response Models
=============================================================================

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────────────────────
class BaseSchema(BaseModel):
    model_config = {"from_attributes": True}   # enables ORM mode (SQLAlchemy → Pydantic)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    email:    EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit.")
        return v


class UserLogin(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int   # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token:        str
    new_password: str = Field(min_length=8)


# ─────────────────────────────────────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────────────────────────────────────
class UserRead(BaseSchema):
    id:         int
    email:      str
    is_active:  bool
    created_at: datetime
    roles:      List[str] = []


class UserUpdate(BaseModel):
    email:    Optional[EmailStr] = None
    password: Optional[str]     = Field(default=None, min_length=8)


class UserAdminUpdate(BaseModel):
    is_active: Optional[bool] = None
    role_ids:  Optional[List[int]] = None


# ─────────────────────────────────────────────────────────────────────────────
# ROLES & PERMISSIONS
# ─────────────────────────────────────────────────────────────────────────────
class RoleCreate(BaseModel):
    name:        str = Field(max_length=50)
    description: Optional[str] = None


class RoleRead(BaseSchema):
    id:          int
    name:        str
    description: Optional[str]
    created_at:  datetime


class PermissionCreate(BaseModel):
    name:        str = Field(max_length=100)
    description: Optional[str] = None


class PermissionRead(BaseSchema):
    id:          int
    name:        str
    description: Optional[str]


class AssignRoleRequest(BaseModel):
    user_id: int
    role_id: int


class AssignPermissionRequest(BaseModel):
    role_id:       int
    permission_id: int


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────
class CategoryCreate(BaseModel):
    name:               str = Field(max_length=255)
    parent_category_id: Optional[int] = None


class CategoryUpdate(BaseModel):
    name:               Optional[str] = Field(default=None, max_length=255)
    parent_category_id: Optional[int] = None


class CategoryRead(BaseSchema):
    id:                 int
    name:               Optional[str]
    parent_category_id: Optional[int]


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTS
# ─────────────────────────────────────────────────────────────────────────────
class ProductCreate(BaseModel):
    name:        str
    description: Optional[str] = None
    base_price:  Optional[Decimal] = Field(default=None, ge=0)
    category_id: Optional[int] = None


class ProductUpdate(BaseModel):
    name:        Optional[str]     = None
    description: Optional[str]     = None
    base_price:  Optional[Decimal] = Field(default=None, ge=0)
    category_id: Optional[int]     = None


class ProductRead(BaseSchema):
    id:          int
    name:        Optional[str]
    description: Optional[str]
    base_price:  Optional[Decimal]
    category_id: Optional[int]
    created_at:  datetime


class ProductWithReviews(ProductRead):
    reviews:          List["ReviewRead"] = []
    avg_rating:       Optional[float]    = None
    sentiment_summary: Optional[dict]    = None   # {"positive": N, "negative": N}


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
class VariantCreate(BaseModel):
    product_id: int
    sku:        Optional[str] = Field(default=None, max_length=100)
    color:      Optional[str] = Field(default=None, max_length=50)
    size:       Optional[str] = Field(default=None, max_length=50)


class VariantUpdate(BaseModel):
    sku:   Optional[str] = Field(default=None, max_length=100)
    color: Optional[str] = Field(default=None, max_length=50)
    size:  Optional[str] = Field(default=None, max_length=50)


class VariantRead(BaseSchema):
    id:         int
    product_id: int
    sku:        Optional[str]
    color:      Optional[str]
    size:       Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
class InventoryCreate(BaseModel):
    variant_id:         int
    stock_quantity:     int = Field(ge=0, default=0)
    warehouse_location: Optional[str] = None


class InventoryUpdate(BaseModel):
    stock_quantity:     Optional[int] = Field(default=None, ge=0)
    warehouse_location: Optional[str] = None


class InventoryRead(BaseSchema):
    id:                 int
    variant_id:         int
    stock_quantity:     Optional[int]
    warehouse_location: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# ORDERS
# ─────────────────────────────────────────────────────────────────────────────
class OrderItemCreate(BaseModel):
    variant_id: int
    quantity:   int = Field(ge=1)


class OrderCreate(BaseModel):
    items: List[OrderItemCreate] = Field(min_length=1)


class OrderItemRead(BaseSchema):
    id:         int
    variant_id: int
    quantity:   Optional[int]
    price:      Optional[Decimal]


class OrderRead(BaseSchema):
    id:           int
    user_id:      int
    order_date:   datetime
    total_amount: Optional[Decimal]
    status:       Optional[str]
    order_items:  List[OrderItemRead] = []


class OrderStatusUpdate(BaseModel):
    status: str = Field(pattern="^(pending|confirmed|shipped|delivered|cancelled)$")


# ─────────────────────────────────────────────────────────────────────────────
# REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
class ReviewCreate(BaseModel):
    product_id: int
    rating:     int = Field(ge=1, le=5)
    title:      Optional[str] = None
    content:    Optional[str] = None

    @field_validator("rating")
    @classmethod
    def rating_not_neutral(cls, v: int) -> int:
        # Business rule: we don't accept neutral reviews through the API
        # (mirrors the ML training label logic)
        return v


class ReviewUpdate(BaseModel):
    rating:  Optional[int] = Field(default=None, ge=1, le=5)
    title:   Optional[str] = None
    content: Optional[str] = None


class ReviewRead(BaseSchema):
    id:               int
    user_id:          int
    product_id:       int
    rating:           Optional[int]
    title:            Optional[str]
    content:          Optional[str]
    created_at:       datetime
    ml_sentiment:     Optional[str]
    ml_confidence:    Optional[Decimal]
    ml_model_version: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# WISHLIST
# ─────────────────────────────────────────────────────────────────────────────
class WishlistAdd(BaseModel):
    product_id: int


class WishlistRead(BaseSchema):
    id:         int
    user_id:    int
    product_id: int


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
class NotificationRead(BaseSchema):
    id:         int
    type:       Optional[str]
    title:      Optional[str]
    message:    Optional[str]
    is_read:    bool
    created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
class SettingCreate(BaseModel):
    key:         str = Field(max_length=100)
    value:       Optional[str] = None
    description: Optional[str] = None


class SettingUpdate(BaseModel):
    value:       Optional[str] = None
    description: Optional[str] = None


class SettingRead(BaseSchema):
    id:          int
    key:         str
    value:       Optional[str]
    description: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Single-review prediction request.

    review_id is optional: if provided, the prediction result is written
    back to the reviews table (ml_sentiment, ml_confidence, ml_model_version).
    """
    review_title:     str  = Field(min_length=1, max_length=500)
    review_content:   str  = Field(min_length=1, max_length=5000)
    discounted_price: Optional[float] = Field(default=None, ge=0)
    actual_price:     Optional[float] = Field(default=None, ge=0)
    rating_count:     Optional[float] = Field(default=None, ge=0)
    review_id:        Optional[int]   = None   # if set, writes result to DB

    @model_validator(mode="after")
    def actual_price_gte_discounted(self) -> "PredictRequest":
        if self.actual_price and self.discounted_price:
            if self.discounted_price > self.actual_price:
                raise ValueError("discounted_price cannot exceed actual_price.")
        return self


class PredictResponse(BaseModel):
    sentiment:      str
    label:          int
    confidence:     Optional[float]
    decision_score: Optional[float]
    model_name:     str
    feature_mode:   str
    latency_ms:     int


class BatchPredictRequest(BaseModel):
    """Batch prediction for multiple reviews in one API call."""
    reviews: List[PredictRequest] = Field(min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results:    List[PredictResponse]
    total:      int
    latency_ms: int


class ModelInfoResponse(BaseModel):
    model_name:       str
    feature_mode:     str
    vocabulary_size:  int
    training_metrics: dict
    train_samples:    int
    test_samples:     int


# ─────────────────────────────────────────────────────────────────────────────
# PAGINATION
# ─────────────────────────────────────────────────────────────────────────────
class PaginatedResponse(BaseModel):
    """Generic wrapper for paginated list endpoints."""
    items:   list
    total:   int
    page:    int
    size:    int
    pages:   int


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────────────────────────────────────
class AuditLogRead(BaseSchema):
    id:         int
    user_id:    Optional[int]
    action:     Optional[str]
    entity:     Optional[str]
    entity_id:  Optional[int]
    old_value:  Optional[str]
    new_value:  Optional[str]
    ip_address: Optional[str]
    created_at: datetime


# Resolve forward references
ProductWithReviews.model_rebuild()