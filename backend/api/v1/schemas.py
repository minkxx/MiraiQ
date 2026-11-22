from __future__ import annotations

from pydantic import BaseModel, Field


class SyncRequest(BaseModel):
    force: bool = False


class CatalogRequest(BaseModel):
    limit: int = Field(default=2000, ge=1, le=10000)


class TrainRequest(BaseModel):
    model_name: str = "gbr"
    min_ratings: int = Field(default=5, ge=1)
    tag_threshold: int = Field(default=50, ge=0, le=100)
    min_studio_count: int = Field(default=3, ge=1)
    include_format: bool = False
    include_source: bool = False
    include_metadata: bool = True


class EvaluateRequest(BaseModel):
    models: list[str] | None = None
    test_size: float = Field(default=0.2, gt=0, lt=1)
    n_splits: int = Field(default=5, ge=2, le=10)
    temporal: bool = False


class RecommendRequest(BaseModel):
    model_name: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    sort_by: str = Field(default="predicted")
    min_tag_rank: int = Field(default=50, ge=0, le=100)
    include_planning: bool = True
    show_sequels: bool = False
    include_genres: list[str] | None = None
    exclude_genres: list[str] | None = None
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
