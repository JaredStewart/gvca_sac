"""Shared pagination models for API endpoints."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Common pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    items: list[T]
    page: int
    per_page: int = Field(alias="perPage")
    total_items: int = Field(alias="totalItems")
    total_pages: int = Field(alias="totalPages")

    class Config:
        populate_by_name = True

    @classmethod
    def create(
        cls,
        items: list[T],
        page: int,
        per_page: int,
        total_items: int,
    ) -> "PaginatedResponse[T]":
        """Create a paginated response with computed total pages."""
        total_pages = (total_items + per_page - 1) // per_page if per_page > 0 else 0
        return cls(
            items=items,
            page=page,
            per_page=per_page,
            total_items=total_items,
            total_pages=total_pages,
        )
