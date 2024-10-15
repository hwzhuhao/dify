from typing import Optional

from pydantic import BaseModel, Field


class UpstashVectorConfig(BaseModel):
    """
    Configuration settings for Upstash Vector Service
    """

    UPSTASH_VECTOR_REST_URL: str = Field(
        description="URL of the Upstash Vector Database",
        default=None,
    )
    UPSTASH_VECTOR_REST_TOKEN: Optional[str] = Field(
        description="Token for authenticating with the Upstash Vector database",
        default=None,
    )
