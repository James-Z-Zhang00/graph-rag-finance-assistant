import time
import json
from typing import Any, Dict, Optional


class CacheItem:
    """Cache item wrapper supporting metadata and serialization."""

    def __init__(self, content: Any, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = self._initialize_metadata(metadata)

    def _initialize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize metadata with required default fields."""
        meta = metadata or {}

        defaults = {
            "created_at": time.time(),
            "quality_score": 0,
            "user_verified": False,
            "access_count": 0,
            "fast_path_eligible": False,
            "last_accessed": None,
            "similarity_score": None,
            "matched_via_vector": False,
            "original_query": None
        }

        for key, default_value in defaults.items():
            if key not in meta:
                meta[key] = default_value

        return meta

    def get_content(self) -> Any:
        """Return the cached content."""
        return self.content

    def is_high_quality(self) -> bool:
        """Return True if this item is considered high quality."""
        return (self.metadata.get("user_verified", False) or
                self.metadata.get("quality_score", 0) > 2 or
                self.metadata.get("fast_path_eligible", False))

    def mark_quality(self, is_positive: bool) -> None:
        """Update the quality score based on user feedback."""
        if is_positive:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = current_score + 1
            self.metadata["user_verified"] = True
            self.metadata["fast_path_eligible"] = True
        else:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = max(-5, current_score - 2)  # floor at -5
            self.metadata["fast_path_eligible"] = False

    def update_access_stats(self) -> None:
        """Increment access count and update last accessed timestamp."""
        self.metadata["access_count"] = self.metadata.get("access_count", 0) + 1
        self.metadata["last_accessed"] = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict."""
        return {
            "content": self.content,
            "metadata": self.metadata
        }

    def to_json(self, ensure_ascii: bool = False) -> str:
        """Serialize to a JSON string."""
        try:
            return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, default=str)
        except (TypeError, ValueError) as e:
            return json.dumps({
                "content": f"Serialization failed: {str(e)}",
                "metadata": self.metadata
            })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheItem':
        """Deserialize from a dict."""
        try:
            if isinstance(data, dict):
                if "content" in data and "metadata" in data:
                    metadata = data["metadata"]
                    if not isinstance(metadata, dict):
                        metadata = {}
                    return cls(data["content"], metadata)
                else:
                    return cls(data)
            else:
                return cls(data)
        except Exception as e:
            return cls(f"Error deserializing cache item: {str(e)}", {
                "created_at": time.time(),
                "quality_score": -10,  # mark as low quality
                "user_verified": False,
                "access_count": 0,
                "error": str(e)
            })

    @classmethod
    def from_json(cls, json_str: str) -> 'CacheItem':
        """Deserialize from a JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            return cls(f"JSON decode error: {str(e)}", {
                "created_at": time.time(),
                "quality_score": -10,
                "user_verified": False,
                "access_count": 0,
                "error": str(e)
            })

    @classmethod
    def from_any(cls, data: Any) -> 'CacheItem':
        """Create a CacheItem from any data type with automatic type detection."""
        if isinstance(data, cls):
            return data
        elif isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                return cls.from_dict(parsed_data)
            except json.JSONDecodeError:
                return cls(data)
        elif isinstance(data, dict) and "content" in data:
            return cls.from_dict(data)
        else:
            return cls(data)

    def get_age(self) -> float:
        """Return the age of this cache item in seconds."""
        created_at = self.metadata.get("created_at", time.time())
        return time.time() - created_at

    def is_expired(self, max_age: float) -> bool:
        """Return True if this item is older than max_age seconds."""
        return self.get_age() > max_age

    def __repr__(self) -> str:
        content_preview = str(self.content)[:50]
        if len(str(self.content)) > 50:
            content_preview += "..."
        return (f"CacheItem(content='{content_preview}', "
                f"quality_score={self.metadata.get('quality_score', 0)}, "
                f"access_count={self.metadata.get('access_count', 0)})")
