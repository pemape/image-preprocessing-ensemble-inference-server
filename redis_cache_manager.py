"""
Redis Cache Manager for Fundus Inference Server

Provides optional caching of inference results using Redis.
Falls back gracefully when Redis is not available.

Cache Key Format: {image_name}:{sha256_hash}
Cache Value: JSON-serialized inference result
"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any
import numpy as np
import os

# Try to import Redis, but make it optional
try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisError = Exception
    RedisConnectionError = Exception


class RedisCacheManager:
    """
    Manages caching of inference results in Redis.
    Gracefully handles Redis unavailability.
    """

    def __init__(
        self,
        enabled: bool = False,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 86400,  # 24 hours default
        key_prefix: str = "fundus_inference",
    ):
        """
        Initialize Redis cache manager.

        Args:
            enabled: Whether caching is enabled
            host: Redis host (can be overridden by REDIS_HOST env var)
            port: Redis port (can be overridden by REDIS_PORT env var)
            db: Redis database number
            password: Redis password (optional, can be overridden by REDIS_PASSWORD env var)
            ttl: Time-to-live for cached entries in seconds (default: 24 hours)
            key_prefix: Prefix for all cache keys
        """
        # Allow environment variables to override defaults
        self.enabled = enabled
        self.host = os.getenv("REDIS_HOST", host)
        self.port = int(os.getenv("REDIS_PORT", port))
        self.db = db
        self.password = os.getenv("REDIS_PASSWORD", password)
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.logger = logging.getLogger(self.__class__.__name__)

        self.redis_client = None
        self.cache_available = False

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
        }

        if self.enabled:
            self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            self.logger.warning(
                "Redis caching is enabled but redis package is not installed. "
                "Install with: pip install redis"
            )
            self.enabled = False
            return

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )

            # Test connection
            self.redis_client.ping()
            self.cache_available = True
            self.logger.info(
                f"Redis cache initialized successfully at {self.host}:{self.port} "
                f"(DB: {self.db}, TTL: {self.ttl}s)"
            )

        except (RedisConnectionError, RedisError) as e:
            self.logger.warning(
                f"Redis connection failed: {e}. Caching disabled, "
                "server will continue without cache."
            )
            self.redis_client = None
            self.cache_available = False
        except Exception as e:
            self.logger.error(f"Unexpected error initializing Redis: {e}")
            self.redis_client = None
            self.cache_available = False

    def _generate_cache_key(
        self,
        image_name: str,
        image_data: np.ndarray,
        voting_strategy: str = "soft",
        model_architecture: str = "unknown",
        ensemble_size: int = 1,
    ) -> str:
        """
        Generate cache key from image name, content, and model configuration.

        Format: {prefix}:{image_name}:{sha256_hash}:{voting}:{arch}:{ensemble}

        Args:
            image_name: Original filename
            image_data: Image numpy array
            voting_strategy: Voting strategy (soft/hard)
            model_architecture: Model architecture name
            ensemble_size: Number of models in ensemble

        Returns:
            Cache key string
        """
        # Convert image to bytes for hashing
        image_bytes = image_data.tobytes()

        # Compute SHA256 hash
        sha256_hash = hashlib.sha256(image_bytes).hexdigest()

        # Create cache key with model configuration parameters
        cache_key = (
            f"{self.key_prefix}:{image_name}:{sha256_hash}:"
            f"{voting_strategy}:{model_architecture}:{ensemble_size}"
        )

        return cache_key

    def get(
        self,
        image_name: str,
        image_data: np.ndarray,
        voting_strategy: str = "soft",
        model_architecture: str = "unknown",
        ensemble_size: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result.

        Args:
            image_name: Original filename
            image_data: Image numpy array
            voting_strategy: Voting strategy (soft/hard)
            model_architecture: Model architecture name
            ensemble_size: Number of models in ensemble

        Returns:
            Cached result dictionary or None if not found
        """
        if not self.enabled or not self.cache_available:
            return None

        try:
            cache_key = self._generate_cache_key(
                image_name,
                image_data,
                voting_strategy,
                model_architecture,
                ensemble_size,
            )

            cached_value = self.redis_client.get(cache_key)

            if cached_value:
                self.stats["hits"] += 1
                self.logger.debug(f"Cache HIT for key: {cache_key}")
                return json.loads(cached_value)
            else:
                self.stats["misses"] += 1
                self.logger.debug(f"Cache MISS for key: {cache_key}")
                return None

        except (RedisError, RedisConnectionError) as e:
            self.stats["errors"] += 1
            self.logger.warning(f"Redis get error: {e}")
            return None
        except json.JSONDecodeError as e:
            self.stats["errors"] += 1
            self.logger.error(f"Failed to decode cached value: {e}")
            return None
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Unexpected error getting from cache: {e}")
            return None

    def set(
        self,
        image_name: str,
        image_data: np.ndarray,
        result: Dict[str, Any],
        voting_strategy: str = "soft",
        model_architecture: str = "unknown",
        ensemble_size: int = 1,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store result in cache.

        Args:
            image_name: Original filename
            image_data: Image numpy array
            result: Result dictionary to cache
            voting_strategy: Voting strategy (soft/hard)
            model_architecture: Model architecture name
            ensemble_size: Number of models in ensemble
            ttl: Time-to-live override (optional)

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self.cache_available:
            return False

        try:
            cache_key = self._generate_cache_key(
                image_name,
                image_data,
                voting_strategy,
                model_architecture,
                ensemble_size,
            )
            cache_value = json.dumps(result)

            # Use provided TTL or default
            expiration = ttl if ttl is not None else self.ttl

            # Set with expiration
            self.redis_client.setex(cache_key, expiration, cache_value)

            self.stats["sets"] += 1
            self.logger.debug(f"Cache SET for key: {cache_key} (TTL: {expiration}s)")
            return True

        except (RedisError, RedisConnectionError) as e:
            self.stats["errors"] += 1
            self.logger.warning(f"Redis set error: {e}")
            return False
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Unexpected error setting cache: {e}")
            return False

    def delete(
        self,
        image_name: str,
        image_data: np.ndarray,
        voting_strategy: str = "soft",
        model_architecture: str = "unknown",
        ensemble_size: int = 1,
    ) -> bool:
        """
        Delete cached result.

        Args:
            image_name: Original filename
            image_data: Image numpy array
            voting_strategy: Voting strategy (soft/hard)
            model_architecture: Model architecture name
            ensemble_size: Number of models in ensemble

        Returns:
            True if successfully deleted, False otherwise
        """
        if not self.enabled or not self.cache_available:
            return False

        try:
            cache_key = self._generate_cache_key(
                image_name,
                image_data,
                voting_strategy,
                model_architecture,
                ensemble_size,
            )
            deleted = self.redis_client.delete(cache_key)

            self.logger.debug(f"Cache DELETE for key: {cache_key}")
            return deleted > 0

        except (RedisError, RedisConnectionError) as e:
            self.stats["errors"] += 1
            self.logger.warning(f"Redis delete error: {e}")
            return False
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Unexpected error deleting from cache: {e}")
            return False

    def clear_all(self, pattern: Optional[str] = None) -> int:
        """
        Clear cached entries matching pattern.

        Args:
            pattern: Key pattern to match (default: all keys with prefix)

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.cache_available:
            return 0

        try:
            search_pattern = pattern or f"{self.key_prefix}:*"
            keys = list(self.redis_client.scan_iter(match=search_pattern))

            if keys:
                deleted = self.redis_client.delete(*keys)
                self.logger.info(
                    f"Cleared {deleted} cached entries matching '{search_pattern}'"
                )
                return deleted

            return 0

        except (RedisError, RedisConnectionError) as e:
            self.stats["errors"] += 1
            self.logger.warning(f"Redis clear error: {e}")
            return 0
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Unexpected error clearing cache: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "enabled": self.enabled,
            "available": self.cache_available,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "errors": self.stats["errors"],
            "hit_rate": (
                self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                if (self.stats["hits"] + self.stats["misses"]) > 0
                else 0.0
            ),
        }

        if self.cache_available and self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_version"] = info.get("redis_version", "unknown")
                stats["used_memory_human"] = info.get("used_memory_human", "unknown")
                stats["connected_clients"] = info.get("connected_clients", 0)
            except Exception as e:
                self.logger.debug(f"Could not get Redis info: {e}")

        return stats

    def health_check(self) -> Dict[str, Any]:
        """
        Check Redis connection health.

        Returns:
            Health status dictionary
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "Redis caching is disabled",
            }

        if not self.cache_available or not self.redis_client:
            return {
                "status": "unavailable",
                "message": "Redis connection is not available",
            }

        try:
            response_time_start = __import__("time").time()
            self.redis_client.ping()
            response_time = (__import__("time").time() - response_time_start) * 1000

            return {
                "status": "healthy",
                "message": "Redis connection is healthy",
                "response_time_ms": round(response_time, 2),
                "host": self.host,
                "port": self.port,
                "db": self.db,
            }
        except Exception as e:
            self.cache_available = False
            return {
                "status": "unhealthy",
                "message": f"Redis connection failed: {str(e)}",
            }
