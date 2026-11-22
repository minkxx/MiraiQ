from __future__ import annotations

import asyncio
from typing import Any

from miraiq.cache.cache import CacheManager


class BaseService:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def cm(self, username: str) -> CacheManager:
        return CacheManager(self.cache_dir)

    @staticmethod
    def run_async(coro) -> Any:
        return asyncio.run(coro)
