from redis import asyncio as aioredis
import logging
import os
from typing import Optional
from contextlib import asynccontextmanager
import pendulum


class RedisManager:
    """
    Manages Redis connections and operations.
    """

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._host = os.getenv("REDIS_HOST", "redis.semoss.svc.cluster.local")
        # self._host = "127.0.0.1"
        self._port = int(os.getenv("REDIS_PORT", "6379"))
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> None:
        """
        Establishes connection to Redis if not already connected.
        Raises:
            ConnectionError: If connection to Redis fails
        """
        if self._redis is not None:
            self.logger.warning("Redis connection already exists")
            return

        try:
            self.logger.info(f"Connecting to Redis at {self._host}:{self._port}")
            self._redis = await aioredis.from_url(
                f"redis://{self._host}:{self._port}",
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            await self._redis.ping()
            self.logger.info("Successfully connected to Redis")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {str(e)}")
            self._redis = None
            raise ConnectionError(f"Could not connect to Redis: {str(e)}")

    async def disconnect(self) -> None:
        """
        Closes the Redis connection if it exists.
        """
        if self._redis is not None:
            try:
                self.logger.info("Closing Redis connection")
                await self._redis.aclose()
                self.logger.info("Redis connection closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing Redis connection: {str(e)}")
            finally:
                self._redis = None

    @property
    def is_connected(self) -> bool:
        """
        Checks if Redis client exists and is connected.
        Returns:
            bool: True if connected, False otherwise
        """
        return self._redis is not None and self._redis.connection is not None

    @asynccontextmanager
    async def get_connection(self):
        """
        Context manager for getting a Redis connection.
        Ensures connection is established before yielding.
        Yields:
            aioredis.Redis: Active Redis connection
        Raises:
            ConnectionError: If Redis is not connected and connection attempt fails
        """
        if not self.is_connected:
            await self.connect()

        try:
            yield self._redis
        except Exception as e:
            self.logger.error(f"Error during Redis operation: {str(e)}")
            raise

    async def get_deployment_status(self, model_id: str):
        deployment_key = f"{model_id}:deployment"
        status = await self._redis.get(deployment_key)
        return {k.decode(): v.decode() for k, v in status.items()}

    async def update_deployment_status(self, model_id: str):
        """
        Updates the deployment status by setting the last_request time
        and incrementing the generations counter.
        Args:
            model_id (str): The ID of the model deployment to update
        Raises:
            ConnectionError: If Redis connection fails
            ValueError: If deployment status doesn't exist
        """
        deployment_key = f"{model_id}:deployment"
        current_time = pendulum.now("America/New_York").isoformat()

        try:
            async with self.get_connection() as redis:
                if not await redis.exists(deployment_key):
                    raise ValueError(f"No deployment status found for {model_id}")

                async with redis.pipeline() as pipe:
                    await pipe.hset(deployment_key, "last_request", current_time)
                    await pipe.hincrby(deployment_key, "generations", 1)
                    await pipe.execute()

                self.logger.info(
                    f"Successfully updated deployment status for {model_id}"
                )

        except ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error updating deployment status in Redis: {e}")
            raise
