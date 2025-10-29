"""Background job for cleaning up S3 logs in multi-instance deployments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    S3_LOG_RETENTION_DELETE_BATCH_SIZE,
    S3_LOG_RETENTION_JOB_NAME,
)
from litellm.litellm_core_utils.duration_parser import duration_in_seconds

if TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging
else:
    ProxyLogging = Any


class S3LogRetentionJob:
    """Coordinates S3 log retention across multiple proxy instances."""

    def __init__(
        self,
        general_settings: Optional[dict] = None,
        proxy_logging_obj: Optional[ProxyLogging] = None,
    ) -> None:
        if general_settings is None or proxy_logging_obj is None:
            try:
                from litellm.proxy.proxy_server import (
                    general_settings as global_general_settings,
                )
                from litellm.proxy.proxy_server import (
                    proxy_logging_obj as global_proxy_logging_obj,
                )
            except Exception:
                global_general_settings = {}
                global_proxy_logging_obj = None

            general_settings = general_settings or global_general_settings
            proxy_logging_obj = proxy_logging_obj or global_proxy_logging_obj

        self.general_settings: dict = general_settings or {}
        self.proxy_logging_obj: Optional[ProxyLogging] = proxy_logging_obj
        self.pod_lock_manager = None
        if (
            self.proxy_logging_obj is not None
            and getattr(self.proxy_logging_obj, "db_spend_update_writer", None)
            is not None
        ):
            self.pod_lock_manager = (
                self.proxy_logging_obj.db_spend_update_writer.pod_lock_manager
            )

        self.retention_seconds: Optional[int] = None
        self.max_delete_per_run: Optional[int] = None

    def _parse_configuration(self) -> bool:
        retention_setting = self.general_settings.get("s3_log_retention_period")
        if retention_setting is None:
            verbose_proxy_logger.debug(
                "Skipping S3 log retention - no s3_log_retention_period configured"
            )
            return False

        try:
            if isinstance(retention_setting, int):
                retention_setting = str(retention_setting)
            parsed_seconds = duration_in_seconds(retention_setting)
            self.retention_seconds = int(parsed_seconds)
        except (TypeError, ValueError) as exc:
            verbose_proxy_logger.error(
                "Invalid s3_log_retention_period value: %s, error: %s",
                retention_setting,
                exc,
            )
            return False

        max_delete_setting = self.general_settings.get("s3_log_retention_max_delete")
        if max_delete_setting is None:
            self.max_delete_per_run = S3_LOG_RETENTION_DELETE_BATCH_SIZE
        else:
            try:
                parsed_max = int(max_delete_setting)
            except (TypeError, ValueError):
                verbose_proxy_logger.error(
                    "Invalid s3_log_retention_max_delete value: %s",
                    max_delete_setting,
                )
                self.max_delete_per_run = S3_LOG_RETENTION_DELETE_BATCH_SIZE
            else:
                if parsed_max <= 0:
                    verbose_proxy_logger.info(
                        "s3_log_retention_max_delete is non-positive; skipping retention"
                    )
                    return False
                self.max_delete_per_run = parsed_max

        return True

    def _get_s3_logger(self) -> Optional[Any]:
        for callback_name in ("s3_v2", "s3"):
            try:
                logger = (
                    litellm.logging_callback_manager.get_active_custom_logger_for_callback_name(  # type: ignore[arg-type]
                        callback_name
                    )
                )
            except ValueError:
                continue
            except Exception as exc:  # pragma: no cover - unexpected errors
                verbose_proxy_logger.exception(
                    "Error retrieving %s logger: %s", callback_name, exc
                )
                continue

            if logger is not None:
                return logger

        verbose_proxy_logger.info("No active S3 logger found; skipping retention")
        return None

    async def run(self) -> None:
        if not self._parse_configuration():
            return

        s3_logger = self._get_s3_logger()
        if s3_logger is None:
            return

        cleanup_callable = getattr(s3_logger, "async_cleanup_old_logs", None)
        if cleanup_callable is None or not callable(cleanup_callable):
            verbose_proxy_logger.info(
                "Active S3 logger does not support retention cleanup; skipping"
            )
            return

        lock_acquired = False
        try:
            if (
                self.pod_lock_manager is not None
                and getattr(self.pod_lock_manager, "redis_cache", None) is not None
            ):
                lock_result = await self.pod_lock_manager.acquire_lock(
                    cronjob_id=S3_LOG_RETENTION_JOB_NAME,
                )
                lock_acquired = bool(lock_result)
                if not lock_acquired:
                    verbose_proxy_logger.info(
                        "Another pod is already running S3 log retention"
                    )
                    return

            deleted_count = await cleanup_callable(  # type: ignore[misc]
                retention_seconds=int(self.retention_seconds or 0),
                max_objects=self.max_delete_per_run,
            )
            verbose_proxy_logger.info(
                "S3 log retention deleted %s objects", deleted_count
            )

        except Exception as exc:  # pragma: no cover - unexpected errors
            verbose_proxy_logger.exception(
                "Error executing S3 log retention job: %s", exc
            )
        finally:
            if (
                lock_acquired
                and self.pod_lock_manager is not None
                and getattr(self.pod_lock_manager, "redis_cache", None) is not None
            ):
                try:
                    await self.pod_lock_manager.release_lock(
                        cronjob_id=S3_LOG_RETENTION_JOB_NAME
                    )
                    verbose_proxy_logger.info("Released S3 log retention lock")
                except Exception as exc:  # pragma: no cover - unexpected errors
                    verbose_proxy_logger.exception(
                        "Error releasing S3 log retention lock: %s", exc
                    )
