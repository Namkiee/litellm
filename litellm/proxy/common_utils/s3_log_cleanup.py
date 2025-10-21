"""Utilities for cleaning up S3-based proxy logs."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import boto3
import litellm
from botocore.client import BaseClient

from litellm._logging import verbose_proxy_logger
from litellm.constants import S3_LOG_CLEANUP_JOB_NAME
from litellm.litellm_core_utils.duration_parser import duration_in_seconds


_MAX_DELETE_BATCH = 1000


class S3LogCleanup:
    """Cleanup job for removing aged S3 log objects."""

    def __init__(
        self,
        general_settings: Optional[Dict[str, Any]] = None,
        pod_lock_manager: Optional[Any] = None,
    ) -> None:
        resolved_settings = general_settings
        resolved_pod_lock_manager = pod_lock_manager

        if resolved_settings is None or resolved_pod_lock_manager is None:
            from litellm.proxy.proxy_server import general_settings as default_settings
            from litellm.proxy.proxy_server import proxy_logging_obj

            if resolved_settings is None:
                resolved_settings = default_settings
            if resolved_pod_lock_manager is None:
                resolved_pod_lock_manager = (
                    proxy_logging_obj.db_spend_update_writer.pod_lock_manager
                )

        self.general_settings = resolved_settings
        self.retention_seconds: Optional[int] = None
        self.bucket_name: Optional[str] = None
        self.s3_path: Optional[str] = None
        self.pod_lock_manager = resolved_pod_lock_manager

    def _should_delete_s3_logs(self) -> bool:
        retention_setting = self.general_settings.get("s3_logs_retention_period")
        verbose_proxy_logger.info(
            f"Checking S3 retention setting: {retention_setting}"
        )

        if retention_setting is None:
            verbose_proxy_logger.info("No S3 retention setting found")
            return False

        try:
            if isinstance(retention_setting, int):
                retention_setting = str(retention_setting)
            self.retention_seconds = duration_in_seconds(retention_setting)
            verbose_proxy_logger.info(
                f"S3 retention period set to {self.retention_seconds} seconds"
            )
            return True
        except ValueError as exc:
            verbose_proxy_logger.error(
                "Invalid s3_logs_retention_period value: %s, error: %s",
                retention_setting,
                str(exc),
            )
            return False

    def _load_s3_params(self) -> Optional[Dict[str, Any]]:
        s3_params = dict(litellm.s3_callback_params or {})
        if not s3_params:
            verbose_proxy_logger.info(
                "Skipping S3 cleanup — no s3_callback_params configured"
            )
            return None

        for key, value in s3_params.items():
            if isinstance(value, str) and value.startswith("os.environ/"):
                s3_params[key] = litellm.get_secret(value)

        bucket_name = s3_params.get("s3_bucket_name")
        if bucket_name is None:
            verbose_proxy_logger.info(
                "Skipping S3 cleanup — missing s3_bucket_name in configuration"
            )
            return None

        self.bucket_name = bucket_name
        self.s3_path = s3_params.get("s3_path")
        return s3_params

    def _get_s3_client(self, s3_params: Dict[str, Any]) -> BaseClient:
        return boto3.client(
            "s3",
            region_name=s3_params.get("s3_region_name"),
            endpoint_url=s3_params.get("s3_endpoint_url"),
            api_version=s3_params.get("s3_api_version"),
            use_ssl=s3_params.get("s3_use_ssl", True),
            verify=s3_params.get("s3_verify"),
            aws_access_key_id=s3_params.get("s3_aws_access_key_id"),
            aws_secret_access_key=s3_params.get("s3_aws_secret_access_key"),
            aws_session_token=s3_params.get("s3_aws_session_token"),
            config=s3_params.get("s3_config"),
        )

    def _prefix(self) -> Optional[str]:
        if not self.s3_path:
            return None
        return self.s3_path.rstrip("/") + "/"

    def _should_delete_object(self, last_modified: datetime, cutoff: datetime) -> bool:
        if last_modified.tzinfo is None:
            last_modified = last_modified.replace(tzinfo=timezone.utc)
        return last_modified <= cutoff

    def _batched(self, items: Iterable[Dict[str, str]], size: int) -> Iterable[List[Dict[str, str]]]:
        batch: List[Dict[str, str]] = []
        for item in items:
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _delete_old_logs_sync(self, client: BaseClient, cutoff: datetime) -> int:
        total_deleted = 0
        prefix = self._prefix()
        continuation_token: Optional[str] = None

        while True:
            list_kwargs: Dict[str, Any] = {"Bucket": self.bucket_name}
            if prefix:
                list_kwargs["Prefix"] = prefix
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = client.list_objects_v2(**list_kwargs)
            contents = response.get("Contents", []) or []

            keys_to_delete: List[Dict[str, str]] = []
            for obj in contents:
                key = obj.get("Key")
                last_modified = obj.get("LastModified")
                if key is None or not isinstance(last_modified, datetime):
                    continue
                if self._should_delete_object(last_modified, cutoff):
                    keys_to_delete.append({"Key": key})

            for batch in self._batched(keys_to_delete, _MAX_DELETE_BATCH):
                client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={"Objects": batch, "Quiet": True},
                )
                total_deleted += len(batch)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

        return total_deleted

    async def cleanup_old_s3_logs(self) -> None:
        verbose_proxy_logger.info(
            "S3 log cleanup job triggered at %s", datetime.now(timezone.utc)
        )

        if not self._should_delete_s3_logs():
            verbose_proxy_logger.info(
                "Skipping S3 cleanup — invalid or missing retention configuration"
            )
            return

        if self.retention_seconds is None:
            verbose_proxy_logger.error(
                "S3 log cleanup aborted — retention_seconds is not set"
            )
            return

        s3_params = self._load_s3_params()
        if s3_params is None:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=float(self.retention_seconds)
        )

        verbose_proxy_logger.info(
            "Deleting S3 log objects older than %s", cutoff.isoformat()
        )

        if self.pod_lock_manager and self.pod_lock_manager.redis_cache:
            lock_acquired = await self.pod_lock_manager.acquire_lock(
                cronjob_id=S3_LOG_CLEANUP_JOB_NAME
            )
            if not lock_acquired:
                verbose_proxy_logger.info(
                    "Another pod is already running S3 cleanup"
                )
                return

        try:
            client = self._get_s3_client(s3_params)
            deleted = await asyncio.to_thread(
                self._delete_old_logs_sync, client, cutoff
            )
            verbose_proxy_logger.info(
                "Deleted %s S3 log objects from bucket %s",
                deleted,
                self.bucket_name,
            )
        except Exception as exc:  # noqa: BLE001
            verbose_proxy_logger.error(
                "Error during S3 log cleanup: %s", str(exc)
            )
        finally:
            if self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                await self.pod_lock_manager.release_lock(
                    cronjob_id=S3_LOG_CLEANUP_JOB_NAME
                )
                verbose_proxy_logger.info("Released S3 cleanup lock")
