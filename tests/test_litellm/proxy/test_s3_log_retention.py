import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.constants import (
    S3_LOG_RETENTION_DELETE_BATCH_SIZE,
    S3_LOG_RETENTION_JOB_NAME,
)
from litellm.proxy.logging.s3_log_retention import S3LogRetentionJob


def test_s3_log_retention_runs_with_lock():
    general_settings = {"s3_log_retention_period": "1h"}

    mock_pod_lock_manager = MagicMock()
    mock_pod_lock_manager.redis_cache = object()
    mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    mock_pod_lock_manager.release_lock = AsyncMock()

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.db_spend_update_writer.pod_lock_manager = mock_pod_lock_manager

    mock_s3_logger = MagicMock()
    mock_s3_logger.async_cleanup_old_logs = AsyncMock(return_value=4)

    with patch(
        "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name",
        return_value=mock_s3_logger,
    ):
        job = S3LogRetentionJob(
            general_settings=general_settings,
            proxy_logging_obj=mock_proxy_logging,
        )
        asyncio.run(job.run())

    mock_pod_lock_manager.acquire_lock.assert_awaited_once_with(
        cronjob_id=S3_LOG_RETENTION_JOB_NAME
    )
    mock_s3_logger.async_cleanup_old_logs.assert_awaited_once()
    cleanup_kwargs = mock_s3_logger.async_cleanup_old_logs.await_args.kwargs
    assert cleanup_kwargs["retention_seconds"] == 3600
    assert cleanup_kwargs["max_objects"] == S3_LOG_RETENTION_DELETE_BATCH_SIZE
    mock_pod_lock_manager.release_lock.assert_awaited_once_with(
        cronjob_id=S3_LOG_RETENTION_JOB_NAME
    )


def test_s3_log_retention_respects_max_delete():
    general_settings = {
        "s3_log_retention_period": "1h",
        "s3_log_retention_max_delete": 5,
    }

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.db_spend_update_writer.pod_lock_manager = MagicMock()
    mock_proxy_logging.db_spend_update_writer.pod_lock_manager.redis_cache = object()
    mock_proxy_logging.db_spend_update_writer.pod_lock_manager.acquire_lock = AsyncMock(
        return_value=True
    )
    mock_proxy_logging.db_spend_update_writer.pod_lock_manager.release_lock = AsyncMock()

    mock_s3_logger = MagicMock()
    mock_s3_logger.async_cleanup_old_logs = AsyncMock(return_value=1)

    with patch(
        "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name",
        return_value=mock_s3_logger,
    ):
        job = S3LogRetentionJob(
            general_settings=general_settings,
            proxy_logging_obj=mock_proxy_logging,
        )
        asyncio.run(job.run())

    cleanup_kwargs = mock_s3_logger.async_cleanup_old_logs.await_args.kwargs
    assert cleanup_kwargs["max_objects"] == 5


def test_s3_log_retention_skips_when_lock_unavailable():
    general_settings = {"s3_log_retention_period": "1h"}

    mock_pod_lock_manager = MagicMock()
    mock_pod_lock_manager.redis_cache = object()
    mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=False)
    mock_pod_lock_manager.release_lock = AsyncMock()

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.db_spend_update_writer.pod_lock_manager = mock_pod_lock_manager

    mock_s3_logger = MagicMock()
    mock_s3_logger.async_cleanup_old_logs = AsyncMock()

    with patch(
        "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name",
        return_value=mock_s3_logger,
    ):
        job = S3LogRetentionJob(
            general_settings=general_settings,
            proxy_logging_obj=mock_proxy_logging,
        )
        asyncio.run(job.run())

    mock_s3_logger.async_cleanup_old_logs.assert_not_called()
    mock_pod_lock_manager.release_lock.assert_not_called()


def test_s3_log_retention_runs_without_redis_lock():
    general_settings = {"s3_log_retention_period": "1h"}

    mock_pod_lock_manager = MagicMock()
    mock_pod_lock_manager.redis_cache = None
    mock_pod_lock_manager.acquire_lock = AsyncMock()
    mock_pod_lock_manager.release_lock = AsyncMock()

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.db_spend_update_writer.pod_lock_manager = mock_pod_lock_manager

    mock_s3_logger = MagicMock()
    mock_s3_logger.async_cleanup_old_logs = AsyncMock(return_value=2)

    with patch(
        "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name",
        return_value=mock_s3_logger,
    ):
        job = S3LogRetentionJob(
            general_settings=general_settings,
            proxy_logging_obj=mock_proxy_logging,
        )
        asyncio.run(job.run())

    mock_pod_lock_manager.acquire_lock.assert_not_called()
    mock_s3_logger.async_cleanup_old_logs.assert_awaited_once()


def test_s3_log_retention_invalid_period():
    general_settings = {"s3_log_retention_period": "invalid"}

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.db_spend_update_writer.pod_lock_manager = MagicMock()

    with patch(
        "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name",
    ) as mock_get_logger:
        job = S3LogRetentionJob(
            general_settings=general_settings,
            proxy_logging_obj=mock_proxy_logging,
        )
        asyncio.run(job.run())

    mock_get_logger.assert_not_called()
