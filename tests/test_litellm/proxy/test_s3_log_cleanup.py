import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import litellm
from litellm.proxy.common_utils.s3_log_cleanup import S3LogCleanup


def test_should_delete_s3_logs():
    pod_lock = MagicMock()
    pod_lock.redis_cache = None

    cleaner = S3LogCleanup(general_settings={}, pod_lock_manager=pod_lock)
    assert cleaner._should_delete_s3_logs() is False

    cleaner = S3LogCleanup(
        general_settings={"s3_logs_retention_period": "3600s"},
        pod_lock_manager=pod_lock,
    )
    assert cleaner._should_delete_s3_logs() is True

    cleaner = S3LogCleanup(
        general_settings={"s3_logs_retention_period": "1d"},
        pod_lock_manager=pod_lock,
    )
    assert cleaner._should_delete_s3_logs() is True

    cleaner = S3LogCleanup(
        general_settings={"s3_logs_retention_period": "invalid"},
        pod_lock_manager=pod_lock,
    )
    assert cleaner._should_delete_s3_logs() is False


def test_cleanup_skips_without_s3_configuration(monkeypatch):
    original_params = litellm.s3_callback_params
    litellm.s3_callback_params = None

    pod_lock = MagicMock()
    pod_lock.redis_cache = None

    cleaner = S3LogCleanup(
        general_settings={"s3_logs_retention_period": "1h"},
        pod_lock_manager=pod_lock,
    )

    async def fake_to_thread(func, *args, **kwargs):
        raise AssertionError("to_thread should not be called")

    monkeypatch.setattr(
        "litellm.proxy.common_utils.s3_log_cleanup.asyncio.to_thread",
        fake_to_thread,
    )

    asyncio.run(cleaner.cleanup_old_s3_logs())

    litellm.s3_callback_params = original_params


def test_cleanup_deletes_old_objects(monkeypatch):
    old_timestamp = datetime.now(timezone.utc) - timedelta(days=10)
    fresh_timestamp = datetime.now(timezone.utc) - timedelta(hours=1)

    original_params = litellm.s3_callback_params
    litellm.s3_callback_params = {
        "s3_bucket_name": "test-bucket",
        "s3_path": "logs",
    }

    mock_client = MagicMock()
    mock_client.list_objects_v2.side_effect = [
        {
            "Contents": [
                {"Key": "logs/old.json", "LastModified": old_timestamp},
                {"Key": "logs/new.json", "LastModified": fresh_timestamp},
            ],
            "IsTruncated": False,
        }
    ]
    mock_client.delete_objects = MagicMock()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "litellm.proxy.common_utils.s3_log_cleanup.asyncio.to_thread",
        fake_to_thread,
    )
    monkeypatch.setattr(
        "litellm.proxy.common_utils.s3_log_cleanup.boto3.client",
        lambda *args, **kwargs: mock_client,
    )

    pod_lock = MagicMock()
    pod_lock.redis_cache = None

    cleaner = S3LogCleanup(
        general_settings={"s3_logs_retention_period": "7d"},
        pod_lock_manager=pod_lock,
    )

    asyncio.run(cleaner.cleanup_old_s3_logs())

    mock_client.delete_objects.assert_called_once()
    delete_kwargs = mock_client.delete_objects.call_args.kwargs
    assert delete_kwargs["Bucket"] == "test-bucket"
    assert delete_kwargs["Delete"]["Objects"] == [{"Key": "logs/old.json"}]

    litellm.s3_callback_params = original_params
