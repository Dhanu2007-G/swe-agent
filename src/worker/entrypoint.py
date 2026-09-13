"""
src/worker/entrypoint.py — RQ worker process entrypoint.
Handles graceful shutdown on SIGTERM, registers queues, configures logging.
"""

from __future__ import annotations

import os
import signal
import sys

import structlog
from rq import Queue, Worker
from rq.timeouts import JobTimeoutException

from src.config import get_settings
from src.observability.tracing import configure_logging
from src.worker.queue import HIGH_PRIORITY_QUEUE, QUEUE_NAME, get_sync_redis

_RQ_AVAILABLE = True

log = structlog.get_logger(__name__)


def main() -> None:
    if not _RQ_AVAILABLE:
        log.error("worker.rq_unavailable", error="RQ is not installed")
        sys.exit(1)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    redis_conn = get_sync_redis()
    queues = [
        Queue(HIGH_PRIORITY_QUEUE, connection=redis_conn),
        Queue(QUEUE_NAME, connection=redis_conn),
    ]  # high priority processed first

    worker = Worker(
        queues=queues,
        connection=redis_conn,
        name=f"worker-{os.getpid()}",
        log_job_description=True,
        exception_handlers=[_handle_job_exception],
    )

    log.info(
        "worker.starting",
        pid=os.getpid(),
        queues=queues,
        timeout=settings.redis_job_timeout,
    )

    # Graceful shutdown on SIGTERM
    def _sigterm_handler(signum: int, frame: object) -> None:
        log.info("worker.sigterm_received", pid=os.getpid())
        worker.request_stop(signum, frame)  # type: ignore[no-untyped-call]

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        worker.work(with_scheduler=False)
    except KeyboardInterrupt:
        log.info("worker.keyboard_interrupt")
    finally:
        log.info("worker.stopped", pid=os.getpid())


def _handle_job_exception(
    job: object, exc_type: type, exc_value: Exception, traceback: object
) -> bool:
    """
    Custom exception handler for RQ jobs.
    Returns True to stop further exception handling (job stays failed).
    Returns False to let RQ's default handler run.
    """
    log.error(
        "worker.job_exception",
        job_id=getattr(job, "id", "unknown"),
        exc_type=exc_type.__name__,
        error=str(exc_value),
    )

    if isinstance(exc_value, JobTimeoutException):
        log.warning(
            "worker.job_timeout",
            job_id=getattr(job, "id", "unknown"),
        )

    return False  # let RQ default handler also run (moves to failed queue)


if __name__ == "__main__":
    main()
