import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    # 1) stdout 永远开（容器 docker logs 仍能看到）
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    stdout_handler.setLevel(level)
    root_logger.addHandler(stdout_handler)

    # 2) 文件日志：写到 settings.log_dir，挂载点 /app/logs/ai。
    #    目录不存在或无写权限时 silently fallback 仅 stdout（保证容器不挂）。
    if settings.log_file_enable:
        try:
            os.makedirs(settings.log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=os.path.join(settings.log_dir, "app.log"),
                maxBytes=settings.log_file_max_bytes,
                backupCount=settings.log_file_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(fmt)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover — 仅启动期诊断
            root_logger.warning(
                "log file handler disabled (dir=%s): %s",
                settings.log_dir, exc,
            )

    # 3) 让 uvicorn 自带的三个 logger 走 root（默认它们自带 handler 不 propagate）
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(level)
