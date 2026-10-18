"""Start the local HTTP service with: python -m model_anatomy.api."""

import logging

import uvicorn

from .app import create_app
from .settings import Settings


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(settings),
        host="127.0.0.1",
        port=settings.port,
        access_log=False,
        server_header=False,
        limit_concurrency=32,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
