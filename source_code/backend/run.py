"""Start the local Agent 1-5 API and chat interface."""

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "source_code.backend.api:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=False,
    )
