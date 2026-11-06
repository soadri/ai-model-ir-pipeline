import os

import uvicorn


def main():
    uvicorn.run("server.app:app", host=os.environ.get("RNGD_HOST", "127.0.0.1"),
                port=int(os.environ.get("RNGD_PORT", "8100")), log_level="info")


if __name__ == "__main__":
    main()
