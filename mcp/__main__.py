import os
import sys


def main():
    try:
        import uvicorn
    except Exception as e:
        print("uvicorn is required to run the MCP server. pip install uvicorn", file=sys.stderr)
        raise
    from .server import app

    port = int(os.environ.get("MCP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

