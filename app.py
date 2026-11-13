"""Shared persistent research assistant entry point; no import-time side effects."""
from assistant.ui import build_ui


if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False)
