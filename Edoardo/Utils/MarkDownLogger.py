import os
from datetime import datetime

class MarkdownLogger:
    def __init__(self, filename="evolution_report.md"):
        self.filename = filename
        # Create file with a header if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write(f"# ERA Evolution Report\n")
                f.write(f"Started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def log_header(self, text, level=2):
        prefix = "#" * level
        self._write(f"✨ \n{prefix} {text}\n")

    def log_stat(self, label, value):
        self._write(f"📊 - **{label}:** `{value}`\n")

    def log_event(self, text):
        self._write(f"🧬 {text}  \n") # Double space for MD line break

    def _write(self, content):
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(content)

# Example usage:
#md_logger.log_event(f"Generated new Problem Pool with {len(problem_pool)} tasks.")

# Initialize global logger
md_logger = MarkdownLogger()