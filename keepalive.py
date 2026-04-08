import os
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        # Silence access logs (keeps HF logs focused on CLI inference output)
        return

    def do_GET(self):  # noqa: N802
        # Return "No Content" for all requests
        self.send_response(204)
        self.end_headers()


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    HTTPServer(("0.0.0.0", port), _Handler).serve_forever()


if __name__ == "__main__":
    main()

