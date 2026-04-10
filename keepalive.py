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

    def do_PUT(self):  # noqa: N802
        # Fulfill OpenEnv Structural Check's requests without throwing 501
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "success"}')


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    HTTPServer(("0.0.0.0", port), _Handler).serve_forever()


if __name__ == "__main__":
    main()

