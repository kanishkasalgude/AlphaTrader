import http.server
import socketserver
import json

class MockProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        print(f"[MOCK PROXY] Received POST at {self.path}")
        print("[MOCK PROXY] Headers:")
        for k, v in self.headers.items():
            print(f"  {k}: {v}")
            
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        print(f"[MOCK PROXY] Body: {post_data}")
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mocked proxy response."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

if __name__ == "__main__":
    with socketserver.TCPServer(("", 8089), MockProxyHandler) as httpd:
        print("Mock proxy running on port 8089...")
        httpd.serve_forever()
