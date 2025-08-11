"""
This script starts an HTTP server to serve XML files with the correct content type and cache headers.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import shutil
from pathlib import Path
import os
import urllib.parse
import posixpath
from logging_setup import setup_logging, get_logger

# Initialize logging
logger = setup_logging()
server_logger = get_logger('server')

# Define directory paths and filenames
UGLYFEED_FILE = "uglyfeed.xml"  # Define this at the top with other constants
uglyfeed_file = UGLYFEED_FILE  # Alias for UGLYFEED_FILE
UGLYFEEDS_DIR = Path("uglyfeeds").resolve()
STATIC_DIR = Path(".streamlit") / "static" / "uglyfeeds"

class CustomXMLHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler to serve XML files with correct content type and cache headers."""

    protocol_version = "HTTP/1.1"

    def do_GET(self):
        """Handle GET requests."""
        # Health endpoint
        if self.path == "/_health":
            self._send_text_response(200, "ok\n")
            return

        # Only allow .xml requests
        if not self.path.lower().endswith(".xml"):
            self._not_found("Only .xml is served")
            return
        
        file_path = self._safe_resolve_path(self.path)
        if not file_path:
            self._not_found("Invalid path")
            return

        if file_path.exists() and file_path.is_file():
            try:
                data = file_path.read_bytes()
            except Exception as e:
                server_logger.error("Error reading file %s: %s", file_path, e)
                self.send_error(500, "Internal server error")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(data)))
            # No caching by default; adjust if you want CDN caching
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0")
            self.end_headers()
            self.wfile.write(data)
            server_logger.info("Served XML file: %s", file_path)
        else:
            self._not_found(f"File not found: {file_path}")

    def _serve_xml_file(self):
        """Serve an XML file with appropriate headers."""
        file_path = STATIC_DIR / self.path.lstrip('/')

        if file_path.exists() and file_path.is_file():
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0")
            self.end_headers()

            with open(file_path, 'rb') as file:
                self.wfile.write(file.read())
            server_logger.info("Served XML file: %s", file_path)
        else:
            self.send_error(404, "File not found")
            server_logger.warning("XML file not found: %s", file_path)
    
    def do_HEAD(self):
        # Mirror GET logic, but only send headers
        if self.path == "/_health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", "3")
            self.end_headers()
            return

        if not self.path.lower().endswith(".xml"):
            self._not_found("Only .xml is served")
            return

        file_path = self._safe_resolve_path(self.path)
        if file_path and file_path.exists() and file_path.is_file():
            try:
                size = file_path.stat().st_size
            except Exception as e:
                server_logger.error("Error stat'ing file %s: %s", file_path, e)
                self.send_error(500, "Internal server error")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0")
            self.end_headers()
        else:
            self._not_found("File not found")
    
    # Disable directory listing
    def list_directory(self, path):
        self._not_found("Directory listing disabled")
        return None
    
    def _safe_resolve_path(self, url_path: str) -> Path | None:
        # Strip query/fragment and normalize
        raw = url_path.split("?", 1)[0].split("#", 1)[0]
        norm = posixpath.normpath(urllib.parse.unquote(raw))
        parts = [p for p in norm.split("/") if p not in ("", ".", "..")]

        # Map to UGLYFEEDS_DIR
        candidate = UGLYFEEDS_DIR.joinpath(*parts).resolve()
        try:
            candidate.relative_to(UGLYFEEDS_DIR)
        except ValueError:
            # Attempted escape outside allowed directory
            server_logger.warning("Path traversal blocked: %s -> %s", url_path, candidate)
            return None
        return candidate

    def _send_text_response(self, code: int, text: str):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self, msg="Not Found"):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        body = f"404: {msg}\n".encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        server_logger.warning("%s %s -> 404 (%s)", self.command, self.path, msg)

def start_http_server(port):
    """Start the HTTP server to serve XML files."""
    try:
        server_address = ('', port)
        httpd = HTTPServer(server_address, CustomXMLHandler)
        server_logger.info("Starting server on port %d", port)
        httpd.serve_forever()
    except Exception as e:
        server_logger.error("Failed to start server on port %d: %s", port, e)
        raise

def toggle_server(start, port, session_state):
    """Start the HTTP server bound to localhost to avoid direct exposure.
    Cloudflare Tunnel (cloudflared) should connect to http://127.0.0.1:<port>."""
    try:
        server_address = ('127.0.0.1', port)
        httpd = HTTPServer(server_address, CustomXMLHandler)
        server_logger.info("Starting XML server on http://127.0.0.1:%d serving %s", port, UGLYFEEDS_DIR)
        httpd.serve_forever()
    except Exception as e:
        server_logger.error("Failed to start server on port %d: %s", port, e)
        raise

def toggle_server(start: bool, port: int, session_state: dict):
    """
    Toggle the HTTP server on or off in a background thread.
    """
    if start:
        if not session_state.get('server_thread') or not session_state['server_thread'].is_alive():
            # Ensure the serving directory exists
            UGLYFEEDS_DIR.mkdir(parents=True, exist_ok=True)

            session_state['server_thread'] = threading.Thread(
                target=start_http_server, args=(port,), daemon=True
            )
            session_state['server_thread'].start()
            server_logger.info("Server started on port %d.", port)
        else:
            server_logger.info("Server is already running.")
    else:
        if session_state.get('server_thread') and session_state['server_thread'].is_alive():
            # We can't directly stop HTTPServer here; rely on process/lifecycle or add a shutdown pipe
            session_state['server_thread'] = None
            server_logger.info("Server stop requested.")
        else:
            server_logger.info("Server is not running.")

def copy_xml_to_static():
    """
    No-op copy retained for compatibility.
    We now serve directly from UGLYFEEDS_DIR, so copying is not required.
    Returns the expected path if the file exists.
    """
    source_file = UGLYFEEDS_DIR / UGLYFEED_FILE
    if source_file.exists() and source_file.is_file():
        server_logger.info("XML available at %s; static copy not required.", source_file)
        return source_file
    else:
        server_logger.warning("Source file %s does not exist in %s.", UGLYFEED_FILE, UGLYFEEDS_DIR)
        return None
