import os
import sys
import json
import subprocess
import threading

class MCPClient:
    """A framework-free Client for communicating with an MCP server over stdio pipes."""
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.process = None
        self.msg_id = 0
        self.stderr_thread = None

    def start(self):
        """Spawns the MCP server as a subprocess using python."""
        print(f"\033[94m[MCP Client] Launching MCP Server: {os.path.basename(self.server_script_path)}...\033[0m")
        # Run python script in unbuffered mode (-u) so stdio lines are flushed instantly
        self.process = subprocess.Popen(
            [sys.executable, "-u", self.server_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Start a background thread to log server's stderr to our client console
        self.stderr_thread = threading.Thread(target=self._log_server_stderr, daemon=True)
        self.stderr_thread.start()

    def _log_server_stderr(self):
        """Continuously reads server's stderr and prints it to prevent pipe buffer blocks."""
        try:
            for line in self.process.stderr:
                # Print server logging in gray to distinguish it from client logs
                sys.stderr.write(f"\033[90m    | {line.strip()}\033[0m\n")
        except Exception:
            pass

    def _send_request(self, method: str, params: dict = None) -> dict:
        """Helper to format and send JSON-RPC requests, then wait for the response."""
        self.msg_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self.msg_id,
            "method": method,
            "params": params or {}
        }
        
        # Send query down the pipe
        req_json = json.dumps(req) + "\n"
        self.process.stdin.write(req_json)
        self.process.stdin.flush()
        
        # Read the immediate next response line from stdout
        resp_line = self.process.stdout.readline()
        if not resp_line:
            raise IOError("No response received from MCP Server (pipe closed).")
            
        resp = json.loads(resp_line)
        if "error" in resp:
            raise RuntimeError(f"JSON-RPC Error: {resp['error']['message']}")
        return resp.get("result", {})

    def initialize(self) -> dict:
        """Handshake capability negotiation step."""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "NoNonsense-Host-Client", "version": "1.0.0"}
        }
        print("\033[94m[MCP Client] Sending 'initialize' request...\033[0m")
        return self._send_request("initialize", params)

    def list_tools(self) -> list:
        """Retrieves list of tools the server exposes."""
        print("\033[94m[MCP Client] Sending 'tools/list' request...\033[0m")
        result = self._send_request("tools/list")
        return result.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invokes a specific tool on the server and returns its text value."""
        print(f"\033[94m[MCP Client] Sending 'tools/call' for '{tool_name}' with args {arguments}...\033[0m")
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        result = self._send_request("tools/call", params)
        content_list = result.get("content", [])
        if content_list and content_list[0].get("type") == "text":
            return content_list[0].get("text")
        return ""

    def close(self):
        """Cleans up pipes and terminates the server process."""
        if self.process:
            print("\033[94m[MCP Client] Closing connection to MCP Server...\033[0m")
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            print("\033[94m[MCP Client] MCP Server process terminated.\033[0m")
