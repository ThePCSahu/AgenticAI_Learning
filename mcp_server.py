import sys
import json
import sqlite3
import traceback

def log(msg: str):
    """Write logs to sys.stderr so it doesn't pollute the JSON-RPC stdio pipe."""
    sys.stderr.write(f"[MCP Server] {msg}\n")
    sys.stderr.flush()

def init_db():
    conn = sqlite3.connect("inventory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY,
            item_name TEXT,
            quantity INTEGER,
            location TEXT
        )
    """)
    # Seed data if empty
    cursor.execute("SELECT COUNT(*) FROM inventory")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO inventory (item_name, quantity, location) VALUES (?, ?, ?)
        """, [
            ("Laptop", 15, "Bin A4"),
            ("Smartphone", 30, "Bin B2"),
            ("Monitor", 8, "Bin C1"),
        ])
        conn.commit()
        log("Database initialized and seeded.")
    else:
        log("Database already initialized.")
    conn.close()

def query_database(query: str) -> str:
    log(f"Executing Query: {query}")
    # Validate to ensure SELECT only for safety
    if not query.strip().lower().startswith("select"):
        return json.dumps({"error": "Only SELECT queries are allowed via this tool."})
        
    try:
        conn = sqlite3.connect("inventory.db")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        result = [dict(zip(columns, row)) for row in rows]
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_inventory(item_name: str) -> str:
    log(f"Fetching inventory details for: {item_name}")
    try:
        conn = sqlite3.connect("inventory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inventory WHERE item_name LIKE ?", (f"%{item_name}%",))
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        result = [dict(zip(columns, row)) for row in rows]
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

def db_mutation(mutation_query: str) -> str:
    """A write/mutation tool for database modifications (used for HITL demonstration)."""
    log(f"Executing Mutation: {mutation_query}")
    try:
        conn = sqlite3.connect("inventory.db")
        cursor = conn.cursor()
        cursor.execute(mutation_query)
        conn.commit()
        changes = conn.total_changes
        conn.close()
        return json.dumps({"success": True, "rows_changed": changes, "message": "Database mutated successfully."})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# MCP Tool Specification
AVAILABLE_TOOLS = [
    {
        "name": "query_database",
        "description": "Safe SELECT queries on the SQLite inventory database. Ideal for fetching items and stock levels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The SQLite SELECT query to run."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_inventory",
        "description": "Get inventory counts and location for a specific item name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string", "description": "Item name to look up."}
            },
            "required": ["item_name"]
        }
    },
    {
        "name": "db_mutation",
        "description": "Run database write operations (mutations like CREATE INDEX, UPDATE, DELETE). This is a destructive/mutating action.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mutation_query": {"type": "string", "description": "The SQLite SQL write query to execute."}
            },
            "required": ["mutation_query"]
        }
    }
]

def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "NoNonsense-Inventory-Server",
                    "version": "1.0.0"
                }
            }
        }
        
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": AVAILABLE_TOOLS
            }
        }
        
    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        result_content = ""
        if tool_name == "query_database":
            result_content = query_database(arguments.get("query", ""))
        elif tool_name == "get_inventory":
            result_content = get_inventory(arguments.get("item_name", ""))
        elif tool_name == "db_mutation":
            result_content = db_mutation(arguments.get("mutation_query", ""))
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method/Tool {tool_name} not found"
                }
            }
            
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result_content
                    }
                ]
            }
        }
        
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method {method} not found"
            }
        }

def main():
    init_db()
    log("MCP Server is running over stdio...")
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Server processing error: {str(e)}",
                    "data": traceback.format_exc()
                }
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
