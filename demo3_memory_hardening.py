import os
import sys
import json
import sqlite3
import argparse
from llm_client import get_llm_client
from mcp_client import MCPClient
from demo1_react_loop import parse_react_output
from demo2_mcp_integration import build_system_instruction

DB_PATH = "agent_state.db"
PERSONA_PATH = "persona_store.json"
SESSION_ID = "presentation_session_001"

# ----------------- Database Checkpoint Store (LTSM) -----------------
def init_state_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            session_id TEXT PRIMARY KEY,
            loop_count INTEGER,
            status TEXT,
            messages TEXT,
            pending_action TEXT,
            original_prompt TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_checkpoint(session_id: str, loop_count: int, status: str, messages: list, pending_action: dict = None, prompt: str = ""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO checkpoints (session_id, loop_count, status, messages, pending_action, original_prompt)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            loop_count = excluded.loop_count,
            status = excluded.status,
            messages = excluded.messages,
            pending_action = excluded.pending_action,
            original_prompt = excluded.original_prompt
    """, (
        session_id,
        loop_count,
        status,
        json.dumps(messages),
        json.dumps(pending_action) if pending_action else None,
        prompt
    ))
    conn.commit()
    conn.close()
    print(f"\033[90m[LTSM] Checkpoint saved successfully. Status: {status}, Loop: {loop_count}.\033[0m")

def load_checkpoint(session_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT loop_count, status, messages, pending_action, original_prompt FROM checkpoints WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "loop_count": row[0],
            "status": row[1],
            "messages": json.loads(row[2]),
            "pending_action": json.loads(row[3]) if row[3] else None,
            "original_prompt": row[4]
        }
    return None

def clear_checkpoint(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    print("\033[90m[LTSM] Checkpoint cleared from database.\033[0m")

# ----------------- Long-Term Fact Memory -----------------
def init_persona_store():
    if not os.path.exists(PERSONA_PATH):
        default_store = {
            "user_preferences": [
                "User prefers database index optimizations to be verified afterward."
            ]
        }
        with open(PERSONA_PATH, "w") as f:
            json.dump(default_store, f, indent=2)

def load_long_term_facts() -> str:
    if os.path.exists(PERSONA_PATH):
        try:
            with open(PERSONA_PATH, "r") as f:
                data = json.load(f)
                facts = data.get("user_preferences", [])
                if facts:
                    fact_str = "\n".join([f"- {fact}" for fact in facts])
                    return f"\n[Long-Term Fact Store (Appended Persona Profile)]:\n{fact_str}\n"
        except Exception as e:
            print(f"[Memory Store] Error reading facts: {e}")
    return ""

def learn_new_fact(fact: str):
    """Saves a new fact to the long-term store."""
    if os.path.exists(PERSONA_PATH):
        try:
            with open(PERSONA_PATH, "r+") as f:
                data = json.load(f)
                if "user_preferences" not in data:
                    data["user_preferences"] = []
                if fact not in data["user_preferences"]:
                    data["user_preferences"].append(fact)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                    print(f"\033[92m[Long-Term Fact Memory] Learned and persisted new fact: '{fact}'\033[0m")
        except Exception as e:
            print(f"[Memory Store] Error updating facts: {e}")

# ----------------- Short-Term Recursive Summarization -----------------
def run_recursive_summarization(llm, messages: list) -> list:
    """Compresses older message turns into a single summary block to avoid context bloom."""
    print("\033[93m[Short-Term Memory] Turning threshold reached! Running recursive summarization...\033[0m")
    
    # We take all messages except system instructions and the last turn, and ask the LLM to summarize them
    summary_messages = [
        {"role": "system", "content": "Summarize the key decisions, tools used, and status of the current task in one brief paragraph."},
        {"role": "user", "content": f"Please summarize the history of this interaction:\n{json.dumps(messages[1:-1])}"}
    ]
    summary_content = llm.generate(summary_messages)
    print(f"\033[92m[Short-Term Memory] Compressed Status Summary:\033[0m {summary_content}")
    
    # Keep the system instruction, inject the compressed summary, and append the latest turns
    new_messages = [
        messages[0], # Keep system instructions
        {"role": "system", "content": f"[COMPRESSED HISTORY SUMMARY]: {summary_content}"},
        messages[-1] # Keep the very last user/observation turn
    ]
    return new_messages

# ----------------- Main Agent Execution -----------------
def run_hardened_agent(prompt: str, mcp_client: MCPClient, force_approve: bool = False):
    llm = get_llm_client()
    
    # Handshake & Discover tools
    mcp_client.initialize()
    discovered_tools = mcp_client.list_tools()
    
    # Load long-term facts and compile system instructions
    lt_facts = load_long_term_facts()
    system_instruction = build_system_instruction(discovered_tools) + lt_facts
    
    # Check for active checkpoint
    checkpoint = load_checkpoint(SESSION_ID)
    
    if checkpoint and checkpoint["status"] == "PAUSED_HITL":
        print("\n" + "=" * 60)
        print("\033[93m[LTSM] FOUND ACTIVE CHECKPOINT IN DATABASE! REHYDRATING SESSION STATE...\033[0m")
        print(f"\033[93m[LTSM] Rehydrated Prompt Goal:\033[0m {checkpoint['original_prompt']}")
        print(f"\033[93m[LTSM] Loop Count resumed at:\033[0m {checkpoint['loop_count']}")
        print("=" * 60)
        
        messages = checkpoint["messages"]
        loop_count = checkpoint["loop_count"]
        pending_action = checkpoint["pending_action"]
        original_prompt = checkpoint["original_prompt"]
        
        # We are paused awaiting approval
        print(f"\033[93m[Awaiting Human Approval] Pending Action:\033[0m {json.dumps(pending_action)}")
        
        user_input = ""
        if force_approve:
            user_input = "approve"
            print("[HITL] Flag '--approve' provided. Automatically approving action.")
        else:
            user_input = input("\nEnter 'approve' to execute, 'reject' to abort, or press enter to keep paused: ").strip().lower()
            
        if user_input == "approve":
            print("\033[92m[HITL] Human Approval Granted. Resuming execution...\033[0m")
            tool_name = pending_action["tool"]
            tool_args = pending_action["args"]
            
            # Execute mutating tool via MCP client
            observation = mcp_client.call_tool(tool_name, tool_args)
            print(f"\033[92mObservation (from MCP Server):\033[0m {observation}")
            
            messages.append({"role": "observation", "content": observation})
            
            # Update checkpoint state to RUNNING and clear pending action
            save_checkpoint(SESSION_ID, loop_count, "RUNNING", messages, None, original_prompt)
        elif user_input == "reject":
            print("\033[91m[HITL] Human Approval Rejected. Aborting agent run.\033[0m")
            clear_checkpoint(SESSION_ID)
            return
        else:
            print("\033[93m[HITL] Still paused. Exiting script. Run with '--approve' or answer 'approve' to resume.\033[0m")
            return
            
    else:
        # Start new session
        print("=" * 60)
        print("\033[94m[Agent Init] Starting Hardened Agentic Loop (Summarization + Checkpointing + HITL)...\033[0m")
        print(f"\033[94m[Agent Init] Goal:\033[0m {prompt}")
        print("=" * 60)
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        loop_count = 0
        original_prompt = prompt
        save_checkpoint(SESSION_ID, loop_count, "RUNNING", messages, None, original_prompt)

    max_loops = 5
    
    # Core loop
    while True:
        loop_count += 1
        print(f"\n--- Loop Iteration {loop_count} (Max Limit: {max_loops}) ---")
        
        if loop_count > max_loops:
            print("\033[91m[Safeguard] REACHED MAX ITERATION LIMIT (5). Breaking loop.\033[0m")
            clear_checkpoint(SESSION_ID)
            break

        # 1. Short-Term Recursive Summarization Check
        # Threshold: if message turns exceeds 6 messages, compress old logs
        if len(messages) > 6:
            messages = run_recursive_summarization(llm, messages)

        # Perceive
        response = llm.generate(messages)
        
        # Parse output
        thought, msg_type, payload = parse_react_output(response)
        
        print(f"\033[35mThought:\033[0m {thought}")
        messages.append({"role": "assistant", "content": response})

        if msg_type == "action":
            tool_name = payload.get("tool")
            tool_args = payload.get("args", {})
            
            # 2. HITL Safeguard: Intercept destructive write mutations
            if tool_name == "db_mutation":
                print(f"\n\033[93m[HITL Safeguard] DETECTED DESTRUCTIVE ACTION: '{tool_name}' targeting query '{tool_args.get('mutation_query')}'\033[0m")
                print("\033[93m[HITL Safeguard] Pausing execution, serialization active, writing state checkpoint...\033[0m")
                
                # Save state as PAUSED_HITL to SQLite DB
                save_checkpoint(SESSION_ID, loop_count, "PAUSED_HITL", messages, payload, original_prompt)
                
                print("\033[93m[Awaiting Human Approval] Process suspended. Please restart this script and type 'approve' to resume execution.\033[0m")
                return # Exit process
                
            # If normal tool, run it
            print(f"\033[93mAction:\033[0m Call tool '{tool_name}' via MCP client...")
            try:
                observation = mcp_client.call_tool(tool_name, tool_args)
                print(f"\033[92mObservation:\033[0m {observation}")
                messages.append({"role": "observation", "content": observation})
            except Exception as e:
                error_msg = f"Error calling MCP tool '{tool_name}': {str(e)}"
                print(f"\033[91mObservation (Error):\033[0m {error_msg}")
                messages.append({"role": "observation", "content": error_msg})
                
            # 3. Save State Checkpoint (LTSM) after every loop iteration
            save_checkpoint(SESSION_ID, loop_count, "RUNNING", messages, None, original_prompt)

        elif msg_type == "final":
            print(f"\n\033[92m[Success] Final Answer:\033[0m {payload}")
            
            # Learn new preferences if the prompt was database optimization
            if "optimize" in original_prompt.lower():
                learn_new_fact("User prefers database index optimizations to be verified afterward.")
                
            clear_checkpoint(SESSION_ID)
            break
            
        elif msg_type == "error":
            print(f"\033[91m[Error parsing output]:\033[0m {payload}")
            messages.append({"role": "observation", "content": f"Error: {payload}. Please format action exactly as Action: {{\"tool\": \"name\", \"args\": {{...}}}}"})
            save_checkpoint(SESSION_ID, loop_count, "RUNNING", messages, None, original_prompt)
        else:
            print(f"\n\033[92m[Fallback Final Answer]:\033[0m {payload}")
            clear_checkpoint(SESSION_ID)
            break

def main():
    parser = argparse.ArgumentParser(description="Demo 3: Memory Matrix & Production Hardening")
    parser.add_argument("--approve", action="store_true", help="Automatically approve the suspended HITL action")
    parser.add_argument("--prompt", type=str, default="Optimize the database by running a mutation query to create an index for faster search.", help="Goal prompt for the agent")
    args = parser.parse_args()

    # Initialize data files
    init_state_db()
    init_persona_store()
    
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    mcp_client = MCPClient(server_script)
    mcp_client.start()
    
    try:
        run_hardened_agent(args.prompt, mcp_client, force_approve=args.approve)
    finally:
        mcp_client.close()

if __name__ == "__main__":
    main()
