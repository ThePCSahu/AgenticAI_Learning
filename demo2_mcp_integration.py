import os
import json
import re
from llm_client import get_llm_client
from mcp_client import MCPClient
from demo1_react_loop import parse_react_output

# Helper to construct System Prompt dynamically based on discovered tools
def build_system_instruction(tools: list) -> str:
    tool_desc = ""
    for i, t in enumerate(tools):
        tool_desc += f"{i+1}. `{t['name']}`: {t['description']}\n"
        tool_desc += f"   Input Schema: {json.dumps(t['inputSchema'])}\n"
        
    return f"""You are a helpful AI Agent working in a Reasoning and Acting (ReAct) loop.
You have access to the following tools discovered dynamically from the MCP Server:
{tool_desc}

You MUST respond using the following structured formats exactly. Do not output anything else.

To call a tool, format your response as:
Thought: <your reasoning explaining why you need this tool>
Action: {{"tool": "<tool_name>", "args": {{<arguments>}}}}

When you have the final answer to the user request, format your response as:
Thought: <your final reasoning>
Final Answer: <your response to the user>

Ensure your 'Action' output is a valid JSON block on its own line.
"""

def run_mcp_agent(user_prompt: str):
    print("=" * 60)
    print("\033[94m[Agent Init] Starting ReAct Loop + MCP Integration...\033[0m")
    print(f"\033[94m[Agent Init] User Goal:\033[0m {user_prompt}")
    print("=" * 60)

    # Initialize MCP Client (looking for local mcp_server.py)
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    mcp_client = MCPClient(server_script)
    
    try:
        # Start server subprocess and execute handshake
        mcp_client.start()
        mcp_client.initialize()
        
        # Discover tools dynamically from the server
        discovered_tools = mcp_client.list_tools()
        print(f"\033[92m[MCP Client] Discovered {len(discovered_tools)} tools from server:\033[0m")
        for tool in discovered_tools:
            print(f"  - {tool['name']}: {tool['description'][:60]}...")

        # Build dynamic system instruction using discovered schemas
        system_instruction = build_system_instruction(discovered_tools)

        # Initialize LLM Client
        llm = get_llm_client()
        
        # Initialize conversation log
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]

        max_loops = 5
        loop_count = 0
        
        # Core execution loop
        while True:
            loop_count += 1
            print(f"\n--- Loop Iteration {loop_count} (Max Limit: {max_loops}) ---")
            
            # Guard clause
            if loop_count > max_loops:
                print("\033[91m[Safeguard] REACHED MAX ITERATION LIMIT (5). Breaking loop.\033[0m")
                break

            # Perceive: Send context to LLM
            response = llm.generate(messages)
            
            # Parse output
            thought, msg_type, payload = parse_react_output(response)
            
            print(f"\033[35mThought:\033[0m {thought}")
            messages.append({"role": "assistant", "content": response})

            if msg_type == "action":
                tool_name = payload.get("tool")
                tool_args = payload.get("args", {})
                print(f"\033[93mAction:\033[0m Call tool '{tool_name}' via MCP client...")
                
                # Check if tool is in discovered tools list
                if any(t['name'] == tool_name for t in discovered_tools):
                    try:
                        # Call tool via JSON-RPC over the stdio pipe
                        observation = mcp_client.call_tool(tool_name, tool_args)
                        print(f"\033[92mObservation (from MCP Server):\033[0m {observation}")
                        messages.append({"role": "observation", "content": observation})
                    except Exception as e:
                        error_msg = f"Error calling MCP tool '{tool_name}': {str(e)}"
                        print(f"\033[91mObservation (Error):\033[0m {error_msg}")
                        messages.append({"role": "observation", "content": error_msg})
                else:
                    error_msg = f"Tool '{tool_name}' is not supported by MCP server."
                    print(f"\033[91mObservation (Error):\033[0m {error_msg}")
                    messages.append({"role": "observation", "content": error_msg})
                    
            elif msg_type == "final":
                print(f"\n\033[92m[Success] Final Answer:\033[0m {payload}")
                break
                
            elif msg_type == "error":
                print(f"\033[91m[Error parsing output]:\033[0m {payload}")
                messages.append({"role": "observation", "content": f"Error: {payload}. Please format action exactly as Action: {{\"tool\": \"name\", \"args\": {{...}}}}"})
                
            else:
                print(f"\n\033[92m[Fallback Final Answer]:\033[0m {payload}")
                break

    except Exception as e:
        print(f"\033[91m[Fatal Host Error]:\033[0m {str(e)}")
        
    finally:
        # Clean up process pipes to prevent zombie subprocesses
        mcp_client.close()

    print("=" * 60)
    print("\033[94m[Agent Done] MCP-integrated execution finished.\033[0m")
    print("=" * 60)

if __name__ == "__main__":
    # Test request requiring database access over MCP
    run_mcp_agent("Can you query the database to find the stock level of 'Laptop'?")
