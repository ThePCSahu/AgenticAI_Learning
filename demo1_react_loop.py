import json
import re
from llm_client import get_llm_client

# Define local mock tools
def get_order_status(order_id: str) -> str:
    """Mock tool to retrieve order status from a database."""
    print(f"\033[96m[Local Tool] Running 'get_order_status' for order_id: {order_id}...\033[0m")
    orders = {
        "12345": {"status": "Shipped", "carrier": "FedEx", "delivery_date": "2026-06-25"},
        "67890": {"status": "Delivered", "carrier": "UPS", "delivery_date": "2026-06-21"},
    }
    order = orders.get(order_id)
    if order:
        return json.dumps(order)
    return json.dumps({"error": f"Order {order_id} not found."})

def calculate_discount(price: float, member_tier: str) -> str:
    """Mock tool to calculate member discount."""
    print(f"\033[96m[Local Tool] Running 'calculate_discount' for price: {price}, tier: {member_tier}...\033[0m")
    multiplier = 0.95
    if member_tier.lower() == "gold":
        multiplier = 0.85
    elif member_tier.lower() == "platinum":
        multiplier = 0.80
    discounted = round(price * multiplier, 2)
    return json.dumps({"original_price": price, "discounted_price": discounted, "member_tier": member_tier})

# Map available tools
TOOL_REGISTRY = {
    "get_order_status": get_order_status,
    "calculate_discount": calculate_discount
}

SYSTEM_INSTRUCTION = """You are a helpful AI Agent working in a Reasoning and Acting (ReAct) loop.
You have access to the following tools:
1. `get_order_status(order_id: str)`: Returns status and shipping info for an order.
2. `calculate_discount(price: float, member_tier: str)`: Returns the discounted price based on membership tier.

You MUST respond using the following structured formats exactly. Do not output anything else.

To call a tool, format your response as:
Thought: <your reasoning explaining why you need this tool>
Action: {"tool": "<tool_name>", "args": {<arguments>}}

When you have the final answer to the user request, format your response as:
Thought: <your final reasoning>
Final Answer: <your response to the user>

Ensure your 'Action' output is a valid JSON block on its own line.
"""

def parse_react_output(text: str):
    """Parses Thought and Action or Final Answer from LLM response."""
    thought_match = re.search(r"Thought:\s*(.*?)(?=(Action:|Final Answer:|$))", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else "Reasoning..."

    action_match = re.search(r"Action:\s*(\{.*)", text, re.DOTALL)
    if action_match:
        action_str = action_match.group(1).strip()
        # Find balanced braces to extract the exact JSON block
        open_braces = 0
        json_end = -1
        for i, char in enumerate(action_str):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
                if open_braces == 0:
                    json_end = i + 1
                    break
        if json_end != -1:
            action_json_str = action_str[:json_end]
            try:
                action_data = json.loads(action_json_str)
                return thought, "action", action_data
            except json.JSONDecodeError:
                return thought, "error", f"Failed to parse action JSON: {action_json_str}"

    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if final_match:
        return thought, "final", final_match.group(1).strip()

    return thought, "text", text.strip()

def run_agent(user_prompt: str):
    print("=" * 60)
    print("\033[94m[Agent Init] Starting Bare-Metal ReAct Loop...\033[0m")
    print(f"\033[94m[Agent Init] User Goal:\033[0m {user_prompt}")
    print("=" * 60)

    # Initialize client
    llm = get_llm_client()
    
    # Initialize message list
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt}
    ]

    max_loops = 5
    loop_count = 0
    
    # Core execution loop
    while True:
        loop_count += 1
        print(f"\n--- Loop Iteration {loop_count} (Max Limit: {max_loops}) ---")
        
        # Guard clause to prevent runaway infinite execution cascades
        if loop_count > max_loops:
            print("\033[91m[Safeguard] REACHED MAX ITERATION LIMIT (5). Breaking loop to prevent infinite token budget bleed.\033[0m")
            break

        # Perceive: Send entire context history to the Brain
        response = llm.generate(messages)
        
        # Parse output
        thought, msg_type, payload = parse_react_output(response)
        
        # Display thinking block
        print(f"\033[35mThought:\033[0m {thought}")
        
        # Append assistant's thoughts/action to history
        messages.append({"role": "assistant", "content": response})

        if msg_type == "action":
            tool_name = payload.get("tool")
            tool_args = payload.get("args", {})
            print(f"\033[93mAction:\033[0m Call tool '{tool_name}' with args {tool_args}")
            
            # Execute tool in the Host environment
            if tool_name in TOOL_REGISTRY:
                try:
                    # Execute tool function dynamically
                    observation = TOOL_REGISTRY[tool_name](**tool_args)
                    print(f"\033[92mObservation:\033[0m {observation}")
                    
                    # Feed observation back into context
                    messages.append({"role": "observation", "content": observation})
                except Exception as e:
                    error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                    print(f"\033[91mObservation (Error):\033[0m {error_msg}")
                    messages.append({"role": "observation", "content": error_msg})
            else:
                error_msg = f"Tool '{tool_name}' is not registered."
                print(f"\033[91mObservation (Error):\033[0m {error_msg}")
                messages.append({"role": "observation", "content": error_msg})
                
        elif msg_type == "final":
            print(f"\n\033[92m[Success] Final Answer:\033[0m {payload}")
            break
            
        elif msg_type == "error":
            print(f"\033[91m[Error parsing output]:\033[0m {payload}")
            # Feed parsing error back to LLM to self-correct
            messages.append({"role": "observation", "content": f"Error: {payload}. Please format action exactly as Action: {{\"tool\": \"name\", \"args\": {{...}}}}"})
            
        else:
            # Simple text output, treat as final answer fallback
            print(f"\n\033[92m[Fallback Final Answer]:\033[0m {payload}")
            break

    print("=" * 60)
    print("\033[94m[Agent Done] ReAct execution finished.\033[0m")
    print("=" * 60)

if __name__ == "__main__":
    # Test request requiring tool use
    run_agent("Can you check the shipping status for order 12345?")
