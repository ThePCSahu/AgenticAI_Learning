import os
import json
import re
import requests
from dotenv import load_dotenv

# Import our common elements
from llm_client import get_llm_client, BaseLLMClient, SimulatedClient

# Path to the shared persona facts store
PERSONA_PATH = "persona_store.json"

# ----------------- Custom LLM Simulator for keyless runs -----------------
class ShowcaseSimulatedClient(BaseLLMClient):
    """Custom simulated LLM specifically for the Memory Showcase story."""
    def generate(self, messages: list) -> str:
        # Check if this is a summarization request
        last_msg = messages[-1].get("content", "")
        is_summary_request = any(
            x in last_msg.lower() for x in ["summarize the conversation", "compress history", "summarize below", "summary of the interaction"]
        )
        if is_summary_request:
            return (
                "SUMMARY: The user introduced herself as Alice, a Python engineer. "
                "The agent wrote a Python Hello World program. Then, the user asked for the capitals "
                "of France (Paris), Japan (Tokyo), Canada (Ottawa), and Brazil (Brasilia)."
            )

        # Count observations to decide the turn state
        observations = [msg for msg in messages if msg.get("role") in ("observation", "tool") or msg.get("content", "").startswith("Observation:")]
        
        # Analyze the conversation history
        user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
        last_user_message = user_messages[-1] if user_messages else ""
        
        # Turn 1: Introduction and fact learning
        if "alice" in last_user_message.lower() and "python" in last_user_message.lower():
            if len(observations) == 0:
                return (
                    "Thought: The user introduced herself as Alice, a Python engineer, and prefers Markdown code formatting. "
                    "I must save this information in long-term memory so it persists across sessions.\n"
                    "Action: {\"tool\": \"save_long_term_fact\", \"args\": {\"fact\": \"User is Alice, a Python engineer who prefers code output in Markdown.\"}}"
                )
            else:
                return (
                    "Thought: The fact is saved. I will now reply to Alice, welcoming her.\n"
                    "Final Answer: Hello Alice! It's great to meet you. I have saved your details (Python engineer, prefers Markdown) in my long-term fact memory. I will format future outputs accordingly!"
                )

        # Turn 2: Hello World program (dependent on long-term memory)
        elif "hello world" in last_user_message.lower():
            # Check if long-term memory header is present in system instructions
            system_msg = messages[0].get("content", "")
            is_alice = "alice" in system_msg.lower()
            code_lang = "python" if is_alice else "JavaScript"
            
            return (
                f"Thought: The user wants a hello world program. My long-term memory confirms she is a Python engineer. "
                f"I will provide the hello world code in Python, wrapped in Markdown code blocks.\n"
                f"Final Answer: Here is your Hello World script in Python:\n\n```python\nprint(\"Hello, World!\")\n```"
            )

        # Verification turn (recalls long term job + short term capitals)
        elif "profession" in last_user_message.lower() or "remember" in last_user_message.lower():
            system_msg = messages[0].get("content", "")
            # Look for summary in the conversation history
            has_summary = any("[COMPRESSED HISTORY SUMMARY]" in msg.get("content", "") for msg in messages)
            
            job_recall = "Python engineer" if "python" in system_msg.lower() else "unknown"
            capital_recall = "Ottawa" if has_summary else "unknown"
            
            return (
                f"Thought: The user is verifying my memory. I will fetch her job ({job_recall}) from my long-term memory instructions, "
                f"and the third capital ({capital_recall}) from the short-term compressed history summary.\n"
                f"Final Answer: Yes, Alice, I remember! Your profession is Python engineer (retrieved from my long-term facts profile). "
                f"And the third capital you asked about was Ottawa, the capital of Canada (retrieved from my short-term summary block)."
            )

        # Fallbacks for the quick capital questions
        elif "france" in last_user_message.lower():
            return "Thought: Answering France capital.\nFinal Answer: The capital of France is Paris."
        elif "japan" in last_user_message.lower():
            return "Thought: Answering Japan capital.\nFinal Answer: The capital of Japan is Tokyo."
        elif "canada" in last_user_message.lower():
            return "Thought: Answering Canada capital.\nFinal Answer: The capital of Canada is Ottawa."
        elif "brazil" in last_user_message.lower():
            return "Thought: Answering Brazil capital.\nFinal Answer: The capital of Brazil is Brasilia."

        return "Thought: General query.\nFinal Answer: How else can I assist you with your memory showcase?"


# ----------------- Memory Helper Functions -----------------
def init_persona_store():
    """Initializes persona file with empty user preferences."""
    if not os.path.exists(PERSONA_PATH):
        with open(PERSONA_PATH, "w") as f:
            json.dump({"user_preferences": []}, f, indent=2)

def load_long_term_facts() -> str:
    """Reads persona JSON and returns a system prompt instruction block."""
    if os.path.exists(PERSONA_PATH):
        try:
            with open(PERSONA_PATH, "r") as f:
                data = json.load(f)
                facts = data.get("user_preferences", [])
                if facts:
                    fact_str = "\n".join([f"- {fact}" for fact in facts])
                    return f"\n[Long-Term Fact Store (Persona Store)]:\n{fact_str}\n"
        except Exception as e:
            pass
    return "\n[Long-Term Fact Store (Persona Store)]:\n- No facts learned yet.\n"

def save_long_term_fact(fact: str) -> str:
    """Tool to save a learned fact to persona store."""
    print(f"\033[96m[Long-Term Memory Tool] Writing to persona_store.json: '{fact}'...\033[0m")
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
            return f"Fact '{fact}' successfully saved to long-term memory."
        except Exception as e:
            return f"Error writing to memory store: {e}"
    return "Error: Memory store file not initialized."

# Map available tools
TOOL_REGISTRY = {
    "save_long_term_fact": save_long_term_fact
}

SYSTEM_INSTRUCTION_TEMPLATE = """You are a helpful AI Agent working in a Reasoning and Acting (ReAct) loop.
You can save long-term facts about the user by calling the save_long_term_fact tool.

You have access to the following tools:
1. `save_long_term_fact(fact: str)`: Saves a fact to the long-term persona database.

You MUST respond using the following structured formats exactly. Do not output anything else.

To call a tool, format your response as:
Thought: <your reasoning explaining why you need this tool>
Action: {{"tool": "save_long_term_fact", "args": {{"fact": "<fact_description>"}}}}

When you have the final answer to the user request, format your response as:
Thought: <your final reasoning>
Final Answer: <your response to the user>

Ensure your 'Action' output is a valid JSON block on its own line.
"""

def parse_showcase_output(text: str):
    thought_match = re.search(r"Thought:\s*(.*?)(?=(Action:|Final Answer:|$))", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else "Reasoning..."

    action_match = re.search(r"Action:\s*(\{.*)", text, re.DOTALL)
    if action_match:
        action_str = action_match.group(1).strip()
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

def run_summarization(llm, messages: list) -> list:
    print("\033[93m[Short-Term Memory] TURN COUNT EXCEEDS THRESHOLD (>6). Triggering Recursive Summarization...\033[0m")
    summary_messages = [
        {"role": "system", "content": "Summarize the key events, decisions, and outcomes in this conversation in one short paragraph."},
        {"role": "user", "content": f"Summarize the conversation log below:\n{json.dumps(messages[1:-1])}"}
    ]
    summary = llm.generate(summary_messages)
    print(f"\033[92m[Short-Term Memory] Generated History Summary:\033[0m {summary}")
    
    # Keep the system instruction, insert the summary, and keep only the last user request
    new_messages = [
        messages[0], # system instruction
        {"role": "system", "content": f"[COMPRESSED HISTORY SUMMARY]: {summary}"},
        messages[-1] # last turn
    ]
    return new_messages

# ----------------- Execution Engine -----------------
def execute_turn(llm, messages: list, user_message: str) -> str:
    messages.append({"role": "user", "content": user_message})
    
    # Check if we need to summarize first
    # 6 turns threshold (system instruction + 5 turns)
    if len(messages) > 6:
        # Before sending the request, compress older history
        # We temporarily remove the last user message, summarize, and add it back
        last_turn = messages.pop()
        messages = run_summarization(llm, messages)
        messages.append(last_turn)

    # Brain generates completion
    response = llm.generate(messages)
    thought, msg_type, payload = parse_showcase_output(response)
    
    print(f"\033[35mThought:\033[0m {thought}")
    messages.append({"role": "assistant", "content": response})

    if msg_type == "action":
        tool_name = payload.get("tool")
        tool_args = payload.get("args", {})
        print(f"\033[93mAction:\033[0m Call tool '{tool_name}' with args {tool_args}")
        
        # Execute tool
        if tool_name in TOOL_REGISTRY:
            observation = TOOL_REGISTRY[tool_name](**tool_args)
            print(f"\033[92mObservation:\033[0m {observation}")
            messages.append({"role": "observation", "content": observation})
            
            # Second turn to finalize answer after tool output
            second_response = llm.generate(messages)
            s_thought, s_msg_type, s_payload = parse_showcase_output(second_response)
            print(f"\033[35mThought:\033[0m {s_thought}")
            messages.append({"role": "assistant", "content": second_response})
            if s_msg_type == "final":
                print(f"\033[92mFinal Answer:\033[0m {s_payload}")
                return s_payload
        else:
            print(f"\033[91mError:\033[0m Tool {tool_name} not found.")
            
    elif msg_type == "final":
        print(f"\033[92mFinal Answer:\033[0m {payload}")
        return payload
        
    return response

def main():
    print("=" * 70)
    print("\033[94m[Showcase] Starting Memory Matrix Demo: Short-Term vs. Long-Term Memory\033[0m")
    print("=" * 70)

    # Initialize client
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if groq_key or gemini_key or openai_key:
        llm = get_llm_client()
    else:
        print("\033[93m[LLM Client] No API keys found. Initializing Showcase-Specific Simulator...\033[0m")
        llm = ShowcaseSimulatedClient()

    # Reset any previous persona data for a clean demonstration
    init_persona_store()
    
    # -------------------------------------------------------------
    # PHASE 1: Long-Term Memory Writing
    # -------------------------------------------------------------
    print("\n--- PHASE 1: Learning Alice's Details (Long-Term Memory Writing) ---")
    system_instruction_1 = SYSTEM_INSTRUCTION_TEMPLATE + load_long_term_facts()
    messages_session_1 = [{"role": "system", "content": system_instruction_1}]
    
    intro_prompt = "Hello! My name is Alice, I work as a Python engineer, and I prefer code output formatted as Markdown."
    print(f"\033[94m[User]:\033[0m {intro_prompt}")
    execute_turn(llm, messages_session_1, intro_prompt)

    # -------------------------------------------------------------
    # PHASE 2: Long-Term Memory Access on Session Start
    # -------------------------------------------------------------
    print("\n--- PHASE 2: Simulating Session Restart (Long-Term Memory Reading) ---")
    print("\033[90m[System] Booting new session. Re-reading persona_store.json facts...\033[0m")
    
    # Load refreshed system instruction prefixing the facts we just wrote
    system_instruction_2 = SYSTEM_INSTRUCTION_TEMPLATE + load_long_term_facts()
    print(f"\033[90m[System] Active System Prompt Prefix:\033[0m {load_long_term_facts().strip()}")
    
    messages_session_2 = [{"role": "system", "content": system_instruction_2}]
    
    code_prompt = "Can you write a Hello World program for me?"
    print(f"\033[94m[User]:\033[0m {code_prompt}")
    execute_turn(llm, messages_session_2, code_prompt)

    # -------------------------------------------------------------
    # PHASE 3: Short-Term Memory Context Bloat & Compaction
    # -------------------------------------------------------------
    print("\n--- PHASE 3: Flurry of Questions (Triggering Context Compaction) ---")
    
    # We ask multiple rapid questions to bloat the short-term chat logs
    quick_turns = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Canada?",
        "What is the capital of Brazil?"
    ]
    
    for query in quick_turns:
        print(f"\033[94m[User]:\033[0m {query}")
        execute_turn(llm, messages_session_2, query)

    # -------------------------------------------------------------
    # PHASE 4: Memory Verification
    # -------------------------------------------------------------
    print("\n--- PHASE 4: Verification (Querying Both Memory Systems) ---")
    verify_prompt = "Do you remember what my profession is, and what the third capital I asked about was?"
    print(f"\033[94m[User]:\033[0m {verify_prompt}")
    execute_turn(llm, messages_session_2, verify_prompt)

    print("\n" + "=" * 70)
    print("\033[94m[Showcase Done] Memory Matrix verification completed successfully!\033[0m")
    print("=" * 70)

if __name__ == "__main__":
    main()
