import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class BaseLLMClient:
    """Base interface for LLM client wrappers, designed to match LangChain style."""
    def generate(self, messages: list) -> str:
        raise NotImplementedError("Subclasses must implement generate().")

class GroqClient(BaseLLMClient):
    """Client for Groq LPU API (OpenAI-compatible endpoint)."""
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, messages: list) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Standardize message format to openai-compliant roles
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            # OpenAI API only supports system, user, assistant, tool. 
            # If the role is 'observation', map it to 'user' for safety.
            if role == "observation" or role == "tool":
                formatted_messages.append({"role": "user", "content": f"Observation: {msg.get('content')}"})
            else:
                formatted_messages.append({"role": role, "content": msg.get("content")})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": 0.0
        }
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Groq API: {str(e)}"

class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    def generate(self, messages: list) -> str:
        headers = {"Content-Type": "application/json"}
        
        # Convert message history to Gemini API format
        # Gemini expects a systemInstruction for system, and contents: [{'role': 'user'|'model', 'parts': [{'text': '...'}]}]
        contents = []
        system_instruction = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role in ("observation", "tool"):
                contents.append({"role": "user", "parts": [{"text": f"Observation: {content}"}]})

        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI Chat Completions API."""
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.openai.com/v1/chat/completions"

    def generate(self, messages: list) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            if role in ("observation", "tool"):
                formatted_messages.append({"role": "user", "content": f"Observation: {msg.get('content')}"})
            else:
                formatted_messages.append({"role": role, "content": msg.get("content")})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": 0.0
        }
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

class SimulatedClient(BaseLLMClient):
    """Simulated offline LLM client for keyless presentation runs."""
    def generate(self, messages: list) -> str:
        # Check if the prompt asks for summarization
        last_msg = messages[-1].get("content", "")
        
        # Check if this is a summarization request
        is_summary_request = any(
            x in last_msg.lower() for x in ["summarize the conversation", "compress history", "summarize below", "summary of the interaction"]
        )
        if is_summary_request:
            return (
                "SUMMARY: The user initiated a transaction/database operation. "
                "The agent analyzed the requirements and resolved it step-by-step using available tools."
            )

        # Reconstruct the sequence of messages to decide the state
        # Find the original goal (usually in the first user message)
        goal = ""
        for msg in messages:
            if msg.get("role") == "user" and not msg.get("content", "").startswith("Observation:"):
                goal = msg.get("content", "")
                break
                
        # Count observations
        observations = [msg for msg in messages if msg.get("role") in ("observation", "tool") or msg.get("content", "").startswith("Observation:")]

        # SIMULATION logic based on the user's goal
        goal_lower = goal.lower()
        
        if "order" in goal_lower:
            # Demo 1 flow
            if len(observations) == 0:
                return (
                    "Thought: The user is asking for the status of order ID 12345. I need to call the order status tool to fetch this information.\n"
                    "Action: {\"tool\": \"get_order_status\", \"args\": {\"order_id\": \"12345\"}}"
                )
            else:
                # Observation received
                obs_content = observations[-1].get("content") or observations[-1].get("content", "")
                return (
                    f"Thought: I have received the order status update: '{obs_content}'. I can now formulate the final answer to the user.\n"
                    f"Final Answer: The status of order ID 12345 is Shipped. It is currently in transit and expected to be delivered on 2026-06-25."
                )

        elif "database" in goal_lower or "query" in goal_lower or "inventory" in goal_lower:
            # Demo 2 flow
            if len(observations) == 0:
                return (
                    "Thought: The user wants to query the inventory database for a Laptop. I will use the query_database tool via the MCP server interface.\n"
                    "Action: {\"tool\": \"query_database\", \"args\": {\"query\": \"SELECT * FROM inventory WHERE item_name = 'Laptop';\"}}"
                )
            else:
                obs_content = observations[-1].get("content", "")
                return (
                    f"Thought: The MCP database tool returned the raw database records: {obs_content}. I have the stock levels and can answer the user.\n"
                    "Final Answer: There are currently 15 Laptops in stock in the inventory database (stored in bin A4)."
                )

        elif "optimize" in goal_lower or "index" in goal_lower or "mutation" in goal_lower:
            # Demo 3 flow (checkpointing and HITL)
            if len(observations) == 0:
                return (
                    "Thought: The user wants to optimize the database index. Creating an index is a write operation (mutation) on the database. "
                    "I will generate the query to create a new index idx_item_name on the inventory table.\n"
                    "Action: {\"tool\": \"db_mutation\", \"args\": {\"mutation_query\": \"CREATE INDEX idx_item_name ON inventory(item_name);\"}}"
                )
            else:
                obs_content = observations[-1].get("content", "")
                return (
                    f"Thought: The index mutation has successfully completed, and the tool returned: '{obs_content}'. "
                    "The optimization is done.\n"
                    "Final Answer: The index idx_item_name has been successfully created on the inventory table. Future queries will run faster."
                )

        # Fallback response if message doesn't match predefined demo paths
        return (
            "Thought: I need to help the user but no specific demo path was matched. I will provide a final answer directly.\n"
            "Final Answer: I am running in Simulated Mode. Please check my instructions to run Demo 1, Demo 2, or Demo 3."
        )

def get_llm_client() -> BaseLLMClient:
    """Helper function to load the correct client based on available environment variables."""
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        print("\033[92m[LLM Client] Found GROQ_API_KEY. Initializing real Groq client (llama-3.3-70b-versatile)...\033[0m")
        return GroqClient(api_key=groq_key)
    elif gemini_key:
        print("\033[92m[LLM Client] Found GEMINI_API_KEY. Initializing real Gemini client (gemini-2.5-flash)...\033[0m")
        return GeminiClient(api_key=gemini_key)
    elif openai_key:
        print("\033[92m[LLM Client] Found OPENAI_API_KEY. Initializing real OpenAI client (gpt-4o-mini)...\033[0m")
        return OpenAIClient(api_key=openai_key)
    else:
        print("\033[93m[LLM Client] No API keys found. Initializing keyless Simulated LLM Client (ideal for presentation demo runs)...\033[0m")
        return SimulatedClient()
