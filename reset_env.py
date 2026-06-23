import os
import json

def reset_environment():
    print("=" * 60)
    print("\033[94m[Reset] Cleaning up Agentic AI Demo Environment...\033[0m")
    print("=" * 60)

    # Files to remove
    files_to_remove = ["inventory.db", "agent_state.db"]
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                print(f"\033[92m[Deleted]\033[0m {file_name} successfully removed.")
            except Exception as e:
                print(f"\033[91m[Error]\033[0m Could not delete {file_name}: {e}")
        else:
            print(f"\033[90m[Skipped]\033[0m {file_name} does not exist.")

    # Reset persona store to defaults
    persona_path = "persona_store.json"
    default_store = {
        "user_preferences": [
            "User prefers database index optimizations to be verified afterward."
        ]
    }
    try:
        with open(persona_path, "w") as f:
            json.dump(default_store, f, indent=2)
        print(f"\033[92m[Reset]\033[0m {persona_path} restored to default preferences.")
    except Exception as e:
        print(f"\033[91m[Error]\033[0m Could not reset {persona_path}: {e}")

    print("=" * 60)
    print("\033[94m[Reset] Environment reset complete. Ready for a clean run!\033[0m")
    print("=" * 60)

if __name__ == "__main__":
    reset_environment()
