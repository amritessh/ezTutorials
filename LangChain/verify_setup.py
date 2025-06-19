import os
from dotenv import load_dotenv

load_dotenv()

def verify_setup():
    required_key = ["GEMINI_API_KEY"]

    for key in required_key:
        if not os.getenv(key):
            print(f"Error: {key} is not set")
            return False
        return True

if __name__ == "__main__":
    if verify_setup():
        print("Setup verified")
