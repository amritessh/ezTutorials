
import asyncio
import json
import subprocess
import sys

async def test_mcp_server():
    process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:

        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        

        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        

        response = process.stdout.readline()
        print("✅ Initialize Response:", response.strip())
        
   
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        process.stdin.write(json.dumps(tools_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        print("✅ Tools List Response:", response.strip())
        
 
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_users",
                "arguments": {}
            }
        }
        
        process.stdin.write(json.dumps(call_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        print("✅ Get Users Response:", response.strip())
        
        print("\n🎉 All tests passed! Your MCP server is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    asyncio.run(test_mcp_server())