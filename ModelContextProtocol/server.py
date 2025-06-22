#!/usr/bin/env python3
import asyncio
import json
import sqlite3
from typing import Any, Dict, List

import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

class QuickMCPServer:
    def __init__(self):
        self.server = Server("quickstart-server")
        self.setup_database()
        self.setup_handlers()
    
    def setup_database(self):
        """Create a simple database with sample data."""
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        
        cursor = self.db.cursor()
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL
            )
        """)
        
        # Add sample data
        sample_users = [
            ("Alice", "alice@example.com"),
            ("Bob", "bob@example.com"),
            ("Carol", "carol@example.com")
        ]
        cursor.executemany("INSERT INTO users (name, email) VALUES (?, ?)", sample_users)
        self.db.commit()
    
    def setup_handlers(self):
        """Setup MCP request handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="get_users",
                    description="Get all users from database",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="add_user",
                    description="Add a new user",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"}
                        },
                        "required": ["name", "email"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[types.TextContent]:
            if name == "get_users":
                cursor = self.db.cursor()
                cursor.execute("SELECT * FROM users")
                users = [dict(row) for row in cursor.fetchall()]
                return [types.TextContent(
                    type="text", 
                    text=json.dumps(users, indent=2)
                )]
            
            elif name == "add_user":
                name_val = arguments.get("name")
                email_val = arguments.get("email")
                
                cursor = self.db.cursor()
                cursor.execute(
                    "INSERT INTO users (name, email) VALUES (?, ?)",
                    (name_val, email_val)
                )
                self.db.commit()
                
                return [types.TextContent(
                    type="text",
                    text=f"Added user: {name_val} ({email_val})"
                )]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        @self.server.list_resources()
        async def list_resources() -> List[types.Resource]:
            return [
                types.Resource(
                    uri="sqlite://users",
                    name="All users",
                    description="Complete user database",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            if uri == "sqlite://users":
                cursor = self.db.cursor()
                cursor.execute("SELECT * FROM users")
                users = [dict(row) for row in cursor.fetchall()]
                return json.dumps(users, indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="quickstart-server",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {},
                        "resources": {}
                    }
                ),
            )

# Run the server
if __name__ == "__main__":
    server = QuickMCPServer()
    asyncio.run(server.run())