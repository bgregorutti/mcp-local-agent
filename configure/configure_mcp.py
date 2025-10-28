#!/usr/bin/env python3
"""
Configure MCP server for local agent delegation in Claude Code
"""

import json
import os
import shutil
from pathlib import Path

def configure_mcp_server():
    """Add local agent MCP server to Claude Code configuration"""

    claude_config = Path.home() / ".claude.json"

    if not claude_config.exists():
        print(f"❌ Claude config not found at {claude_config}")
        return False

    # Backup existing config
    backup_path = claude_config.with_suffix('.json.backup')
    shutil.copy(claude_config, backup_path)
    print(f"✅ Backed up config to {backup_path}")

    # Load config
    with open(claude_config, 'r') as f:
        config = json.load(f)

    # Get current project path
    current_project = os.getcwd()

    # MCP server configuration
    mcp_server_config = {
        "local-agents": {
            "command": "python",
            "args": [str(Path(__file__).parent / "local_agent_mcp_server.py")],
            "env": {
                "BACKEND_TYPE": "lmstudio",
                "BACKEND_URL": "http://localhost:1234/v1"
            },
            "disabled": False
        }
    }

    # Add to current project configuration
    if 'projects' not in config:
        config['projects'] = {}

    if current_project not in config['projects']:
        config['projects'][current_project] = {
            "allowedTools": [],
            "mcpServers": {}
        }

    # Ensure mcpServers exists
    if 'mcpServers' not in config['projects'][current_project]:
        config['projects'][current_project]['mcpServers'] = {}

    # Add our MCP server
    config['projects'][current_project]['mcpServers'].update(mcp_server_config)

    # Write back
    with open(claude_config, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ MCP server configured for project: {current_project}")
    print(f"\n📋 Added MCP server: 'local-agents'")
    print(f"   - Backend: LM Studio")
    print(f"   - URL: http://localhost:1234/v1")
    print(f"   - Server: {Path(__file__).parent / 'local_agent_mcp_server.py'}")

    print(f"\n🔄 Next steps:")
    print(f"   1. Restart Claude Code (close and reopen terminal/VS Code)")
    print(f"   2. Verify MCP tools are available with /mcp command")
    print(f"   3. Check for tools: execute_with_feedback_loop, execute_with_local_model, etc.")

    return True

if __name__ == "__main__":
    print("🚀 Configuring Local Agent MCP Server for Claude Code\n")
    print("=" * 60)

    success = configure_mcp_server()

    if success:
        print("\n" + "=" * 60)
        print("✅ Configuration complete!")
        print("\nℹ️  If you need to revert, your backup is at:")
        print(f"   ~/.claude.json.backup")
    else:
        print("\n❌ Configuration failed")
        exit(1)
