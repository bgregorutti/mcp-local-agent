#!/usr/bin/env python3
"""
Configure MCP server globally for ALL projects in Claude Code
"""

import json
import shutil
from pathlib import Path

def configure_mcp_global():
    """Add local agent MCP server to Claude Code global configuration"""

    claude_config = Path.home() / ".claude.json"

    if not claude_config.exists():
        print(f"❌ Claude config not found at {claude_config}")
        return False

    # Backup existing config
    backup_path = claude_config.with_suffix('.json.backup-global')
    shutil.copy(claude_config, backup_path)
    print(f"✅ Backed up config to {backup_path}")

    # Load config
    with open(claude_config, 'r') as f:
        config = json.load(f)

    # MCP server configuration
    mcp_server_config = {
        "local-agents": {
            "command": "python",
            "args": [str(Path(__file__).parent.parent / "local_agent_mcp_server.py")],
            "env": {
                "BACKEND_TYPE": "lmstudio",
                "BACKEND_URL": "http://localhost:1234/v1"
            },
            "disabled": False
        }
    }

    # Add to GLOBAL mcpServers (not project-specific)
    if 'mcpServers' not in config:
        config['mcpServers'] = {}

    # Add our MCP server globally
    config['mcpServers'].update(mcp_server_config)

    print(f"\n✅ MCP server configured GLOBALLY for all projects")

    # Write back
    with open(claude_config, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n📋 Added GLOBAL MCP server: 'local-agents'")
    print(f"   - Scope: ALL PROJECTS")
    print(f"   - Backend: LM Studio")
    print(f"   - URL: http://localhost:1234/v1")
    print(f"   - Server: {Path(__file__).parent.parent / 'local_agent_mcp_server.py'}")

    print(f"\n🌍 This MCP server will now be available in:")
    print(f"   ✅ Current project")
    print(f"   ✅ All future projects")
    print(f"   ✅ Any directory you work in")

    print(f"\n🔄 Next steps:")
    print(f"   1. Restart Claude Code (close and reopen terminal/VS Code)")
    print(f"   2. Navigate to ANY project")
    print(f"   3. Verify MCP tools are available with /mcp command")
    print(f"   4. Check for tools: execute_with_feedback_loop, execute_with_local_model, etc.")

    return True

if __name__ == "__main__":
    print("🌍 Configuring Local Agent MCP Server GLOBALLY\n")
    print("=" * 60)

    success = configure_mcp_global()

    if success:
        print("\n" + "=" * 60)
        print("✅ Global configuration complete!")
        print("\n💡 The MCP server is now available in ALL projects")
        print("\nℹ️  If you need to revert, your backup is at:")
        print(f"   ~/.claude.json.backup-global")
    else:
        print("\n❌ Configuration failed")
        exit(1)
