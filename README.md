# Local Agent MCP Server

Delegate code generation tasks from Claude Code to local models (LM Studio/Ollama) while maintaining quality through iterative feedback. **Save 60-90% on API costs.**

## Core Features

- 🎯 **Intelligent Delegation** - Claude Code automatically delegates tasks to local models
- 🔄 **Feedback Loop** - Iterative review and improvement (max 3 iterations)
- 📊 **Real-Time Statistics** - Track costs and savings on every delegation
- 💾 **Persistent Tracking** - Statistics saved across sessions
- ⚙️ **Configurable Pricing** - Accurate cost calculations for any model
- 🌍 **Global or Per-Project** - Works everywhere or specific projects

## How It Works

```
User Request → Claude Code (Sonnet) → Delegates Simple/Medium Tasks
                                     ↓
                               MCP Server
                                     ↓
                          Local Model (Free)
                                     ↓
                          Review & Iterate
                                     ↓
                              Final Output ✅
```

**Result:** Claude handles architecture and complex decisions. Local model handles code generation. You save money.

## Quick Start

### 1. Prerequisites

```bash
# Install MCP SDK
pip install mcp

# Start a local model backend (choose one):

# Option A: LM Studio (recommended)
# 1. Download from lmstudio.ai
# 2. Load Qwen 2.5 Coder 7B
# 3. Start server (http://localhost:1234)

# Option B: Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-coder:6.7b
ollama serve
```

### 2. Configure MCP Server

```bash
# Global (all projects)
python configure/configure_mcp_global.py

# Or project-specific
python configure/configure_mcp.py
```

This adds the MCP server to `~/.claude.json`.

### 3. Enable Automatic Delegation (Recommended)

```bash
# Copy template to your project
cp CLAUDE-template.md CLAUDE.md

# Or create global default
mkdir -p ~/.claude
cp CLAUDE-template.md ~/.claude/CLAUDE.md
```

**Important:**
- MCP config = Makes tools **available**
- CLAUDE.md = Tells Claude **when to use them**

Both are needed for automatic delegation.

### 4. Restart Claude Code

Close and reopen VS Code/Claude Code completely.

## Available Tools

### 1. `execute_with_feedback_loop` (Primary)
Best for most coding tasks. Local model generates → Claude reviews → iterates until approved.

```json
{
  "task_id": "add-auth-001",
  "task_type": "backend",
  "prompt": "Create user authentication endpoints",
  "quality_criteria": ["Include error handling", "Add type hints"]
}
```

### 2. `execute_with_local_model` (Quick)
For simple tasks that don't need review (boilerplate, formatting, docs).

```json
{
  "prompt": "Generate a Python dataclass for User with name, email, age"
}
```

### 3. `provide_feedback` (Review)
Claude uses this to provide specific feedback for iteration.

### 4. `compare_iterations` (Analysis)
View improvement across iterations with statistics.

### 5. `get_statistics_summary` (Reporting)
Get cumulative statistics across all delegation tasks.

```json
{
  "total_tasks": 47,
  "cost_analysis": {
    "total_savings_usd": 3.71,
    "savings_percent": 86.9
  }
}
```

## Configuration

### Environment Variables

**Backend:**
```bash
export BACKEND_TYPE=lmstudio  # or ollama, openai-compatible
export BACKEND_URL=http://localhost:1234/v1
```

**Pricing (for accurate cost tracking):**
```bash
export REMOTE_INPUT_COST_PER_1K=0.003   # Claude Sonnet input
export REMOTE_OUTPUT_COST_PER_1K=0.015  # Claude Sonnet output
export LOCAL_COST_PER_1K=0.0            # Local models are free
```

## Statistics & Cost Tracking

Every delegation returns comprehensive statistics:

```json
{
  "statistics": {
    "local_model_usage": {
      "tokens": {"sent": 1500, "received": 800},
      "time_seconds": 2.34
    },
    "cost_analysis": {
      "local_model_cost_usd": 0.00,
      "estimated_cost_if_fully_remote_usd": 0.0165,
      "actual_cost_usd": 0.0024,
      "savings_usd": 0.0141,
      "savings_percent": 85.5
    }
  }
}
```

Statistics are automatically saved to `.mcp_stats.json` and persist across sessions.

## Delegation Strategy

Claude Code should delegate:

**✅ To Local Model:**
- Boilerplate and simple code generation
- Refactoring and formatting
- Test generation
- API endpoints and CRUD operations
- Documentation

**🎯 Handle Yourself (Remote):**
- Architecture and design decisions
- Security-critical code
- Complex algorithms
- Novel/ambiguous requirements
- Final integration and review

**Target:** 60-80% of tasks delegated for 60-90% cost savings.

## Tips

1. **Use Feedback Loops** - Better quality than quick delegation
2. **Be Specific in Feedback** - "Add email validation" beats "improve validation"
3. **Check Statistics** - Use `get_statistics_summary` to track savings
4. **Iterate Up to 3 Times** - If not fixed, handle it yourself
5. **Customize CLAUDE.md** - Tailor delegation rules per project

## Troubleshooting

**"Connection error"**
- Ensure LM Studio/Ollama is running
- Check model is loaded
- Verify server URL: `curl http://localhost:1234/v1/models`

**"Tools not available"**
- Run config script: `python configure/configure_mcp_global.py`
- Restart Claude Code completely
- Check `~/.claude.json` contains "local-agents"

**"Not delegating automatically"**
- Create CLAUDE.md from template
- Restart Claude Code after adding CLAUDE.md

## File Structure

```
mcp-local-agent/
├── local_agent_mcp_server.py      # Main MCP server
├── configure/
│   ├── configure_mcp_global.py    # Global setup
│   └── configure_mcp.py           # Project setup
├── CLAUDE-template.md             # Template for project guidelines
├── README.md                      # This file
└── .mcp_stats.json               # Statistics (auto-generated)
```

## Advanced

### Custom System Prompts

```json
{
  "system_prompt": "You are a senior Python developer. Follow PEP 8 strictly. Always include comprehensive error handling and type hints."
}
```

### Global vs Project-Specific

- **Global:** `configure_mcp_global.py` → Works everywhere
- **Project:** `configure_mcp.py` → Only current project
- **Both supported:** Can override global with project settings

### Recommended Models

**LM Studio:**
- Qwen 2.5 Coder 7B (best balance)
- DeepSeek Coder V2 16B (highest quality)
- CodeLlama 13B (alternative)

**Ollama:**
- `deepseek-coder:6.7b`
- `codellama:13b`

## Documentation

- `CLAUDE-template.md` - Project guidelines template
- `docs/DELEGATION_STRATEGY.md` - Decision framework
- `test_hybrid_workflow.py` - Example workflows

## Support

- Issues: [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- MCP Docs: [MCP Documentation](https://modelcontextprotocol.io)

---

**Status:** 🟢 Production Ready | **Cost Savings:** 60-90% | **Quality:** Maintained
