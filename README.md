# Local Agent MCP Server - Hybrid Intelligence System

This MCP server enables Claude Code (Sonnet) to intelligently delegate work to local models while maintaining quality through iterative feedback loops. Save 60-70% on API costs while maintaining quality.

## ✅ Status

**Ready to use globally** - Works in all projects after restart

```
✅ Quick Delegation Test - PASSED (30s)
✅ Feedback Loop Test - PASSED (2 iterations, perfect result)
✅ Parallel Delegation Test - PASSED (3 tasks simultaneously)

System Status: 🟢 GLOBALLY OPERATIONAL
```

## 🎯 Architecture

```
User Request → Claude Code (Sonnet 4.5) → Analyzes & Delegates
                                          ↓
                                    MCP Server
                                          ↓
                            Local Model (LM Studio/Ollama)
                                          ↓
                        Generated Code ← Review & Feedback Loop
                                          ↓
                                   Final Output ✅
```

**Cost Savings:** 60-70% | **Quality:** Maintained | **Speed:** Faster

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install MCP SDK
pip install mcp

# Optional: Install Ollama support (if using Ollama)
pip install ollama
```

### 2. Start Your Local Model Backend

**Option A: LM Studio (Recommended)**
```bash
# 1. Download and launch LM Studio
# 2. Load a coding model (e.g., Qwen 2.5 Coder 7B)
# 3. Start the local server (default: http://localhost:1234)
```

**Option B: Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-coder:6.7b
ollama serve
```

### 3. Configure Claude Code

Run the configuration script to set up globally (works in all projects):

```bash
python configure_mcp_global.py
```

Or configure for current project only:

```bash
python configure_mcp.py
```

This adds the MCP server to `~/.claude.json` with the correct path and settings.

### 4. Set Up Project Guidelines (Recommended)

Copy the template to your project root and customize it:

```bash
# For this project
cp CLAUDE-template.md CLAUDE.md

# Edit the file to match your project
# - Set language, test command, code style
# - Adjust delegation preferences
```

Or create a global default for all projects:

```bash
# Create global guidelines
mkdir -p ~/.claude
cp CLAUDE-template.md ~/.claude/CLAUDE.md
```

**Why this matters:** While the MCP server configuration makes the tools **available**, the CLAUDE.md file tells Claude Code **when and how to use them** automatically. Think of it as:
- MCP config = Installing the tool
- CLAUDE.md = Instructions on when to use it

### 5. Restart Claude Code

Close and reopen VS Code/Claude Code completely. The MCP tools will now be available.

## 🛠 Configuration Options

### Environment Variables

**Backend Configuration:**

- `BACKEND_TYPE`: Choose backend
  - `lmstudio` (default) - LM Studio OpenAI-compatible API
  - `ollama` - Ollama
  - `openai-compatible` - Any OpenAI-compatible API

- `BACKEND_URL`: Custom backend URL
  - Default for LM Studio: `http://localhost:1234/v1`
  - Default for Ollama: `http://localhost:11434`

**Pricing Configuration (for accurate cost tracking):**

- `REMOTE_INPUT_COST_PER_1K`: Cost per 1K input tokens (default: 0.003)
- `REMOTE_OUTPUT_COST_PER_1K`: Cost per 1K output tokens (default: 0.015)
- `LOCAL_COST_PER_1K`: Cost per 1K tokens for local model (default: 0.0)

Example for Claude Sonnet 4.5:
```bash
export REMOTE_INPUT_COST_PER_1K=0.003   # $3 per million tokens
export REMOTE_OUTPUT_COST_PER_1K=0.015  # $15 per million tokens
export LOCAL_COST_PER_1K=0.0            # Local models are free
```

### Per-Task Configuration

You can also specify backend per task:

```python
# In tool calls
{
  "backend_type": "lmstudio",
  "backend_url": "http://localhost:1234/v1",
  "model": "qwen2.5-coder:7b"
}
```

## 📚 Available Tools

### 1. `execute_with_feedback_loop` ⭐ (Primary Tool)

**Use for:** Most coding tasks where quality matters

**Workflow:**
1. Local model generates code
2. Returns to Claude for review
3. Claude provides specific feedback
4. Local model iterates (max 3 times)
5. Claude approves final output
6. **Comprehensive statistics report generated**

**Example:**
```json
{
  "task_id": "api-endpoints-001",
  "task_type": "backend",
  "prompt": "Create REST API endpoints for user management",
  "system_prompt": "You are an expert Python FastAPI developer",
  "model": "qwen2.5-coder:7b",
  "quality_criteria": [
    "Include error handling",
    "Add type hints",
    "Follow REST best practices"
  ]
}
```

**Statistics Included:**
- Token usage (sent/received) for local model
- Estimated token usage for remote model (review)
- Time spent on each iteration
- Cost analysis and savings
- Iteration breakdown

### 2. `execute_with_local_model` (Quick Delegation)

**Use for:** Simple tasks that don't need review

**Example:**
```json
{
  "prompt": "Generate boilerplate for a Python dataclass with name, email, age fields",
  "model": "qwen2.5-coder:7b"
}
```

**Statistics Included:**
- Token usage (sent/received)
- Duration
- Model used

### 3. `provide_feedback` (Review Tool)

**Called by Claude after reviewing output**

**Example:**
```json
{
  "task_id": "api-endpoints-001",
  "issues": [
    "Missing input validation for email format",
    "No rate limiting on endpoints"
  ],
  "suggestions": [
    "Add email regex validation using pydantic",
    "Implement rate limiting with slowapi"
  ],
  "approve": false
}
```

### 4. `compare_iterations` (Analysis Tool)

View all iterations for a task to see improvement over time. Now includes statistics for the task.

### 5. `get_statistics_summary` 📊 (Reporting Tool)

Get comprehensive cumulative statistics across all delegation tasks. See the Statistics & Performance Tracking section below for details.

## 🎓 Decision Matrix for Claude Code

### Use `execute_with_local_model` (Quick):
- ✅ Boilerplate code generation
- ✅ Simple formatting
- ✅ Basic documentation
- ✅ Straightforward refactoring

### Use `execute_with_feedback_loop` ⭐ (Best Value):
- ✅ API endpoints
- ✅ Test generation
- ✅ UI components
- ✅ Database queries
- ✅ Most "real work"

### Handle Yourself (Remote):
- 🎯 Architecture decisions
- 🎯 Security-critical code
- 🎯 Complex algorithms
- 🎯 Novel/ambiguous requirements
- 🎯 Final integration & review

## 📊 Statistics & Performance Tracking

### Real-Time Statistics Reporting

**NEW:** Statistics are now reported on **every tool call**, not just at completion. This provides complete visibility into delegation costs and savings.

### Available Statistics Tools

#### 5. `get_statistics_summary` 📊 (New!)

Get cumulative statistics across all delegation tasks:

```json
{
  "cumulative_statistics": {
    "total_tasks": 47,
    "task_status_breakdown": {
      "successful": 42,
      "failed": 3,
      "approved": 38,
      "max_iterations_reached": 2
    },
    "cost_analysis": {
      "total_savings_usd": 3.71,
      "savings_percent": 86.9
    }
  }
}
```

### Comprehensive Statistics Report

When a task is completed (approved or max iterations reached), the system generates a detailed statistics report:

```json
{
  "task_id": "api-endpoints-001",
  "status": "approved",
  "summary": {
    "total_duration_seconds": 45.3,
    "total_iterations": 2,
    "feedback_rounds": 1,
    "average_iteration_time_seconds": 22.65
  },
  "local_model_usage": {
    "model": "qwen/qwen3-coder-30b",
    "api_calls": 2,
    "tokens": {
      "sent": 1250,
      "received": 3840,
      "total": 5090
    },
    "time_seconds": 45.3
  },
  "remote_model_usage": {
    "estimated_tokens_for_review": 3840,
    "estimated_cost_usd": 0.0115
  },
  "cost_analysis": {
    "local_model_cost_usd": 0.00,
    "estimated_cost_if_fully_remote_usd": 0.0613,
    "actual_cost_usd": 0.0115,
    "savings_usd": 0.0498,
    "savings_percent": 81.2
  },
  "iteration_breakdown": [
    {
      "iteration": 1,
      "duration": 23.1,
      "tokens_sent": 625,
      "tokens_received": 1920,
      "timestamp": "2025-01-15T10:30:00"
    },
    {
      "iteration": 2,
      "duration": 22.2,
      "tokens_sent": 625,
      "tokens_received": 1920,
      "timestamp": "2025-01-15T10:30:45"
    }
  ]
}
```

### Key Metrics Tracked

1. **Token Usage**
   - Tokens sent to local model (prompts + context)
   - Tokens received from local model (generated code)
   - Estimated tokens for remote review

2. **Time Metrics**
   - Total duration from start to completion
   - Per-iteration timing
   - Average iteration time

3. **Cost Analysis**
   - Local model cost (configurable, default: $0.00)
   - Estimated cost if task was done fully remote
   - Actual cost (remote review + local usage)
   - Savings in USD and percentage
   - Pricing configuration used

4. **Efficiency Metrics**
   - Number of iterations needed
   - Feedback rounds
   - Iteration breakdown with detailed stats

5. **Persistence**
   - All statistics automatically saved to `.mcp_stats.json`
   - Data persists across Claude Code sessions
   - Includes both active and archived tasks (100 task limit in memory)

### How This Helps You (The Orchestrator)

When you receive the statistics report, you can:
- **Evaluate effectiveness**: See actual cost savings per task
- **Optimize delegation**: Learn which tasks work best with local models
- **Track performance**: Monitor if local models are improving over time
- **Report to users**: Show concrete savings and efficiency gains

## 📊 Expected Outcomes

**Cost Savings:** 60-70% reduction in API costs
- Local models handle routine work (free)
- Remote only for decision-making and review
- **Real-time tracking** of actual savings

**Quality:** Maintained or improved
- Iterative feedback ensures standards
- Claude's review catches issues
- **Iteration metrics** show quality progression

**Speed:** Faster for many tasks
- Local models respond instantly
- Parallel work on multiple tasks
- **Timing data** shows actual performance

## 🔧 Troubleshooting

### Connection Errors

```
"LM Studio connection error"
```

**Solution:**
1. Ensure LM Studio is running
2. Check a model is loaded
3. Verify the server is started (green indicator)
4. Test: `curl http://localhost:1234/v1/models`

### Model Not Found

```
"Model 'xyz' not found"
```

**Solution:**
- Use exact model name as shown in LM Studio
- Common names: `qwen2.5-coder:7b`, `deepseek-coder-v2`

### MCP Server Not Starting

**Solution:**
1. Check Python path in MCP config
2. Verify dependencies: `pip list | grep mcp`
3. Check server logs in Claude Code output panel

## 🎯 Example Workflow

```
User: "Build a user authentication system"

Claude Code (Sonnet):
1. Design architecture (MYSELF - complex)
2. Generate user model boilerplate (LOCAL - quick)
   → execute_with_local_model
3. Implement password hashing (MYSELF - security critical)
4. Create CRUD endpoints (LOCAL + FEEDBACK) ⭐
   → execute_with_feedback_loop
   → Local generates
   → I review: "Missing input validation"
   → provide_feedback
   → Local fixes
   → I approve ✅
5. Generate tests (LOCAL + FEEDBACK)
   → Same iterative process
6. Final integration (MYSELF - critical)

Result: 70% cost savings, high quality maintained
```

## 🌍 Global vs Project-Specific Configuration

### Global Configuration (Recommended)
- **Works in ALL projects** automatically
- Configure once with `configure_mcp_global.py`
- Same settings everywhere
- Best for most users

### Project-Specific Configuration
- Works only in specific project folders
- Configure with `configure_mcp.py`
- Different settings per project
- Use when you need fine-grained control

### Switching Between Configurations
```bash
# Switch to global
python configure_mcp_global.py

# Switch to project-specific
cd /path/to/project
python configure_mcp.py
```

## 🔬 Advanced Features

### Feedback Analysis

The server tracks common issues and can enhance prompts:

```python
# Automatically learns from feedback patterns
# Preemptively warns local model about common mistakes
```

### Custom System Prompts

Tailor the local model's behavior:

```json
{
  "system_prompt": "You are a senior Python developer specializing in FastAPI. Always include comprehensive error handling and type hints. Follow PEP 8 strictly."
}
```

## 📝 Recommended Models

**For LM Studio:**
- **Qwen 2.5 Coder 7B** - Best balance (recommended)
- **DeepSeek Coder V2 16B** - Highest quality (needs more RAM)
- **CodeLlama 13B** - Good alternative

**For Ollama:**
- `deepseek-coder:6.7b` - Fast and capable
- `codellama:13b` - Alternative option

## 🎓 Tips for Claude Code

When deciding to delegate:

1. **Start simple** - Use quick delegation for obvious boilerplate
2. **Use feedback loops liberally** - They provide best cost/quality balance
3. **Review thoroughly** - Be specific in feedback
4. **Iterate up to 3 times** - If not fixed by then, handle yourself
5. **Learn patterns** - Track which tasks work well with local models

## 📊 Success Metrics

Track your hybrid workflow effectiveness:
- **Delegation rate**: % of tasks delegated (target: 60-80%)
- **Approval rate**: % approved on first iteration (target: 30-40%)
- **Iteration average**: How many rounds needed (target: 1.5-2.0)
- **Cost savings**: Compare API usage before/after (target: 60-70%)

## 🚀 Testing & Verification

### Test Connection
```bash
python test_connection.py
```

### Test Full Workflow
```bash
python test_hybrid_workflow.py
```

### Verify MCP Tools Loaded
After restarting Claude Code, ask: "What MCP tools do you have?"

Expected response: 4 delegation tools listed

## 📚 Additional Resources

For more detailed information, see:
- `DELEGATION_STRATEGY.md` - Claude Code's decision framework for delegation
- `test_hybrid_workflow.py` - Example test cases and workflows
- `mcp_config_example.json` - Manual configuration template

---

**Questions?** Check Claude Code docs or MCP documentation.
