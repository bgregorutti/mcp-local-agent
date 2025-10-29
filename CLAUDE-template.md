# Claude Code Guidelines

This file provides guidance to Claude Code on how to work efficiently in this project by leveraging local model delegation.

## Delegation Strategy

### Automatic Delegation Rules

* **ALWAYS** use `mcp-local-agents` tools to delegate code generation tasks to local LLM
* Prioritize local model delegation to minimize API costs while maintaining quality
* Use remote model (Claude Sonnet) for complex reasoning, architecture decisions, and code review

### Tool Selection

#### Use `execute_with_local_model` for:
- Boilerplate code generation
- Simple refactoring and formatting
- Documentation generation
- Straightforward implementations with clear requirements
- Unit test generation for existing code

#### Use `execute_with_feedback_loop` for:
- Medium complexity features requiring quality control
- Bug fixes that need careful testing
- Implementations with specific quality criteria
- Code that may require iteration to get right
- Tasks where you want to review and provide feedback

#### Use Remote Model (Sonnet) for:
- Complex architectural decisions and design
- Code review and quality assessment
- Planning multi-step implementations
- Debugging complex issues
- Tasks requiring deep reasoning about system behavior

### Cost Optimization

* Check `get_statistics_summary` periodically to monitor delegation effectiveness
* Target: 80%+ cost savings through local model delegation
* Review statistics before and after major features

## Development Workflow

### Version Control

* Commit changes in the current branch if repository exists
* Write clear, descriptive commit messages
* Include delegation statistics in commit messages for major features
  - Example: "feat: add user authentication (delegated, saved $0.15)"

### Code Quality

* All delegated code must pass existing tests
* Review local model outputs before approval
* Provide specific, actionable feedback when iterating
* Follow existing code style and patterns

### Testing

* Run tests after significant changes
* Generate tests alongside new code when appropriate
* Verify delegated code works as expected

## Project-Specific Configuration

**Language:** [Fill in: Python, TypeScript, etc.]

**Local Model Backend:** LM Studio with qwen2.5-coder:7b (default)

**Test Command:** [Fill in: pytest, npm test, etc.]

**Code Style:** [Fill in: PEP 8, Airbnb, etc.]

## Pricing Configuration (Optional)

To customize cost calculations, set environment variables:

```bash
export REMOTE_INPUT_COST_PER_1K=0.003   # Claude Sonnet input pricing
export REMOTE_OUTPUT_COST_PER_1K=0.015  # Claude Sonnet output pricing
export LOCAL_COST_PER_1K=0.0            # Local models are free
```

## Example Delegation Workflow

1. **Receive task** from user (e.g., "add authentication middleware")
2. **Classify complexity**: Simple/Medium/Complex
3. **If Simple/Medium**: Use appropriate delegation tool
4. **Review output**: Check quality, run tests
5. **If issues**: Provide feedback for iteration
6. **If approved**: Commit with statistics
7. **Check savings**: Periodically use `get_statistics_summary`

## Notes

* This file is read by Claude Code to guide its behavior
* Update these guidelines as project needs evolve
* Statistics are saved to `.mcp_stats.json` (add to .gitignore if desired)
