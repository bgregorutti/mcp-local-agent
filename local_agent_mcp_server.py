from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json
import time
import os

from usage_tracker import StatisticsTracker
from backends import get_backend
from feedback import FeedbackLoop, FeedbackAnalyzer

app = Server("local-agent-server")


# Environment variables for stats tracking

# Remote model pricing (per 1K tokens)
REMOTE_INPUT_COST_PER_1K = float(os.getenv("REMOTE_INPUT_COST_PER_1K"))
REMOTE_OUTPUT_COST_PER_1K = float(os.getenv("REMOTE_OUTPUT_COST_PER_1K"))

# Local model pricing (per 1K tokens) - typically 0 for local models
LOCAL_COST_PER_1K = float(os.getenv("LOCAL_COST_PER_1K"))

# Global statistics tracker
stats_tracker = StatisticsTracker()


# Initialize backend (defaults to LM Studio)
BACKEND = get_backend(
    backend_type=os.getenv("BACKEND_TYPE"),
    base_url=os.getenv("BACKEND_URL")
)

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="execute_with_feedback_loop",
            description="""Execute task with local model in iterative feedback mode.

            This returns intermediate results that YOU (remote model) should review.
            You can then provide feedback and trigger another iteration.

            Workflow:
            1. Call this tool with task
            2. Review the output
            3. If issues found, call 'provide_feedback' with specific issues
            4. Local model will iterate with your feedback
            5. Repeat until satisfactory or max iterations reached

            Use this when you want local model to do the work but maintain quality control.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Unique task identifier for tracking"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["backend", "frontend", "testing", "refactoring", "documentation", "other"]
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task description"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt defining expertise (optional)"
                    },
                    "quality_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of quality requirements (e.g., 'include error handling', 'add type hints')"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g., 'qwen2.5-coder:7b' for LM Studio, 'deepseek-coder:6.7b' for Ollama)"
                    },
                    "backend_type": {
                        "type": "string",
                        "enum": ["lmstudio", "ollama", "openai-compatible"],
                        "description": "Backend to use (default: lmstudio)"
                    },
                    "backend_url": {
                        "type": "string",
                        "description": "Custom backend URL (optional, e.g., 'http://localhost:1234/v1')"
                    }
                },
                "required": ["task_id", "task_type", "prompt"]
            }
        ),

        Tool(
            name="execute_with_local_model",
            description="""Quick delegation to local model without feedback loop.

            Use this for simple tasks where you trust the output without review:
            - Boilerplate code generation
            - Simple formatting
            - Documentation generation
            - Straightforward refactoring

            For tasks requiring quality control, use 'execute_with_feedback_loop' instead.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task description"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt defining expertise (optional)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g., 'qwen2.5-coder:7b' for LM Studio)"
                    },
                    "backend_type": {
                        "type": "string",
                        "enum": ["lmstudio", "ollama", "openai-compatible"],
                        "description": "Backend to use (default: lmstudio)"
                    },
                    "backend_url": {
                        "type": "string",
                        "description": "Custom backend URL (optional)"
                    }
                },
                "required": ["prompt"]
            }
        ),
        
        Tool(
            name="provide_feedback",
            description="""Provide feedback on local model's output to trigger iteration.
            
            Call this after reviewing output from 'execute_with_feedback_loop'.
            The local model will attempt to address your feedback.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific issues found"
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific suggestions for improvement"
                    },
                    "approve": {
                        "type": "boolean",
                        "description": "Set to true if output is acceptable"
                    }
                },
                "required": ["task_id", "issues", "suggestions", "approve"]
            }
        ),
        
        Tool(
            name="compare_iterations",
            description="""Compare all iterations for a task to see improvement over time.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"}
                },
                "required": ["task_id"]
            }
        ),

        Tool(
            name="get_statistics_summary",
            description="""Get comprehensive statistics summary across all delegation tasks.

            Returns cumulative metrics including:
            - Total tasks and status breakdown (successful, failed, approved, etc.)
            - Total local model usage (tokens, calls, time)
            - Total remote model usage (tokens for review)
            - Cost analysis and savings (local vs remote)
            - Breakdown by task type
            - Aggregated metrics (iterations, feedback rounds)

            Use this to understand overall delegation efficiency and cost savings.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


# Store active tasks and their feedback loops
active_tasks = {}

@app.call_tool()
async def call_tool(name: str, arguments: dict):

    if name == "execute_with_local_model":
        # Quick delegation without feedback loop
        prompt = arguments["prompt"]
        system_prompt = arguments.get("system_prompt", "You are an expert software developer.")
        model = arguments.get("model", "qwen2.5-coder:7b")

        # Generate unique task_id for tracking (format: quick-timestamp-random)
        import random
        task_id = f"quick-{int(time.time())}-{random.randint(1000, 9999)}"

        # Initialize statistics tracking
        stats_tracker.init_task(task_id)

        # Get backend (custom or default)
        task_backend = BACKEND
        if arguments.get("backend_type") or arguments.get("backend_url"):
            task_backend = get_backend(
                backend_type=arguments.get("backend_type"),
                base_url=arguments.get("backend_url")
            )

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            output, tokens_sent, tokens_received, duration = await task_backend.generate(messages, model)

            # Track statistics globally
            stats_tracker.record_local_call(task_id, tokens_sent, tokens_received, duration, model)
            stats_tracker.record_iteration(task_id, 1, duration, tokens_sent, tokens_received)
            stats_tracker.finalize_task(task_id, status="completed")

            # Get comprehensive statistics report
            stats_report = stats_tracker.get_task_report(task_id)

            # Archive task and cleanup old tasks
            stats_tracker.archive_task(task_id)
            stats_tracker.cleanup_old_tasks()

            return [TextContent(
                type="text",
                text=json.dumps({
                    "task_id": task_id,
                    "status": "success",
                    "output": output,
                    "statistics": stats_report
                }, indent=2)
            )]
        except Exception as e:
            # Finalize with error status
            stats_tracker.finalize_task(task_id, status="error")

            # Get statistics report
            error_stats = stats_tracker.get_task_report(task_id)

            # Archive task and cleanup old tasks
            stats_tracker.archive_task(task_id)
            stats_tracker.cleanup_old_tasks()

            return [TextContent(
                type="text",
                text=json.dumps({
                    "task_id": task_id,
                    "status": "error",
                    "error": str(e),
                    "statistics": error_stats
                }, indent=2)
            )]

    elif name == "execute_with_feedback_loop":
        task_id = arguments["task_id"]

        # Initialize or continue task
        if task_id not in active_tasks:
            # Get backend for this task
            task_backend = BACKEND
            if arguments.get("backend_type") or arguments.get("backend_url"):
                task_backend = get_backend(
                    backend_type=arguments.get("backend_type"),
                    base_url=arguments.get("backend_url")
                )

            active_tasks[task_id] = {
                "task": arguments,
                "iterations": [],
                "feedback": [],
                "status": "in_progress",
                "feedback_loop": FeedbackLoop(task_backend, stats_tracker)
            }

            # Initialize statistics tracking
            stats_tracker.init_task(task_id)

        task_data = active_tasks[task_id]
        model = arguments.get("model", "qwen2.5-coder:7b")

        # Execute with local model (now with task_id for stats tracking)
        result = await task_data["feedback_loop"].execute_with_feedback(
            arguments,
            model,
            task_data["feedback"],
            task_id=task_id
        )

        # Handle error case
        if result["status"] == "error":
            # Finalize task with error status
            stats_tracker.finalize_task(task_id, status="error")

            # Get statistics even on error to track failed attempts
            error_stats = stats_tracker.get_task_report(task_id)

            # Archive task and cleanup old tasks
            stats_tracker.archive_task(task_id)
            stats_tracker.cleanup_old_tasks()

            return [TextContent(type="text", text=json.dumps(
                {
                    "task_id": task_id, "status": "error", "error": result.get("error"),
                    "message": "Local model execution failed. Check your backend connection.", "statistics": error_stats
                }, indent=2))]

        # Store iteration
        task_data["iterations"].append(result)

        # Get current statistics for real-time reporting
        current_stats = stats_tracker.get_current_stats(task_id)

        return [TextContent(
            type="text",
            text=json.dumps({
                "task_id": task_id,
                "iteration": result["iteration"],
                "status": result["status"],
                "output": result["output"],
                "message": f"Local model completed iteration {result['iteration']}. Please review and provide feedback.",
                "next_step": "Call 'provide_feedback' with your review",
                "statistics": current_stats
            }, indent=2)
        )]

    elif name == "provide_feedback":
        task_id = arguments["task_id"]

        if task_id not in active_tasks:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Task not found"})
            )]

        task_data = active_tasks[task_id]

        # Store feedback and track it
        task_data["feedback"].append({
            "issues": arguments["issues"],
            "suggestions": arguments["suggestions"],
            "approved": arguments.get("approve", False)
        })
        stats_tracker.record_feedback(task_id)

        if arguments.get("approve", False):
            # Approved! Finalize stats and generate comprehensive report
            task_data["status"] = "approved"
            stats_tracker.finalize_task(task_id, status="approved")

            # Generate comprehensive statistics report
            stats_report = stats_tracker.get_task_report(task_id)

            # Archive task and cleanup old tasks
            stats_tracker.archive_task(task_id)
            stats_tracker.cleanup_old_tasks()

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "approved",
                    "message": "Code approved after review",
                    "final_output": task_data["iterations"][-1]["output"],
                    "total_iterations": len(task_data["iterations"]),
                    "statistics": stats_report
                }, indent=2)
            )]

        else:
            # Need another iteration
            if len(task_data["iterations"]) >= 3:
                stats_tracker.finalize_task(task_id, status="max_iterations_reached")
                stats_report = stats_tracker.get_task_report(task_id)

                # Archive task and cleanup old tasks
                stats_tracker.archive_task(task_id)
                stats_tracker.cleanup_old_tasks()

                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "max_iterations_reached",
                        "message": "Max iterations reached. Consider handling this task yourself.",
                        "iterations": len(task_data["iterations"]),
                        "statistics": stats_report
                    }, indent=2)
                )]

            # Trigger another iteration with feedback
            model = task_data["task"].get("model", "qwen2.5-coder:7b")
            result = await task_data["feedback_loop"].execute_with_feedback(
                task_data["task"],
                model,
                task_data["feedback"],
                task_id=task_id
            )

            task_data["iterations"].append(result)

            # Get current statistics for real-time reporting
            current_stats = stats_tracker.get_current_stats(task_id)

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "iteration_complete",
                    "iteration": len(task_data["iterations"]),
                    "output": result["output"],
                    "message": f"Local model addressed feedback. Iteration {len(task_data['iterations'])} complete.",
                    "next_step": "Review again and provide feedback or approve",
                    "statistics": current_stats
                }, indent=2)
            )]
    
    elif name == "compare_iterations":
        task_id = arguments["task_id"]

        if task_id not in active_tasks:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Task not found"})
            )]

        task_data = active_tasks[task_id]

        comparison = {
            "task_id": task_id,
            "total_iterations": len(task_data["iterations"]),
            "status": task_data["status"],
            "iterations": []
        }

        for i, iteration in enumerate(task_data["iterations"], 1):
            feedback = task_data["feedback"][i-1] if i <= len(task_data["feedback"]) else None

            comparison["iterations"].append({
                "iteration": i,
                "output_preview": iteration["output"][:200] + "...",
                "feedback_received": feedback
            })

        # Get current statistics for the task
        current_stats = stats_tracker.get_current_stats(task_id)
        comparison["statistics"] = current_stats

        return [TextContent(
            type="text",
            text=json.dumps(comparison, indent=2)
        )]

    elif name == "get_statistics_summary":
        # Get cumulative statistics across all tasks
        cumulative_stats = stats_tracker.get_cumulative_statistics()

        return [TextContent(
            type="text",
            text=json.dumps({
                "cumulative_statistics": cumulative_stats,
                "message": "Cumulative statistics across all delegation tasks"
            }, indent=2)
        )]

analyzer = FeedbackAnalyzer()

async def main():
    """
    Main entry point for the local agent MCP server.
    """
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())