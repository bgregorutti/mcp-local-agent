from mcp.server import Server
from mcp.types import Tool, TextContent
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import urllib.request
import urllib.error
import time

app = Server("local-agent-server")


# ============================================================================
# Statistics Tracking
# ============================================================================

class StatisticsTracker:
    """Track comprehensive statistics for delegation tasks"""

    def __init__(self):
        self.tasks = {}

    def init_task(self, task_id: str):
        """Initialize statistics for a new task"""
        self.tasks[task_id] = {
            "task_id": task_id,
            "start_time": time.time(),
            "end_time": None,
            "total_duration": None,
            "local_model_stats": {
                "calls": 0,
                "total_tokens_sent": 0,
                "total_tokens_received": 0,
                "total_time": 0,
                "model_name": None
            },
            "iterations": [],
            "feedback_rounds": 0,
            "status": "in_progress"
        }

    def record_local_call(self, task_id: str, tokens_sent: int, tokens_received: int,
                          duration: float, model: str):
        """Record a local model API call"""
        if task_id not in self.tasks:
            self.init_task(task_id)

        stats = self.tasks[task_id]["local_model_stats"]
        stats["calls"] += 1
        stats["total_tokens_sent"] += tokens_sent
        stats["total_tokens_received"] += tokens_received
        stats["total_time"] += duration
        stats["model_name"] = model

    def record_iteration(self, task_id: str, iteration_num: int, duration: float,
                        tokens_sent: int, tokens_received: int):
        """Record details of a specific iteration"""
        if task_id not in self.tasks:
            self.init_task(task_id)

        self.tasks[task_id]["iterations"].append({
            "iteration": iteration_num,
            "duration": duration,
            "tokens_sent": tokens_sent,
            "tokens_received": tokens_received,
            "timestamp": datetime.now().isoformat()
        })

    def record_feedback(self, task_id: str):
        """Record a feedback round"""
        if task_id in self.tasks:
            self.tasks[task_id]["feedback_rounds"] += 1

    def finalize_task(self, task_id: str, status: str = "completed"):
        """Finalize task statistics"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task["end_time"] = time.time()
            task["total_duration"] = task["end_time"] - task["start_time"]
            task["status"] = status

    def get_task_report(self, task_id: str) -> Dict[str, Any]:
        """Generate comprehensive statistics report for a task"""
        if task_id not in self.tasks:
            return {"error": "Task not found"}

        task = self.tasks[task_id]
        local_stats = task["local_model_stats"]

        # Calculate efficiency metrics
        total_iterations = len(task["iterations"])
        avg_iteration_time = (local_stats["total_time"] / total_iterations
                             if total_iterations > 0 else 0)

        # Estimate cost savings (assuming $0.003/1K tokens for input, $0.015/1K tokens for output)
        # vs local model (free)
        estimated_remote_cost = (
            (local_stats["total_tokens_sent"] / 1000) * 0.003 +
            (local_stats["total_tokens_received"] / 1000) * 0.015
        )

        # Remote tokens (orchestrator) - these are the tokens sent TO the orchestrator
        # which is just the final outputs being reviewed
        estimated_remote_tokens = sum(
            iter_data["tokens_received"] for iter_data in task["iterations"]
        )
        estimated_remote_cost_actual = (estimated_remote_tokens / 1000) * 0.003

        cost_savings = estimated_remote_cost - estimated_remote_cost_actual
        cost_savings_percent = (cost_savings / estimated_remote_cost * 100
                               if estimated_remote_cost > 0 else 0)

        report = {
            "task_id": task_id,
            "status": task["status"],
            "summary": {
                "total_duration_seconds": round(task["total_duration"], 2) if task["total_duration"] else None,
                "total_iterations": total_iterations,
                "feedback_rounds": task["feedback_rounds"],
                "average_iteration_time_seconds": round(avg_iteration_time, 2)
            },
            "local_model_usage": {
                "model": local_stats["model_name"],
                "api_calls": local_stats["calls"],
                "tokens": {
                    "sent": local_stats["total_tokens_sent"],
                    "received": local_stats["total_tokens_received"],
                    "total": local_stats["total_tokens_sent"] + local_stats["total_tokens_received"]
                },
                "time_seconds": round(local_stats["total_time"], 2)
            },
            "remote_model_usage": {
                "estimated_tokens_for_review": estimated_remote_tokens,
                "estimated_cost_usd": round(estimated_remote_cost_actual, 4)
            },
            "cost_analysis": {
                "local_model_cost_usd": 0.00,  # Local models are free
                "estimated_cost_if_fully_remote_usd": round(estimated_remote_cost, 4),
                "actual_cost_usd": round(estimated_remote_cost_actual, 4),
                "savings_usd": round(cost_savings, 4),
                "savings_percent": round(cost_savings_percent, 1)
            },
            "iteration_breakdown": task["iterations"]
        }

        return report

# Global statistics tracker
stats_tracker = StatisticsTracker()


# ============================================================================
# Token Estimation
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (GPT-style: ~4 chars per token)
    For more accuracy, could use tiktoken library, but this is sufficient
    """
    return len(text) // 4


# ============================================================================
# Backend Abstraction Layer - Support Multiple Local Model Backends
# ============================================================================

class ModelBackend:
    """Abstract base for different local model backends"""

    async def generate(self, messages: List[Dict[str, str]], model: str) -> tuple[str, int, int, float]:
        """
        Generate response from model.

        Returns:
            tuple: (response_text, tokens_sent, tokens_received, duration_seconds)
        """
        raise NotImplementedError


class OllamaBackend(ModelBackend):
    """Backend for Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate(self, messages: List[Dict[str, str]], model: str) -> tuple[str, int, int, float]:
        start_time = time.time()

        # Calculate input tokens
        input_text = " ".join(msg["content"] for msg in messages)
        tokens_sent = estimate_tokens(input_text)

        try:
            import ollama
            response = ollama.chat(model=model, messages=messages)
            output_text = response['message']['content']

            # Calculate output tokens
            tokens_received = estimate_tokens(output_text)
            duration = time.time() - start_time

            return output_text, tokens_sent, tokens_received, duration

        except ImportError:
            raise RuntimeError("Ollama package not installed. Run: pip install ollama")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")


class LMStudioBackend(ModelBackend):
    """Backend for LM Studio OpenAI-compatible API"""

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url

    async def generate(self, messages: List[Dict[str, str]], model: str) -> tuple[str, int, int, float]:
        """Call LM Studio's OpenAI-compatible API"""
        start_time = time.time()

        # Calculate input tokens
        input_text = " ".join(msg["content"] for msg in messages)
        tokens_sent = estimate_tokens(input_text)

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": -1,  # LM Studio uses -1 for unlimited
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(request_data).encode('utf-8'),
                headers={
                    "Content-Type": "application/json",
                }
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                output_text = result['choices'][0]['message']['content']

                # Calculate output tokens
                tokens_received = estimate_tokens(output_text)
                duration = time.time() - start_time

                return output_text, tokens_sent, tokens_received, duration

        except urllib.error.URLError as e:
            raise RuntimeError(f"LM Studio connection error: {str(e)}. Is LM Studio running on {self.base_url}?")
        except Exception as e:
            raise RuntimeError(f"LM Studio error: {str(e)}")


class OpenAICompatibleBackend(ModelBackend):
    """Backend for any OpenAI-compatible API (vLLM, LocalAI, etc.)"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key

    async def generate(self, messages: List[Dict[str, str]], model: str) -> tuple[str, int, int, float]:
        start_time = time.time()

        # Calculate input tokens
        input_text = " ".join(msg["content"] for msg in messages)
        tokens_sent = estimate_tokens(input_text)

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(request_data).encode('utf-8'),
                headers=headers
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                output_text = result['choices'][0]['message']['content']

                # Calculate output tokens
                tokens_received = estimate_tokens(output_text)
                duration = time.time() - start_time

                return output_text, tokens_sent, tokens_received, duration

        except Exception as e:
            raise RuntimeError(f"API error: {str(e)}")


# Backend factory
def get_backend(backend_type: str = "lmstudio", **kwargs) -> ModelBackend:
    """
    Get the appropriate backend based on type.

    Args:
        backend_type: "ollama", "lmstudio", or "openai-compatible"
        **kwargs: Backend-specific configuration (base_url, api_key, etc.)
    """
    backends = {
        "ollama": OllamaBackend,
        "lmstudio": LMStudioBackend,
        "openai-compatible": OpenAICompatibleBackend,
    }

    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Choose from: {list(backends.keys())}")

    return backends[backend_type](**kwargs)


# ============================================================================
# Feedback Loop System
# ============================================================================

class FeedbackLoop:
    """Manages iterative refinement between local and remote models"""

    def __init__(self, backend: ModelBackend):
        self.max_iterations = 3
        self.history = []
        self.backend = backend

    async def execute_with_feedback(self, task: Dict[str, Any], local_model: str,
                                     feedback_history: Optional[List] = None, task_id: Optional[str] = None):
        """
        Execute task with local model, but track iterations for remote review

        Args:
            task: Task configuration with 'prompt' and optional 'system_prompt'
            local_model: Model identifier (e.g., "qwen2.5-coder:7b" for LM Studio)
            feedback_history: Previous feedback to incorporate
            task_id: Task ID for statistics tracking
        """
        if feedback_history is None:
            feedback_history = []

        iteration = len(self.history) + 1

        if iteration > self.max_iterations:
            return {
                "status": "max_iterations_reached",
                "iteration": iteration - 1,
                "output": self.history[-1]["output"] if self.history else None,
                "can_iterate": False,
                "history": self.history
            }

        # Build prompt with previous feedback
        prompt = self._build_prompt_with_feedback(
            task,
            feedback_history,
            iteration
        )

        # Build messages for the model
        messages = []
        if task.get("system_prompt"):
            messages.append({"role": "system", "content": task["system_prompt"]})
        messages.append({"role": "user", "content": prompt})

        # Execute with local model via backend (now returns tuple with stats)
        try:
            current_output, tokens_sent, tokens_received, duration = await self.backend.generate(messages, local_model)

            # Track statistics if task_id provided
            if task_id:
                stats_tracker.record_local_call(task_id, tokens_sent, tokens_received, duration, local_model)
                stats_tracker.record_iteration(task_id, iteration, duration, tokens_sent, tokens_received)

        except Exception as e:
            return {
                "status": "error",
                "iteration": iteration,
                "error": str(e),
                "can_iterate": False,
                "history": self.history
            }

        # Store this iteration
        self.history.append({
            "iteration": iteration,
            "output": current_output,
            "model": local_model,
            "timestamp": datetime.now().isoformat(),
            "tokens_sent": tokens_sent,
            "tokens_received": tokens_received,
            "duration": duration
        })

        # Return for remote review (break here, remote will call back)
        return {
            "status": "awaiting_review",
            "iteration": iteration,
            "output": current_output,
            "can_iterate": iteration < self.max_iterations,
            "history": self.history,
            "tokens_sent": tokens_sent,
            "tokens_received": tokens_received,
            "duration": duration
        }
    
    def _build_prompt_with_feedback(self, task, feedback_history, iteration):
        """Build prompt that incorporates previous feedback"""
        
        base_prompt = task["prompt"]
        
        if iteration == 1:
            # First attempt - just the task
            return base_prompt
        else:
            # Subsequent attempts - include feedback
            prompt = f"""{base_prompt}

PREVIOUS ATTEMPTS AND FEEDBACK:
"""
            for i, feedback in enumerate(feedback_history, 1):
                prompt += f"""
Attempt {i}:
Issues found: {feedback['issues']}
Suggestions: {feedback['suggestions']}

"""
            prompt += """
Please address all the feedback above and generate improved code.
"""
            return prompt


# Initialize backend (defaults to LM Studio)
# You can change this by setting environment variables:
# BACKEND_TYPE=lmstudio (default)
# BACKEND_TYPE=ollama
# BACKEND_TYPE=openai-compatible
# BACKEND_URL=http://localhost:1234/v1 (for custom URLs)
import os as _os
backend = get_backend(
    backend_type=_os.environ.get("BACKEND_TYPE", "lmstudio"),
    base_url=_os.environ.get("BACKEND_URL", "http://localhost:1234/v1")
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

        # Get backend (custom or default)
        task_backend = backend
        if arguments.get("backend_type") or arguments.get("backend_url"):
            task_backend = get_backend(
                backend_type=arguments.get("backend_type", "lmstudio"),
                base_url=arguments.get("backend_url", "http://localhost:1234/v1")
            )

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            output, tokens_sent, tokens_received, duration = await task_backend.generate(messages, model)

            # Simple stats for quick delegation
            stats = {
                "tokens_sent": tokens_sent,
                "tokens_received": tokens_received,
                "total_tokens": tokens_sent + tokens_received,
                "duration_seconds": round(duration, 2),
                "model": model
            }

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "output": output,
                    "statistics": stats
                }, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "error": str(e)
                }, indent=2)
            )]

    elif name == "execute_with_feedback_loop":
        task_id = arguments["task_id"]

        # Initialize or continue task
        if task_id not in active_tasks:
            # Get backend for this task
            task_backend = backend
            if arguments.get("backend_type") or arguments.get("backend_url"):
                task_backend = get_backend(
                    backend_type=arguments.get("backend_type", "lmstudio"),
                    base_url=arguments.get("backend_url", "http://localhost:1234/v1")
                )

            active_tasks[task_id] = {
                "task": arguments,
                "iterations": [],
                "feedback": [],
                "status": "in_progress",
                "feedback_loop": FeedbackLoop(task_backend)
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
            return [TextContent(
                type="text",
                text=json.dumps({
                    "task_id": task_id,
                    "status": "error",
                    "error": result.get("error"),
                    "message": "Local model execution failed. Check your backend connection."
                }, indent=2)
            )]

        # Store iteration
        task_data["iterations"].append(result)

        return [TextContent(
            type="text",
            text=json.dumps({
                "task_id": task_id,
                "iteration": result["iteration"],
                "status": result["status"],
                "output": result["output"],
                "message": f"Local model completed iteration {result['iteration']}. Please review and provide feedback.",
                "next_step": "Call 'provide_feedback' with your review"
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

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "iteration_complete",
                    "iteration": len(task_data["iterations"]),
                    "output": result["output"],
                    "message": f"Local model addressed feedback. Iteration {len(task_data['iterations'])} complete.",
                    "next_step": "Review again and provide feedback or approve"
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
        
        return [TextContent(
            type="text",
            text=json.dumps(comparison, indent=2)
        )]
    

class FeedbackAnalyzer:
    """Analyzes feedback patterns to improve future prompts"""
    
    def __init__(self):
        self.common_issues = {}
    
    def analyze_feedback(self, task_type, feedback_history):
        """Track which issues come up repeatedly"""
        
        for feedback in feedback_history:
            for issue in feedback['issues']:
                key = f"{task_type}:{issue}"
                self.common_issues[key] = self.common_issues.get(key, 0) + 1
    
    def enhance_prompt(self, task_type, base_prompt):
        """Add preemptive guidance based on past feedback"""
        
        common_for_type = [
            issue for issue, count in self.common_issues.items()
            if issue.startswith(f"{task_type}:") and count >= 3
        ]
        
        if common_for_type:
            enhancement = "\n\nIMPORTANT - Based on past feedback, ensure you:"
            for issue in common_for_type:
                enhancement += f"\n- Avoid: {issue.split(':', 1)[1]}"
            
            return base_prompt + enhancement
        
        return base_prompt

analyzer = FeedbackAnalyzer()


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )

    asyncio.run(main())