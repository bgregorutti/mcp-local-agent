from mcp.server import Server
from mcp.types import Tool, TextContent
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import urllib.request
import urllib.error

app = Server("local-agent-server")


# ============================================================================
# Backend Abstraction Layer - Support Multiple Local Model Backends
# ============================================================================

class ModelBackend:
    """Abstract base for different local model backends"""

    async def generate(self, messages: List[Dict[str, str]], model: str) -> str:
        raise NotImplementedError


class OllamaBackend(ModelBackend):
    """Backend for Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate(self, messages: List[Dict[str, str]], model: str) -> str:
        try:
            import ollama
            response = ollama.chat(model=model, messages=messages)
            return response['message']['content']
        except ImportError:
            raise RuntimeError("Ollama package not installed. Run: pip install ollama")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")


class LMStudioBackend(ModelBackend):
    """Backend for LM Studio OpenAI-compatible API"""

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url

    async def generate(self, messages: List[Dict[str, str]], model: str) -> str:
        """Call LM Studio's OpenAI-compatible API"""

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
                return result['choices'][0]['message']['content']

        except urllib.error.URLError as e:
            raise RuntimeError(f"LM Studio connection error: {str(e)}. Is LM Studio running on {self.base_url}?")
        except Exception as e:
            raise RuntimeError(f"LM Studio error: {str(e)}")


class OpenAICompatibleBackend(ModelBackend):
    """Backend for any OpenAI-compatible API (vLLM, LocalAI, etc.)"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key

    async def generate(self, messages: List[Dict[str, str]], model: str) -> str:
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
                return result['choices'][0]['message']['content']

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

    async def execute_with_feedback(self, task: Dict[str, Any], local_model: str, feedback_history: Optional[List] = None):
        """
        Execute task with local model, but track iterations for remote review

        Args:
            task: Task configuration with 'prompt' and optional 'system_prompt'
            local_model: Model identifier (e.g., "qwen2.5-coder:7b" for LM Studio)
            feedback_history: Previous feedback to incorporate
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

        # Execute with local model via backend
        try:
            current_output = await self.backend.generate(messages, local_model)
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
            "timestamp": datetime.now().isoformat()
        })

        # Return for remote review (break here, remote will call back)
        return {
            "status": "awaiting_review",
            "iteration": iteration,
            "output": current_output,
            "can_iterate": iteration < self.max_iterations,
            "history": self.history
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
            output = await task_backend.generate(messages, model)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "output": output,
                    "model": model
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

        task_data = active_tasks[task_id]
        model = arguments.get("model", "qwen2.5-coder:7b")

        # Execute with local model
        result = await task_data["feedback_loop"].execute_with_feedback(
            arguments,
            model,
            task_data["feedback"]
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

        # Store feedback
        task_data["feedback"].append({
            "issues": arguments["issues"],
            "suggestions": arguments["suggestions"],
            "approved": arguments.get("approve", False)
        })

        if arguments.get("approve", False):
            # Approved! Mark as done
            task_data["status"] = "approved"

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "approved",
                    "message": "Code approved after review",
                    "final_output": task_data["iterations"][-1]["output"],
                    "total_iterations": len(task_data["iterations"])
                }, indent=2)
            )]

        else:
            # Need another iteration
            if len(task_data["iterations"]) >= 3:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "max_iterations_reached",
                        "message": "Max iterations reached. Consider handling this task yourself.",
                        "iterations": len(task_data["iterations"])
                    }, indent=2)
                )]

            # Trigger another iteration with feedback
            model = task_data["task"].get("model", "qwen2.5-coder:7b")
            result = await task_data["feedback_loop"].execute_with_feedback(
                task_data["task"],
                model,
                task_data["feedback"]
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