from datetime import datetime
import json
import os
import time
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()

# Remote model pricing (per 1K tokens)
REMOTE_INPUT_COST_PER_1K = float(os.getenv("REMOTE_INPUT_COST_PER_1K", "0.003"))
REMOTE_OUTPUT_COST_PER_1K = float(os.getenv("REMOTE_OUTPUT_COST_PER_1K", "0.015"))

# Local model pricing (per 1K tokens) - typically 0 for local models
LOCAL_COST_PER_1K = float(os.getenv("LOCAL_COST_PER_1K", "0.0"))


class StatisticsTracker:
    """Track comprehensive statistics for delegation tasks"""

    def __init__(self, stats_file: str = ".mcp_stats.json"):
        self.tasks = {}
        self.archived_tasks = {}
        self.max_active_tasks = 100  # Keep only last 100 tasks in active memory
        self.stats_file = stats_file

        # Store pricing configuration
        self.remote_input_cost = REMOTE_INPUT_COST_PER_1K
        self.remote_output_cost = REMOTE_OUTPUT_COST_PER_1K
        self.local_cost = LOCAL_COST_PER_1K

        # Load persisted statistics on initialization
        self.load_from_file()

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
            # Auto-save after finalization
            self.save_to_file()

    def archive_task(self, task_id: str):
        """Archive a completed task to free up active memory"""
        if task_id in self.tasks:
            self.archived_tasks[task_id] = self.tasks[task_id]
            del self.tasks[task_id]
            # Auto-save after archival
            self.save_to_file()

    def cleanup_old_tasks(self):
        """Clean up old tasks if we exceed the max active tasks limit"""
        if len(self.tasks) > self.max_active_tasks:
            # Sort tasks by end_time (or start_time if not finalized)
            sorted_tasks = sorted(
                self.tasks.items(),
                key=lambda x: x[1].get("end_time") or x[1].get("start_time", 0)
            )

            # Archive oldest tasks until we're under the limit
            tasks_to_archive = len(self.tasks) - self.max_active_tasks
            for task_id, _ in sorted_tasks[:tasks_to_archive]:
                self.archive_task(task_id)

    def get_all_tasks(self) -> Dict[str, Any]:
        """Get all tasks (active and archived)"""
        return {
            "active_tasks": self.tasks,
            "archived_tasks": self.archived_tasks,
            "total_tasks": len(self.tasks) + len(self.archived_tasks)
        }

    def save_to_file(self):
        """Persist statistics to file for cross-session tracking"""
        try:
            data = {
                "tasks": self.tasks,
                "archived_tasks": self.archived_tasks,
                "last_updated": datetime.now().isoformat(),
                "pricing_config": {
                    "remote_input_per_1k": self.remote_input_cost,
                    "remote_output_per_1k": self.remote_output_cost,
                    "local_per_1k": self.local_cost
                }
            }
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Silently fail - don't crash the server if persistence fails
            print(f"Warning: Failed to save statistics to {self.stats_file}: {e}")

    def load_from_file(self):
        """Load persisted statistics from file"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)

                self.tasks = data.get("tasks", {})
                self.archived_tasks = data.get("archived_tasks", {})

                # Note: We don't override pricing config from file
                # Always use environment variables for pricing
                print(f"Loaded {len(self.tasks)} active tasks and {len(self.archived_tasks)} archived tasks from {self.stats_file}")
        except Exception as e:
            # Silently fail - start fresh if can't load
            print(f"Warning: Failed to load statistics from {self.stats_file}: {e}")
            self.tasks = {}
            self.archived_tasks = {}

    def get_cumulative_statistics(self) -> Dict[str, Any]:
        """Generate cumulative statistics across all tasks"""
        all_tasks = {**self.tasks, **self.archived_tasks}

        if not all_tasks:
            return {
                "message": "No tasks tracked yet",
                "total_tasks": 0
            }

        # Initialize counters
        total_tasks = len(all_tasks)
        successful_tasks = 0
        failed_tasks = 0
        approved_tasks = 0
        max_iterations_tasks = 0

        total_local_tokens_sent = 0
        total_local_tokens_received = 0
        total_local_calls = 0
        total_local_time = 0

        total_remote_tokens = 0
        total_iterations = 0
        total_feedback_rounds = 0

        breakdown_by_type = {}

        # Aggregate statistics from all tasks
        for task_id, task in all_tasks.items():
            status = task.get("status", "unknown")

            if status == "completed" or status == "approved":
                successful_tasks += 1
            if status == "approved":
                approved_tasks += 1
            if status == "error":
                failed_tasks += 1
            if status == "max_iterations_reached":
                max_iterations_tasks += 1

            local_stats = task["local_model_stats"]
            total_local_tokens_sent += local_stats["total_tokens_sent"]
            total_local_tokens_received += local_stats["total_tokens_received"]
            total_local_calls += local_stats["calls"]
            total_local_time += local_stats["total_time"]

            # Count remote tokens (tokens sent for review)
            for iter_data in task.get("iterations", []):
                total_remote_tokens += iter_data.get("tokens_received", 0)

            total_iterations += len(task.get("iterations", []))
            total_feedback_rounds += task.get("feedback_rounds", 0)

            # Track by task type if available (for execute_with_feedback_loop)
            if "task" in task and "task_type" in task["task"]:
                task_type = task["task"]["task_type"]
                if task_type not in breakdown_by_type:
                    breakdown_by_type[task_type] = {
                        "tasks": 0,
                        "local_tokens": 0,
                        "estimated_savings_usd": 0.0
                    }

                breakdown_by_type[task_type]["tasks"] += 1
                breakdown_by_type[task_type]["local_tokens"] += (
                    local_stats["total_tokens_sent"] + local_stats["total_tokens_received"]
                )

        # Calculate costs using configurable pricing
        estimated_remote_cost_if_fully_remote = (
            (total_local_tokens_sent / 1000) * self.remote_input_cost +
            (total_local_tokens_received / 1000) * self.remote_output_cost
        )
        actual_local_cost = (
            (total_local_tokens_sent + total_local_tokens_received) / 1000
        ) * self.local_cost
        actual_remote_cost = (total_remote_tokens / 1000) * self.remote_input_cost
        total_savings = estimated_remote_cost_if_fully_remote - (actual_remote_cost + actual_local_cost)
        savings_percent = (total_savings / estimated_remote_cost_if_fully_remote * 100
                          if estimated_remote_cost_if_fully_remote > 0 else 0)

        # Calculate breakdown savings
        for task_type, data in breakdown_by_type.items():
            type_remote_cost = (data["local_tokens"] / 1000) * 0.009  # Avg of input/output
            data["estimated_savings_usd"] = round(type_remote_cost, 4)

        return {
            "total_tasks": total_tasks,
            "task_status_breakdown": {
                "successful": successful_tasks,
                "failed": failed_tasks,
                "approved": approved_tasks,
                "max_iterations_reached": max_iterations_tasks
            },
            "local_model_usage": {
                "total_api_calls": total_local_calls,
                "total_tokens": {
                    "sent": total_local_tokens_sent,
                    "received": total_local_tokens_received,
                    "total": total_local_tokens_sent + total_local_tokens_received
                },
                "total_time_seconds": round(total_local_time, 2),
                "cost_usd": round(actual_local_cost, 4)
            },
            "remote_model_usage": {
                "total_tokens_for_review": total_remote_tokens,
                "estimated_cost_usd": round(actual_remote_cost, 4)
            },
            "aggregated_metrics": {
                "total_iterations": total_iterations,
                "total_feedback_rounds": total_feedback_rounds,
                "average_iterations_per_task": round(total_iterations / total_tasks, 2) if total_tasks > 0 else 0
            },
            "cost_analysis": {
                "local_model_cost_usd": round(actual_local_cost, 4),
                "estimated_cost_if_fully_remote_usd": round(estimated_remote_cost_if_fully_remote, 4),
                "actual_cost_usd": round(actual_remote_cost + actual_local_cost, 4),
                "total_savings_usd": round(total_savings, 4),
                "savings_percent": round(savings_percent, 1),
                "pricing_config": {
                    "remote_input_per_1k": self.remote_input_cost,
                    "remote_output_per_1k": self.remote_output_cost,
                    "local_per_1k": self.local_cost
                }
            },
            "breakdown_by_type": breakdown_by_type if breakdown_by_type else None
        }

    def get_current_stats(self, task_id: str) -> Dict[str, Any]:
        """Get current statistics without finalizing the task (for real-time reporting)"""
        # Check both active and archived tasks
        if task_id not in self.tasks and task_id not in self.archived_tasks:
            return {"error": "Task not found"}

        task = self.tasks.get(task_id) or self.archived_tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}

        local_stats = task["local_model_stats"]

        # Calculate current duration (even if not finalized)
        current_duration = time.time() - task["start_time"]

        # Calculate efficiency metrics
        total_iterations = len(task["iterations"])
        avg_iteration_time = (local_stats["total_time"] / total_iterations
                             if total_iterations > 0 else 0)

        # Estimate cost savings using configurable pricing
        estimated_remote_cost = (
            (local_stats["total_tokens_sent"] / 1000) * self.remote_input_cost +
            (local_stats["total_tokens_received"] / 1000) * self.remote_output_cost
        )

        # Local model cost
        local_model_cost = (
            (local_stats["total_tokens_sent"] + local_stats["total_tokens_received"]) / 1000
        ) * self.local_cost

        # Remote tokens (orchestrator) - these are the tokens sent TO the orchestrator
        # which is just the final outputs being reviewed
        estimated_remote_tokens = sum(
            iter_data["tokens_received"] for iter_data in task["iterations"]
        )
        estimated_remote_cost_actual = (estimated_remote_tokens / 1000) * self.remote_input_cost

        cost_savings = estimated_remote_cost - (estimated_remote_cost_actual + local_model_cost)
        cost_savings_percent = (cost_savings / estimated_remote_cost * 100
                               if estimated_remote_cost > 0 else 0)

        # Get last iteration stats if available
        last_iteration_stats = None
        if task["iterations"]:
            last_iter = task["iterations"][-1]
            last_iteration_stats = {
                "iteration": last_iter["iteration"],
                "duration_seconds": round(last_iter["duration"], 2),
                "tokens_sent": last_iter["tokens_sent"],
                "tokens_received": last_iter["tokens_received"],
                "total_tokens": last_iter["tokens_sent"] + last_iter["tokens_received"]
            }

        report = {
            "task_id": task_id,
            "status": task["status"],
            "current_iteration_stats": last_iteration_stats,
            "cumulative_stats": {
                "duration_so_far_seconds": round(current_duration, 2),
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
                "local_model_cost_usd": round(local_model_cost, 4),
                "estimated_cost_if_fully_remote_usd": round(estimated_remote_cost, 4),
                "actual_cost_usd": round(estimated_remote_cost_actual + local_model_cost, 4),
                "savings_usd": round(cost_savings, 4),
                "savings_percent": round(cost_savings_percent, 1),
                "pricing_config": {
                    "remote_input_per_1k": self.remote_input_cost,
                    "remote_output_per_1k": self.remote_output_cost,
                    "local_per_1k": self.local_cost
                }
            }
        }

        return report

    def get_task_report(self, task_id: str) -> Dict[str, Any]:
        """Generate comprehensive statistics report for a task"""
        # Check both active and archived tasks
        if task_id not in self.tasks and task_id not in self.archived_tasks:
            return {"error": "Task not found"}

        task = self.tasks.get(task_id) or self.archived_tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}

        local_stats = task["local_model_stats"]

        # Calculate efficiency metrics
        total_iterations = len(task["iterations"])
        avg_iteration_time = (local_stats["total_time"] / total_iterations
                             if total_iterations > 0 else 0)

        # Estimate cost savings using configurable pricing
        estimated_remote_cost = (
            (local_stats["total_tokens_sent"] / 1000) * self.remote_input_cost +
            (local_stats["total_tokens_received"] / 1000) * self.remote_output_cost
        )

        # Local model cost
        local_model_cost = (
            (local_stats["total_tokens_sent"] + local_stats["total_tokens_received"]) / 1000
        ) * self.local_cost

        # Remote tokens (orchestrator) - these are the tokens sent TO the orchestrator
        # which is just the final outputs being reviewed
        estimated_remote_tokens = sum(
            iter_data["tokens_received"] for iter_data in task["iterations"]
        )
        estimated_remote_cost_actual = (estimated_remote_tokens / 1000) * self.remote_input_cost

        cost_savings = estimated_remote_cost - (estimated_remote_cost_actual + local_model_cost)
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
                "local_model_cost_usd": round(local_model_cost, 4),
                "estimated_cost_if_fully_remote_usd": round(estimated_remote_cost, 4),
                "actual_cost_usd": round(estimated_remote_cost_actual + local_model_cost, 4),
                "savings_usd": round(cost_savings, 4),
                "savings_percent": round(cost_savings_percent, 1),
                "pricing_config": {
                    "remote_input_per_1k": self.remote_input_cost,
                    "remote_output_per_1k": self.remote_output_cost,
                    "local_per_1k": self.local_cost
                }
            },
            "iteration_breakdown": task["iterations"]
        }

        return report

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (GPT-style: ~4 chars per token)
    For more accuracy, could use tiktoken library, but this is sufficient
    """
    return len(text) // 4
