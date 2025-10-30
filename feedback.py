from backends import ModelBackend
from typing import List, Dict, Any, Optional
from datetime import datetime

from usage_tracker import StatisticsTracker

class FeedbackLoop:
    """Manages iterative refinement between local and remote models"""

    def __init__(self, backend: ModelBackend, stats_tracker: StatisticsTracker):
        self.max_iterations = 3
        self.history = []
        self.backend = backend
        self.stats_tracker = stats_tracker

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
                self.stats_tracker.record_local_call(task_id, tokens_sent, tokens_received, duration, local_model)
                self.stats_tracker.record_iteration(task_id, iteration, duration, tokens_sent, tokens_received)

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
