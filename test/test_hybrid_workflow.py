#!/usr/bin/env python3
"""
Test the hybrid workflow by directly calling the local model
This simulates what will happen when Claude delegates via MCP
"""

import json
import asyncio
from local_agent_mcp_server import get_backend, FeedbackLoop

async def test_quick_delegation():
    """Test 1: Quick delegation without feedback loop"""
    print("\n" + "="*60)
    print("TEST 1: Quick Delegation (Simple Boilerplate)")
    print("="*60)

    backend = get_backend(backend_type="lmstudio", base_url="http://localhost:1234/v1")

    prompt = """Create a Python dataclass for a User with these fields:
- id: UUID
- name: str
- email: str
- created_at: datetime

Include proper imports and type hints."""

    messages = [
        {"role": "system", "content": "You are an expert Python developer."},
        {"role": "user", "content": prompt}
    ]

    print("\n📤 Delegating to local model: mistralai/devstral-small-2505")
    print(f"📝 Task: Generate User dataclass\n")

    try:
        output = await backend.generate(messages, "mistralai/devstral-small-2505")
        print("✅ Local model response:")
        print("-" * 60)
        print(output)
        print("-" * 60)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_feedback_loop():
    """Test 2: Feedback loop with iteration"""
    print("\n" + "="*60)
    print("TEST 2: Feedback Loop (Iterative Refinement)")
    print("="*60)

    backend = get_backend(backend_type="lmstudio", base_url="http://localhost:1234/v1")
    feedback_loop = FeedbackLoop(backend)

    task = {
        "prompt": """Create a FastAPI endpoint for creating a new user.
Requirements:
- POST /users endpoint
- Accept JSON body with name and email
- Return created user with 201 status
- Include proper error handling""",
        "system_prompt": "You are an expert FastAPI developer. Write clean, production-ready code.",
    }

    print("\n📤 Iteration 1: Initial generation")
    print("🤖 Model: qwen/qwen3-coder-30b")

    try:
        # First iteration
        result1 = await feedback_loop.execute_with_feedback(
            task,
            "qwen/qwen3-coder-30b",
            []
        )

        print(f"\n✅ Iteration {result1['iteration']} complete")
        print("📄 Generated code:")
        print("-" * 60)
        print(result1['output'][:500] + "..." if len(result1['output']) > 500 else result1['output'])
        print("-" * 60)

        # Simulate Claude's review
        print("\n🔍 Claude's Review:")
        feedback = {
            "issues": [
                "Missing input validation for email format",
                "No proper error response model",
            ],
            "suggestions": [
                "Add Pydantic EmailStr validator",
                "Create ErrorResponse model for 400/500 errors",
            ]
        }

        print(f"   Issues found: {len(feedback['issues'])}")
        for issue in feedback['issues']:
            print(f"   - {issue}")
        print(f"   Suggestions: {len(feedback['suggestions'])}")
        for suggestion in feedback['suggestions']:
            print(f"   - {suggestion}")

        # Second iteration with feedback
        print("\n📤 Iteration 2: Addressing feedback")

        result2 = await feedback_loop.execute_with_feedback(
            task,
            "qwen/qwen3-coder-30b",
            [feedback]
        )

        print(f"\n✅ Iteration {result2['iteration']} complete")
        print("📄 Improved code:")
        print("-" * 60)
        print(result2['output'][:500] + "..." if len(result2['output']) > 500 else result2['output'])
        print("-" * 60)

        print("\n✅ Feedback loop successful!")
        print(f"   Total iterations: {result2['iteration']}")
        print(f"   Can iterate more: {result2['can_iterate']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_parallel_delegation():
    """Test 3: Simulating parallel delegation of multiple tasks"""
    print("\n" + "="*60)
    print("TEST 3: Parallel Delegation (Multiple Tasks)")
    print("="*60)

    backend = get_backend(backend_type="lmstudio", base_url="http://localhost:1234/v1")

    tasks = [
        {
            "name": "Model",
            "prompt": "Create a Pydantic model for a Task with: id, title, description, status",
            "model": "mistralai/devstral-small-2505"
        },
        {
            "name": "Validation",
            "prompt": "Create a function to validate email addresses using regex",
            "model": "mistralai/devstral-small-2505"
        },
        {
            "name": "Test",
            "prompt": "Create pytest test cases for a function that adds two numbers",
            "model": "mistralai/devstral-small-2505"
        }
    ]

    print(f"\n📤 Delegating {len(tasks)} tasks in parallel...\n")

    # Run all tasks concurrently
    async def run_task(task):
        messages = [
            {"role": "system", "content": "You are an expert Python developer."},
            {"role": "user", "content": task["prompt"]}
        ]
        result = await backend.generate(messages, task["model"])
        return task["name"], result

    try:
        results = await asyncio.gather(*[run_task(task) for task in tasks])

        for name, output in results:
            print(f"✅ Task '{name}' complete")
            print(f"   Output length: {len(output)} chars")
            print(f"   Preview: {output[:100]}...")
            print()

        print(f"✅ All {len(tasks)} tasks completed in parallel!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n🧪 Testing Hybrid Delegation System")
    print("=" * 60)
    print("This simulates how Claude will delegate to local models")
    print("=" * 60)

    results = []

    # Test 1: Quick delegation
    results.append(("Quick Delegation", await test_quick_delegation()))

    # Test 2: Feedback loop
    results.append(("Feedback Loop", await test_feedback_loop()))

    # Test 3: Parallel delegation
    results.append(("Parallel Delegation", await test_parallel_delegation()))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")

    total_passed = sum(1 for _, success in results if success)
    print(f"\n{total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n🎉 All systems ready for hybrid delegation!")
        print("\n📋 Next: Restart Claude Code and try delegating:")
        print("   1. Simple task: Use execute_with_local_model")
        print("   2. Complex task: Use execute_with_feedback_loop")
        print("   3. Review and iterate until quality threshold met")
    else:
        print("\n⚠️  Some tests failed. Check LM Studio connection.")


if __name__ == "__main__":
    asyncio.run(main())
