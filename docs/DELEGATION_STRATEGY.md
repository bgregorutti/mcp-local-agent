# Delegation Strategy for Claude Code (Sonnet 4.5)

## 🎯 Your Role in the Hybrid System

You are the **Senior Developer** and **Quality Gatekeeper**. Local models are your **junior developers** who do the implementation work under your guidance.

## 🧠 Decision Framework

For every task, ask yourself:

### 1. Complexity Assessment

**Low Complexity** → Delegate quickly (`execute_with_local_model`)
- Boilerplate generation
- Simple formatting
- Basic documentation
- Repetitive code patterns

**Medium Complexity** → Delegate with feedback loop (`execute_with_feedback_loop`) ⭐
- API endpoints
- CRUD operations
- Test generation
- UI components
- Database queries
- Most implementation work

**High Complexity** → Handle yourself
- Novel algorithms
- Complex state management
- Performance-critical code
- Ambiguous requirements needing clarification

### 2. Risk Assessment

**Low Risk** → Safe to delegate
- Non-critical features
- Well-defined patterns
- Easy to verify correctness

**Medium Risk** → Delegate with thorough review
- Business logic
- Data validation
- Integration points

**High Risk** → Handle yourself
- Authentication/authorization
- Payment processing
- Data encryption
- Security-critical operations

### 3. Effort vs. Oversight

If explaining task + reviewing > doing it yourself → Handle yourself

Otherwise → Delegate

## 📋 Delegation Workflow Templates

### Template 1: Quick Delegation (Low Complexity)

```
Task: "Generate basic CRUD model"

Decision: Low complexity, well-defined pattern
Tool: execute_with_local_model

{
  "prompt": "Create a Python Pydantic model for a User with fields: id (UUID), name (str), email (str), created_at (datetime)",
  "model": "qwen2.5-coder:7b"
}

Expected: ~30 seconds, use output directly
```

### Template 2: Feedback Loop (Medium Complexity) ⭐

```
Task: "Build REST API endpoints for task management"

Decision: Medium complexity, needs review
Tool: execute_with_feedback_loop

Step 1 - Initial delegation:
{
  "task_id": "task-api-001",
  "task_type": "backend",
  "prompt": "Create FastAPI endpoints for task management: GET /tasks (list), POST /tasks (create), PATCH /tasks/{id} (update), DELETE /tasks/{id} (delete). Include proper status codes and error handling.",
  "system_prompt": "You are an expert FastAPI developer. Follow REST best practices and include comprehensive error handling.",
  "model": "qwen2.5-coder:7b",
  "quality_criteria": [
    "Include input validation",
    "Add type hints",
    "Handle errors properly",
    "Return appropriate status codes"
  ]
}

Step 2 - Review output:
[Examine generated code thoroughly]
Issues found:
- Missing request body validation
- No 404 handling for task not found
- Inconsistent response format

Step 3 - Provide feedback:
{
  "task_id": "task-api-001",
  "issues": [
    "Missing Pydantic model for request body validation",
    "No 404 error handling for invalid task IDs",
    "Response format inconsistent across endpoints"
  ],
  "suggestions": [
    "Add TaskCreate and TaskUpdate Pydantic models",
    "Raise HTTPException(404) when task not found",
    "Use consistent response format: {\"data\": ..., \"message\": ...}"
  ],
  "approve": false
}

Step 4 - Review iteration 2:
[Check if issues addressed]
If good → approve: true
If issues remain → iterate again (max 3 times)

Expected: 2-5 minutes total, high quality output
```

### Template 3: Complex Task (Handle Yourself)

```
Task: "Design authentication system with JWT refresh tokens"

Decision: High complexity + security critical
Action: Handle myself

Reason:
- Architecture decisions required
- Security implications
- Multiple components to coordinate
- Local model unlikely to get it right without extensive guidance

Do it yourself, possibly delegating small pieces:
1. Design overall architecture (YOU)
2. Generate JWT helper boilerplate (DELEGATE - quick)
3. Implement token validation logic (YOU - security critical)
4. Create auth endpoints (DELEGATE - with feedback)
5. Integration and testing (YOU)
```

## 🎯 Quality Review Checklist

When reviewing local model output, check for:

### Code Quality
- [ ] Follows language conventions and style
- [ ] Proper naming conventions
- [ ] Comments where necessary (not excessive)
- [ ] No obvious bugs or logic errors

### Completeness
- [ ] All requirements addressed
- [ ] Edge cases handled
- [ ] Error handling present
- [ ] Type hints included (if applicable)

### Best Practices
- [ ] Security considerations addressed
- [ ] Performance considerations (no obvious inefficiencies)
- [ ] Maintainability (code is readable and modular)
- [ ] Testing considerations (testable code)

### Specific to Task
- [ ] Matches project patterns and style
- [ ] Integrates with existing codebase
- [ ] Documentation adequate
- [ ] All quality criteria met

## 📝 Feedback Guidelines

### Good Feedback (Specific & Actionable)

```json
{
  "issues": [
    "Function `validate_email` at line 23 doesn't handle edge case of emails with + character",
    "Missing type hint for return value of `process_user` function",
    "Error handling in `save_to_db` catches all exceptions, should catch specific SQLAlchemy errors"
  ],
  "suggestions": [
    "Use a regex pattern that includes \\+ in the email validation: r'^[a-zA-Z0-9+._%-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'",
    "Add return type hint: def process_user(user: User) -> ProcessedUser:",
    "Replace 'except Exception' with 'except (IntegrityError, OperationalError) as e'"
  ]
}
```

### Bad Feedback (Vague)

```json
{
  "issues": [
    "Code quality could be better",
    "Some functions need improvement",
    "Error handling is not good"
  ],
  "suggestions": [
    "Make it better",
    "Fix the issues",
    "Improve error handling"
  ]
}
```

## 🎓 Learning from Iterations

Track patterns in what needs fixing:

**Common Issues:**
- Local models often miss edge cases
- Error handling tends to be too broad
- Input validation often incomplete
- Type hints sometimes forgotten

**Mitigation:**
- Include these in `quality_criteria` upfront
- Use specific `system_prompt` emphasizing these
- First feedback round: focus on structure
- Second feedback round: focus on details

## 📊 Efficiency Optimization

### When to Stop Iterating

**After Iteration 1:**
- ✅ If output is perfect or near-perfect
- ✅ If minor tweaks you can make faster yourself

**After Iteration 2:**
- ✅ If quality is acceptable for the task
- ✅ If remaining issues are minor

**After Iteration 3 (Max):**
- ⚠️ If still not acceptable, take over yourself
- 📝 Note: This task type might not be suitable for delegation

### Batch Similar Tasks

If you have multiple similar tasks, delegate them in parallel:

```
Task: "Create 5 similar API endpoints"

Instead of:
1. Delegate endpoint 1 → review → iterate
2. Delegate endpoint 2 → review → iterate
...

Do:
1. Delegate endpoint 1 → review → iterate → approve
2. Use learnings to delegate endpoints 2-5 with better initial prompt
3. Review batch, likely need fewer iterations
```

## 🚀 Progressive Complexity

Start session with simple delegations to "warm up" the system:

```
Session Start:
1. Simple task (boilerplate) → Quick delegation
   - Builds confidence in system
   - Verifies connection works

2. Medium task (feature) → Feedback loop
   - Test review process
   - Establish communication pattern

3. Complex task → Handle yourself or careful delegation
   - Now in rhythm
   - Know what to expect
```

## 💡 Cost-Quality Trade-offs

### Estimated Costs (per task)

**You handling directly:**
- Cost: 100% (baseline)
- Quality: 95-100%
- Time: Variable

**Quick delegation:**
- Cost: 10-15% (local model only)
- Quality: 70-85%
- Time: Fast (30s-2min)
- Use when: Quality bar is lower or easily fixable

**Feedback loop (2 iterations):**
- Cost: 40-50% (local model + your review)
- Quality: 85-95%
- Time: Medium (3-8min)
- Use when: Quality matters, task is delegatable

**Feedback loop (3 iterations):**
- Cost: 60-70%
- Quality: 90-100%
- Time: Longer (8-15min)
- Use when: Quality critical but task is still delegatable

### Target Mix for Optimal Efficiency

**Ideal distribution:**
- 20% You directly (critical/complex)
- 30% Quick delegation (simple/low-risk)
- 50% Feedback loop (medium complexity)

**Result:**
- ~60-70% cost reduction
- Quality maintained
- Faster overall throughput

## 🎯 Contextual Factors

### Consider Codebase Maturity

**Early stage project:**
- More direct involvement (architecture forming)
- Delegate simple utilities and boilerplate
- Review very carefully

**Mature project:**
- More delegation (patterns established)
- Local model can follow existing patterns
- Reviews can be faster

### Consider User Expertise

**Expert user:**
- Can fix minor issues themselves
- Higher delegation rate acceptable
- Focus on architecture and complex logic

**Beginner user:**
- Needs higher quality output
- More thorough reviews
- More direct involvement

## 📈 Success Metrics

Track and optimize:

1. **Delegation Rate**: % of tasks delegated
   - Target: 60-80%

2. **First-iteration Approval Rate**: % approved without feedback
   - Target: 30-40% (means good task selection)

3. **Average Iterations**: How many rounds needed
   - Target: 1.5-2.0

4. **Cost Savings**: Reduction in API costs
   - Target: 60-70%

5. **Quality Score**: User satisfaction with outputs
   - Target: Maintain or improve vs. full remote

## 🎓 Advanced Techniques

### 1. Learned Prompting

After several sessions, you'll notice patterns. Build a mental library:

```
"For FastAPI endpoints, always emphasize:"
- Pydantic models for validation
- HTTPException for errors
- Dependency injection pattern
- Async/await usage

"For React components, always emphasize:"
- TypeScript strict mode
- Props interface
- Error boundaries
- Accessibility
```

### 2. Incremental Delegation

For complex tasks, delegate in phases:

```
Phase 1: Generate structure (DELEGATE)
Phase 2: Review and refine structure (YOU)
Phase 3: Implement functions (DELEGATE with feedback)
Phase 4: Integration and testing (YOU)
```

### 3. Parallel Delegation

Delegate multiple independent tasks simultaneously:

```
While waiting for backend endpoint feedback:
- Delegate frontend component
- Delegate test cases
- Delegate documentation

Then review all in batch
```

## 🎯 Final Principles

1. **Trust but Verify**: Delegate confidently, but always review
2. **Specific Feedback**: Be precise in what needs fixing
3. **Know When to Stop**: Don't iterate endlessly
4. **Learn Patterns**: Get better at delegation over time
5. **Maintain Quality**: Never compromise on critical code
6. **Optimize for Value**: Delegate what saves most time/cost
7. **Stay in Control**: You're the senior dev, final decisions are yours

---

**Remember**: This is a partnership. Local models do the typing, you do the thinking and quality assurance. Together, you're more efficient than either alone.
