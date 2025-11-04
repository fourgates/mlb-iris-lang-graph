# MLB Assistant Evaluation Queries

This document contains test queries for evaluating the MLB assistant agent across different routing paths and verification scenarios.

## Single-Domain Paths

### Player Stats Queries

These should route to `PLAYER_STATS` subgraph and pass verification.

- "What is Aaron Judge's current home run total?"
- "What are Shohei Ohtani's batting statistics?"
- "Tell me about Mike Trout's season stats"

### Document QA Queries

These should route to `DOCUMENT_QA` subgraph and pass verification.

- "According to the official rules, what is the injured list?"
- "What is the disabled list in MLB?"
- "What are the rules about player transactions?"

## Multi-Domain Paths

### Planner Agent Queries

These should route to `MULTI_DOMAIN` → `planner` and pass verification.

- "According to the official rules, what is the injured list, and what is Aaron Judge's current home run total?"
- "What is the disabled list and tell me about Shohei Ohtani's stats this season?"
- "Explain the injured list rules and give me Juan Soto's batting average"

## Edge Cases & REPLAN Testing

### Potentially Incomplete Answers

These queries might trigger REPLAN if the answer is incomplete.

- "Tell me everything about Aaron Judge and the injured list" (might miss details)
- "What are all the MLB rules about player injuries and give me complete stats for three players?" (might be too complex)
- "Give me comprehensive information about the disabled list and detailed stats for multiple players" (complex multi-part query)

### Replan Cap Testing

To test the MAX_REPLANS cap (currently 3):

- Ask a question that consistently fails verification to see if it stops after 3 attempts
- Monitor logs for: `[verify_answer] Max replans (3) reached, routing to END`

## Log Monitoring

When testing, watch for these log patterns:

### Verification Flow
- `[verify_answer] START` - Verification node starting
- `[verification] Judging answer for query=...` - Judge being called
- `[verification] Judge result: status=OK` or `status=REPLAN` - Verification result
- `[verify_answer] END status='OK'` or `status='REPLAN'` - Verification complete

### Replanning Flow
- `[verify_answer] Answer incomplete, incrementing replan_attempts: X -> Y`
- `[verify_answer] Routing to planner (attempt X/3)` - Replan triggered
- `[verify_answer] Max replans (3) reached, routing to END` - Cap reached

### Tool Usage (Planner Agent)
- `[tool] search_for_player called with player_name=...`
- `[tool] get_player_statistics called with player_id=...`
- `[tool] query_document_knowledge_base called with query=...`

## Expected Behavior

### Single-Domain Flow
1. Router → Subgraph (player_stats_sg or document_qa_sg)
2. Subgraph executes → verify_answer_node
3. Verification → END (if OK) or → planner (if REPLAN)
4. If REPLAN: planner → verify_answer_node → END (or repeat if still REPLAN)

### Multi-Domain Flow
1. Router → planner_node
2. Planner agent executes (may call tools multiple times)
3. planner_node → verify_answer_node
4. Verification → END (if OK) or → planner (if REPLAN)
5. If REPLAN: repeat planner → verify (max 3 attempts)

### Replan Cap
- Maximum 3 replan attempts before giving up
- After 3 attempts, returns best-effort answer even if verification fails

