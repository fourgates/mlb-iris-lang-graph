# Testing Interrupts

## Quick Start

To test the interrupt flow without needing ambiguous player names, enable the testing mode:

```bash
# Enable testing mode (all player queries trigger interrupt confirmation)
export ALWAYS_CONFIRM_PLAYER=true

# Run your tests or playground
make playground

# Or run the test script
ALWAYS_CONFIRM_PLAYER=true uv run python test_interrupt_backend.py
```

## Configuration

The `ALWAYS_CONFIRM_PLAYER` flag is controlled via environment variable:

- **Default:** `false` (normal behavior - only ambiguous names trigger interrupt)
- **Testing:** `true` (all player queries trigger interrupt confirmation)

Set in `app/config.py`:
```python
ALWAYS_CONFIRM_PLAYER = os.getenv("ALWAYS_CONFIRM_PLAYER", "false").lower() == "true"
```

## How It Works

When `ALWAYS_CONFIRM_PLAYER=true`:
1. `find_player_id()` returns a list (even for single matches) when `always_return_candidates=True`
2. `player_search_node()` detects the list and triggers interrupt
3. User must confirm player selection even for unambiguous names

## Rolling Back

To disable testing mode:
```bash
unset ALWAYS_CONFIRM_PLAYER
# or
export ALWAYS_CONFIRM_PLAYER=false
```

The flag is **disabled by default**, so simply not setting it restores normal behavior.

## Testing Results

✅ **Phase 1 Backend Complete:**
- Graph compiles with checkpointer ✓
- Interrupt detection works ✓
- Resume flow works ✓
- Testing mode successfully triggers interrupts ✓

## Next Steps

Phase 2: Frontend interrupt detection and UI components

