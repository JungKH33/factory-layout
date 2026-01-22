# Interactive UI (FastAPI + WebSocket)

## Run
```bash
uvicorn web.app:app --reload --port 8000
```

Open: http://localhost:8000

## Notes
- Default model is `greedy` with MCTS on.
- `ppo_topk` requires a checkpoint path in `/session/start`.
