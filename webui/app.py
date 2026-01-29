"""FastAPI WebUI application for Factory Layout."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from envs.wrappers.candidate_set import CandidateSet

from webui.schemas import (
    SessionCreateRequest,
    StepRequest,
    SearchRequest,
    SessionState,
    CandidateInfo,
    SearchProgress,
)
from webui.session import manager, Session


app = FastAPI(title="Factory Layout WebUI")

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the main HTML page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/configs")
async def list_configs():
    """List available environment config files."""
    configs = []
    
    # Check env_configs directory
    env_configs_dir = Path("env_configs")
    if env_configs_dir.exists():
        for f in env_configs_dir.glob("*.json"):
            configs.append(str(f))
    
    # Check converters directory
    converters_dir = Path("converters")
    if converters_dir.exists():
        for f in converters_dir.glob("*.json"):
            configs.append(str(f))
    
    return {"configs": configs}


@app.post("/api/session/create")
async def create_session(req: SessionCreateRequest):
    """Create a new session."""
    try:
        session = await manager.create_session(req)
        state = session.get_state()
        return {"session_id": session.sid, "state": state.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/session/{sid}/state")
async def get_state(sid: str):
    """Get current session state."""
    try:
        session = await manager.get_session(sid)
        state = session.get_state()
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/step")
async def step(sid: str, req: StepRequest):
    """Execute a step with the given action."""
    try:
        session = await manager.get_session(sid)
        
        async with session._lock:
            # Save state before step
            session._save_to_history()
            
            # Execute step
            session.obs, reward, session.terminated, session.truncated, info = \
                session.env.step(req.action)
            
            # Update candidates for new state
            session._update_candidates()
            
            state = session.get_state()
        
        # Broadcast to WebSocket clients
        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        
        return {"state": state.model_dump(), "reward": float(reward)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/session/{sid}/undo")
async def undo(sid: str):
    """Undo the last step."""
    try:
        session = await manager.get_session(sid)
        
        async with session._lock:
            if not session.undo():
                raise HTTPException(status_code=400, detail="Nothing to undo")
            state = session.get_state()
        
        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/redo")
async def redo(sid: str):
    """Redo the last undone step."""
    try:
        session = await manager.get_session(sid)
        
        async with session._lock:
            if not session.redo():
                raise HTTPException(status_code=400, detail="Nothing to redo")
            state = session.get_state()
        
        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/reset")
async def reset(sid: str):
    """Reset the session to initial state."""
    try:
        session = await manager.get_session(sid)
        
        async with session._lock:
            session.obs, _ = session.env.reset(options=session.reset_kwargs)
            session.terminated = False
            session.truncated = False
            session._update_candidates()
            session.history = []
            session.history_index = -1
            session._save_to_history()
            state = session.get_state()
        
        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/search")
async def run_search(sid: str, req: SearchRequest):
    """Run MCTS/Beam search with real-time updates."""
    try:
        session = await manager.get_session(sid)
        
        if session.search is None:
            raise HTTPException(status_code=400, detail="No search algorithm configured")
        
        if session.candidates is None:
            raise HTTPException(status_code=400, detail="No candidates available")
        
        # Run search with progress callback
        async def progress_callback(sim: int, visits: np.ndarray, values: np.ndarray, best_action: int):
            """Called during search to broadcast progress."""
            # Build candidate info with visits/values
            candidates = []
            state = session.get_state()
            for i, cand in enumerate(state.candidates):
                cand.visits = int(visits[i]) if i < len(visits) else 0
                cand.q_value = float(values[i]) if i < len(values) else 0.0
                candidates.append(cand)
            
            progress = SearchProgress(
                simulation=sim,
                total=req.simulations,
                candidates=candidates,
                best_action=int(best_action),
                best_value=float(values[best_action]) if best_action < len(values) else 0.0,
            )
            await _broadcast(session, {"type": "search_progress", "progress": progress.model_dump()})
        
        # Run search in background with periodic updates
        result = await _run_search_with_updates(
            session=session,
            simulations=req.simulations,
            broadcast_interval=req.broadcast_interval,
            callback=progress_callback,
        )
        
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _run_search_with_updates(
    session: Session,
    simulations: int,
    broadcast_interval: int,
    callback,
) -> Dict[str, Any]:
    """Run search with periodic WebSocket updates using unified interface."""
    from search.base import BaseSearch, SearchProgress
    from search.mcts import MCTSSearch, MCTSConfig
    from search.beam import BeamSearch, BeamConfig
    
    search = session.search
    if search is None:
        return {"error": "No search algorithm configured"}
    
    if not isinstance(search, BaseSearch):
        # Fallback for non-BaseSearch implementations
        action, dbg, candidates = session.pipeline.act(
            env=session.env,
            obs=session.obs,
        )
        return {"best_action": int(action), "debug": dbg}
    
    candidates = session.candidates
    if candidates is None:
        return {"error": "No candidates"}
    
    # Save initial snapshot
    env = session.env
    initial_snapshot = env.get_snapshot()
    
    # Store latest progress for final response
    latest_progress: Dict[str, Any] = {}
    
    # Create async-compatible callback
    def sync_callback(progress: SearchProgress) -> None:
        """Synchronous callback that schedules async broadcast."""
        nonlocal latest_progress
        latest_progress = {
            "iteration": progress.iteration,
            "total": progress.total,
            "visits": progress.visits.tolist(),
            "values": progress.values.tolist(),
            "best_action": progress.best_action,
            "best_value": progress.best_value,
            "extra": progress.extra,
        }
        # Schedule async broadcast (fire and forget)
        asyncio.create_task(_broadcast_progress(session, progress, callback))
    
    # Configure search with progress callback
    # For MCTS, we need to adjust num_simulations
    if isinstance(search, MCTSSearch):
        # Temporarily override config for this run
        original_config = search.config
        search.config = MCTSConfig(
            num_simulations=simulations,
            c_puct=original_config.c_puct,
            rollout_enabled=original_config.rollout_enabled,
            rollout_depth=original_config.rollout_depth,
            dirichlet_epsilon=original_config.dirichlet_epsilon,
            dirichlet_concentration=original_config.dirichlet_concentration,
            temperature=original_config.temperature,
            pw_enabled=original_config.pw_enabled,
            pw_c=original_config.pw_c,
            pw_alpha=original_config.pw_alpha,
            pw_min_children=original_config.pw_min_children,
            track_top_k=original_config.track_top_k,
            track_verbose=original_config.track_verbose,
        )
    
    # Set progress callback
    search.set_progress_callback(sync_callback, interval=broadcast_interval)
    
    try:
        # Run search
        best_action = search.select(
            env=env,
            obs=session.obs,
            agent=session.agent,
            root_candidates=candidates,
        )
    finally:
        # Clear callback after search
        search.set_progress_callback(None)
        
        # Restore original config for MCTS
        if isinstance(search, MCTSSearch):
            search.config = original_config
    
    # Restore initial state
    env.set_snapshot(initial_snapshot)
    
    return {
        "best_action": int(best_action),
        **latest_progress,
    }


async def _broadcast_progress(session: Session, progress, callback) -> None:
    """Broadcast search progress to WebSocket clients."""
    try:
        await callback(
            progress.iteration,
            progress.visits,
            progress.values,
            progress.best_action,
        )
        await asyncio.sleep(0.001)  # Small yield to allow other tasks
    except Exception as e:
        print(f"[WebUI] Progress broadcast error: {e}")


@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    """Delete a session."""
    try:
        await manager.delete_session(sid)
        return {"status": "deleted"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.websocket("/ws/{sid}")
async def websocket_endpoint(websocket: WebSocket, sid: str):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    try:
        session = await manager.get_session(sid)
        session.websockets.append(websocket)
        
        # Send initial state
        state = session.get_state()
        await websocket.send_json({"type": "state", "state": state.model_dump()})
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                # Handle ping/pong or other messages
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except KeyError:
        await websocket.close(code=4004, reason="Session not found")
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))
    finally:
        try:
            session = await manager.get_session(sid)
            if websocket in session.websockets:
                session.websockets.remove(websocket)
        except:
            pass


async def _broadcast(session: Session, message: Dict[str, Any]) -> None:
    """Broadcast message to all WebSocket clients of a session."""
    dead = []
    for ws in session.websockets:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            session.websockets.remove(ws)
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
