from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core import Columns

from agents.greedy import GreedyAgent, GreedyConfig
try:
    from agents.ppo import CandidateRLModule, PPOAgent, PPOAgentConfig  # type: ignore
except ImportError:  # legacy RLlib PPO code removed
    CandidateRLModule = None  # type: ignore
    PPOAgent = None  # type: ignore
    PPOAgentConfig = None  # type: ignore

from envs.env_old import Candidate, FactoryLayoutEnvOld, FacilityGroup
from policies.mcts import MCTSAgent, MCTSConfig, MCTSNode
from policies.topk_selector import TopKConfig, TopKSelector

GroupId = Union[int, str]


def _build_demo_groups() -> Tuple[Dict[GroupId, FacilityGroup], Dict[GroupId, Dict[GroupId, float]]]:
    groups = {
        "A": FacilityGroup(id="A", width=80, height=40),
        "B": FacilityGroup(id="B", width=60, height=60, rotatable=False),
        "C": FacilityGroup(id="C", width=50, height=30),
        "D": FacilityGroup(id="D", width=40, height=40),
        "E": FacilityGroup(id="E", width=70, height=50),
        "F": FacilityGroup(id="F", width=90, height=30, rotatable=False),
        "G": FacilityGroup(id="G", width=35, height=35),
        "H": FacilityGroup(id="H", width=120, height=50),
        "I": FacilityGroup(id="I", width=55, height=45),
    }
    flow = {
        "A": {"B": 1.0, "D": 0.6},
        "B": {"C": 1.0, "E": 0.4},
        "C": {"F": 0.7},
        "D": {"E": 0.5, "G": 0.3},
        "E": {"H": 0.6},
        "F": {"I": 0.4},
    }
    return groups, flow


def _load_module(checkpoint_path: str) -> CandidateRLModule:
    algo = Algorithm.from_checkpoint(checkpoint_path)
    return algo.get_module()


@dataclass
class SessionState:
    env: FactoryLayoutEnvOld
    agent: object
    mcts: Optional[MCTSAgent]
    candidates: List[Candidate] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int8))
    root: Optional[MCTSNode] = None
    root_snapshot: Optional[Dict[str, object]] = None
    selector_state: Optional[object] = None
    sockets: List[WebSocket] = field(default_factory=list)


class InteractiveMCTS(MCTSAgent):
    def init_root(self, env: FactoryLayoutEnvOld, candidates: List[Candidate], mask: np.ndarray) -> MCTSNode:
        self._root_snapshot = env.get_snapshot()
        self._selector_state = self.base_agent.get_selector_state()
        priors = self.base_agent.get_action_priors(env, candidates, mask)
        if len(priors) != len(candidates):
            priors = np.zeros((len(candidates),), dtype=np.float32)
            valid = mask.astype(bool)
            if valid.any():
                priors[valid] = 1.0 / float(np.sum(valid))
        priors = self._apply_root_dirichlet(priors, mask)
        return MCTSNode(env.get_snapshot(), candidates, mask, priors=priors, prior=1.0)

    def run(self, env: FactoryLayoutEnvOld, root: MCTSNode, sims: int) -> None:
        for _ in range(max(int(sims), 0)):
            self._simulate(env, root)
        env.set_snapshot(self._root_snapshot)
        self.base_agent.set_selector_state(self._selector_state)


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, model: str, use_mcts: bool, checkpoint: Optional[str]) -> str:
        async with self._lock:
            groups, flow = _build_demo_groups()
            env = FactoryLayoutEnvOld(
                grid_width=500,
                grid_height=500,
                grid_size=1.0,
                groups=groups,
                group_flow=flow,
                max_candidates=70,
                max_steps=200,
            )

            topk_cfg = TopKConfig(
                k=70,
                scan_step=5.0,
                quant_step=10.0,
                p_high=0.1,
                p_near=0.9,
                p_coarse=0.0,
                oversample_factor=2,
                diversity_ratio=0.0,
                min_diversity=10,
                random_seed=7,
            )

            if model == "greedy":
                agent = GreedyAgent(GreedyConfig(topk_config=topk_cfg))
            elif model == "ppo_topk":
                if checkpoint is None:
                    raise ValueError("checkpoint is required for ppo_topk")
                if PPOAgent is None or PPOAgentConfig is None or CandidateRLModule is None:
                    raise RuntimeError(
                        "ppo_topk is not available: legacy RLlib PPO code (`agents/ppo.py`) is missing in this repo."
                    )
                ray.init(ignore_reinit_error=True, include_dashboard=False)
                module = _load_module(checkpoint)
                selector = TopKSelector(topk_cfg)
                agent = PPOAgent(module, selector, PPOAgentConfig(use_argmax=True))
            else:
                raise ValueError(f"unknown model: {model}")

            mcts = None
            if use_mcts:
                mcts = InteractiveMCTS(agent, MCTSConfig(num_simulations=50, c_puct=1.0, rollout_depth=5))

            env.reset()
            candidates, mask = agent.get_candidates(env)
            session = SessionState(env=env, agent=agent, mcts=mcts, candidates=candidates, mask=mask)
            sid = uuid.uuid4().hex[:8]
            self._sessions[sid] = session
            return sid

    async def get(self, sid: str) -> SessionState:
        async with self._lock:
            if sid not in self._sessions:
                raise KeyError("session not found")
            return self._sessions[sid]


manager = SessionManager()
app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.get("/")
async def index():
    return FileResponse("web/static/index.html")


def _serialize_state(session: SessionState) -> Dict:
    env = session.env
    placed = []
    for gid in env.placed:
        x, y, rot = env.positions[gid]
        group = env.groups[gid]
        w, h = env.rotated_size(group, rot)
        placed.append({"id": gid, "x": x, "y": y, "rot": rot, "w": w, "h": h})

    candidates = []
    for i, cand in enumerate(session.candidates):
        score = env.estimate_delta_obj(cand.group_id, cand.x, cand.y, cand.rot)
        candidates.append(
            {
                "index": i,
                "id": cand.group_id,
                "x": cand.x,
                "y": cand.y,
                "rot": cand.rot,
                "score": float(score),
                "mask": int(session.mask[i]) if i < len(session.mask) else 0,
            }
        )

    visits = []
    if session.root is not None:
        visits = [0 for _ in range(len(session.candidates))]
        for action, child in session.root.children.items():
            if 0 <= action < len(visits):
                visits[action] = int(child.visits)

    policy_v = 0.0
    if isinstance(session.agent, PPOAgent):
        env.set_candidates(session.candidates)
        obs = env._build_observation()
        obs_batch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            fwd_out = session.agent.module.forward_inference({"obs": obs_batch})
        vf = fwd_out[Columns.VF_PREDS]
        if vf.numel() > 0:
            policy_v = float(vf.flatten().mean().item())
    elif isinstance(session.agent, GreedyAgent):
        policy_v = -float(env.cal_obj())

    mcts_v = 0.0
    if session.root is not None and session.root.visits > 0:
        mcts_v = float(session.root.total_value / session.root.visits)

    return {
        "grid": {"w": env.grid_width, "h": env.grid_height},
        "placed": placed,
        "candidates": candidates,
        "visits": visits,
        "obj": float(env.cal_obj()),
        "policy_v": policy_v,
        "mcts_v": mcts_v,
        "remaining": len(env.remaining),
        "terminated": len(env.remaining) == 0,
    }


async def _broadcast(session: SessionState, payload: Dict) -> None:
    if not session.sockets:
        return
    message = json.dumps(payload)
    for ws in list(session.sockets):
        try:
            await ws.send_text(message)
        except Exception:
            try:
                session.sockets.remove(ws)
            except ValueError:
                pass


@app.post("/session/start")
async def start_session(payload: Dict):
    model = payload.get("model", "greedy")
    use_mcts = bool(payload.get("use_mcts", True))
    checkpoint = payload.get("checkpoint")
    sid = await manager.create_session(model=model, use_mcts=use_mcts, checkpoint=checkpoint)
    session = await manager.get(sid)
    state = _serialize_state(session)
    return {"session_id": sid, "state": state}


@app.post("/session/step")
async def step_session(payload: Dict):
    sid = payload.get("session_id")
    action = int(payload.get("action", 0))
    session = await manager.get(sid)

    session.root = None
    session.root_snapshot = None
    session.selector_state = None

    session.env.set_candidates(session.candidates)
    session.env.step(action)
    session.candidates, session.mask = session.agent.get_candidates(session.env)
    state = _serialize_state(session)
    await _broadcast(session, {"type": "state", "state": state})
    return {"state": state}


@app.post("/session/mcts")
async def run_mcts(payload: Dict):
    sid = payload.get("session_id")
    sims = int(payload.get("sims", 50))
    session = await manager.get(sid)
    if session.mcts is None:
        return {"error": "mcts is disabled"}

    if session.root is None:
        session.root = session.mcts.init_root(session.env, session.candidates, session.mask)

    session.mcts.run(session.env, session.root, sims)
    state = _serialize_state(session)
    await _broadcast(session, {"type": "state", "state": state})
    return {"state": state}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    session = await manager.get(session_id)
    session.sockets.append(ws)
    try:
        await ws.send_text(json.dumps({"type": "state", "state": _serialize_state(session)}))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in session.sockets:
            session.sockets.remove(ws)
