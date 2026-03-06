"""Session management for WebUI."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from envs.env_loader import load_env
from envs.state import EnvState
from decision_adapters.greedy import GreedyDecisionAdapter
from decision_adapters.greedyv2 import GreedyV2DecisionAdapter
from decision_adapters.greedyv3 import GreedyV3DecisionAdapter
from decision_adapters.alphachip import AlphaChipDecisionAdapter
from decision_adapters.maskplace import MaskPlaceDecisionAdapter
from envs.action_space import ActionSpace as CandidateSet

from agents.greedy import GreedyAgent
from ordering_agents import DifficultyOrderingAgent
from agents.alphachip.agent import AlphaChipAgent
from agents.maskplace import MaskPlaceAgent

from search.mcts import MCTSConfig, MCTSSearch
from search.beam import BeamConfig, BeamSearch

from pipeline import DecisionPipeline

from webui.schemas import (
    SessionCreateRequest,
    SessionState,
    PlacedFacility,
    CandidateInfo,
    SearchProgress,
    ZoneRect,
    FlowEdge,
)


@dataclass
class HistoryEntry:
    """A state checkpoint for undo/redo."""
    state: Dict[str, Any]
    candidates: Optional[CandidateSet]
    scores: Optional[np.ndarray]
    value: float
    cost: float


@dataclass
class Session:
    """A single interactive session."""
    sid: str
    env: Any  # Wrapper env
    agent: Any
    search: Any
    pipeline: DecisionPipeline
    device: torch.device
    reset_kwargs: Dict[str, Any]
    
    # State
    obs: Any = None
    terminated: bool = False
    truncated: bool = False
    
    # Current candidates
    candidates: Optional[CandidateSet] = None
    scores: Optional[np.ndarray] = None
    value: float = 0.0
    
    # History for undo/redo
    history: List[HistoryEntry] = field(default_factory=list)
    history_index: int = -1  # Points to current state in history
    
    # WebSocket connections
    websockets: List[Any] = field(default_factory=list)
    
    # Lock for thread safety
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _save_to_history(self) -> None:
        """Save current state to history (for undo)."""
        state = {
            "engine": self.pipeline.engine.get_state().copy(),
            "adapter": self.env.get_state_copy(),
        }
        entry = HistoryEntry(
            state=state,
            candidates=self.candidates,
            scores=self.scores.copy() if self.scores is not None else None,
            value=self.value,
            cost=float(self.pipeline.engine.cost()),
        )
        # Truncate future history if we're not at the end
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(entry)
        self.history_index = len(self.history) - 1

    def _restore_from_history(self, index: int) -> None:
        """Restore state from history."""
        if index < 0 or index >= len(self.history):
            return
        entry = self.history[index]
        ss = entry.state
        if isinstance(ss, dict):
            eng = ss.get("engine", None)
            adp = ss.get("adapter", None)
            if isinstance(eng, EnvState):
                self.pipeline.engine.set_state(eng)
            if isinstance(adp, dict):
                self.env.set_state(adp)
        self.candidates = entry.candidates
        self.scores = entry.scores
        self.value = entry.value
        self.history_index = index

    def can_undo(self) -> bool:
        return self.history_index > 0

    def can_redo(self) -> bool:
        return self.history_index < len(self.history) - 1

    def undo(self) -> bool:
        if not self.can_undo():
            return False
        self._restore_from_history(self.history_index - 1)
        return True

    def redo(self) -> bool:
        if not self.can_redo():
            return False
        self._restore_from_history(self.history_index + 1)
        return True

    def _update_candidates(self) -> None:
        """Update candidates and scores from current observation."""
        engine = self.pipeline.engine
        cur = getattr(self.env, "current_gid", None)
        next_gid = cur() if callable(cur) else (engine.get_state().remaining[0] if engine.get_state().remaining else None)
        
        if isinstance(self.obs, dict) and "action_mask" in self.obs:
            if "action_xyrot" in self.obs:
                self.candidates = CandidateSet(
                    xyrot=self.obs["action_xyrot"],
                    mask=self.obs["action_mask"],
                    gid=next_gid,
                )
            else:
                self.candidates = None
        else:
            self.candidates = None
        
        # Get policy scores and value from agent
        if self.candidates is not None:
            self.scores = self.agent.policy(
                env=engine,
                obs=self.obs,
                candidates=self.candidates,
            ).detach().cpu().numpy()
            self.value = float(self.agent.value(
                env=engine,
                obs=self.obs,
                candidates=self.candidates,
            ))
        else:
            self.scores = None
            self.value = 0.0

    def get_state(self) -> SessionState:
        """Build SessionState for API response."""
        engine = self.env.engine
        
        # Placed facilities
        placed = []
        for gid in engine.get_state().placed:
            p = engine.get_state().placements[gid]
            x_bl, y_bl, rot = p.pose()
            group = engine.group_specs[gid]
            w, h = group._rotated_size(int(rot))
            placed.append(PlacedFacility(
                gid=str(gid),
                x=float(x_bl),
                y=float(y_bl),
                w=float(w),
                h=float(h),
                rot=int(rot),
            ))
        
        # Candidates
        candidates = []
        if self.candidates is not None:
            mask = self.candidates.mask.detach().cpu().numpy()
            xyrot = self.candidates.xyrot.detach().cpu().numpy()
            scores = self.scores if self.scores is not None else np.zeros(len(xyrot))
            
            for i in range(len(xyrot)):
                x_bl, y_bl, rot = xyrot[i]
                # Convert to center for display
                if self.candidates.gid is not None:
                    cx, cy = engine.group_specs[self.candidates.gid]._center_from_bl(
                        int(x_bl), int(y_bl), int(rot)
                    )
                else:
                    cx, cy = float(x_bl), float(y_bl)
                
                candidates.append(CandidateInfo(
                    index=i,
                    x=float(cx),
                    y=float(cy),
                    rot=int(rot),
                    score=float(scores[i]) if i < len(scores) else 0.0,
                    valid=bool(mask[i]),
                    visits=0,
                    q_value=0.0,
                ))
        
        current_gid = engine.get_state().remaining[0] if engine.get_state().remaining else None
        
        # Extract zones
        def _extract_zones(areas_attr: str) -> list:
            zones = []
            areas = getattr(engine, areas_attr, None)
            if areas and isinstance(areas, list):
                for a in areas:
                    if isinstance(a, dict) and "rect" in a:
                        r = a["rect"]
                        zones.append(ZoneRect(
                            x0=float(r[0]), y0=float(r[1]),
                            x1=float(r[2]), y1=float(r[3]),
                            value=float(a.get("value")) if a.get("value") is not None else None,
                            id=str(a.get("id")) if a.get("id") is not None else None,
                        ))
            return zones
        
        # Extract forbidden_areas from engine
        forbidden_areas_out = _extract_zones("forbidden_areas")
        
        # Extract flow edges with positions
        flow_edges = []
        for src, targets in engine.group_flow.items():
            for dst, weight in targets.items():
                edge = FlowEdge(src=str(src), dst=str(dst), weight=float(weight))
                # Add positions if both are placed
                if src in engine.get_state().placed:
                    p_src = engine.get_state().placements[src]
                    sx = float(getattr(p_src, "cx"))
                    sy = float(getattr(p_src, "cy"))
                    edge.src_x = float(sx)
                    edge.src_y = float(sy)
                if dst in engine.get_state().placed:
                    p_dst = engine.get_state().placements[dst]
                    dx = float(getattr(p_dst, "cx"))
                    dy = float(getattr(p_dst, "cy"))
                    edge.dst_x = float(dx)
                    edge.dst_y = float(dy)
                flow_edges.append(edge)
        
        return SessionState(
            grid_width=int(engine.grid_width),
            grid_height=int(engine.grid_height),
            placed=placed,
            remaining=[str(g) for g in engine.get_state().remaining],
            current_gid=str(current_gid) if current_gid else None,
            candidates=candidates,
            value=float(self.value),
            cost=float(engine.cost()),
            step=len(engine.get_state().placed),
            history_length=len(self.history),
            terminated=self.terminated or len(engine.get_state().remaining) == 0,
            can_undo=self.can_undo(),
            can_redo=self.can_redo(),
            forbidden_areas=forbidden_areas_out,
            placement_zones=_extract_zones("placement_areas"),
            weight_zones=_extract_zones("weight_areas"),
            dry_zones=_extract_zones("dry_areas"),
            height_zones=_extract_zones("height_areas"),
            flow_edges=flow_edges,
        )


class SessionManager:
    """Manages multiple sessions."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._counter = 0
        self._lock = asyncio.Lock()
    
    def _get_param(self, params: dict, prefix: str, name: str, default=None):
        """Get parameter from dynamic params dict."""
        key = f"{prefix}_{name}"
        return params.get(key, default)
    
    async def create_session(self, req: SessionCreateRequest) -> Session:
        """Create a new session."""
        async with self._lock:
            self._counter += 1
            sid = f"session_{self._counter}"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        params = req.params or {}
        
        # Load environment
        loaded = load_env(req.env_json, device=device)
        engine = loaded.env
        engine.log = False  # Disable logging for web
        
        # Create wrapper with dynamic params
        wrapper_params = {k.replace('wrapper_', ''): v 
                        for k, v in params.items() if k.startswith('wrapper_')}
        
        if req.wrapper_mode == "greedy":
            env = GreedyDecisionAdapter(
                k=wrapper_params.get('k', 50),
                scan_step=wrapper_params.get('scan_step', 2000.0),
                quant_step=wrapper_params.get('quant_step', 10.0),
                p_high=wrapper_params.get('p_high', 0.1),
                p_near=wrapper_params.get('p_near', 0.8),
                p_coarse=wrapper_params.get('p_coarse', 0.0),
                oversample_factor=wrapper_params.get('oversample_factor', 2),
                random_seed=42,
            )
        elif req.wrapper_mode == "greedyv2":
            env = GreedyV2DecisionAdapter(
                k=wrapper_params.get('k', 50),
                scan_step=wrapper_params.get('scan_step', 2000.0),
                quant_step=wrapper_params.get('quant_step', 10.0),
                p_high=wrapper_params.get('p_high', 0.1),
                p_near=wrapper_params.get('p_near', 0.1),
                p_coarse=wrapper_params.get('p_coarse', 0.0),
                oversample_factor=wrapper_params.get('oversample_factor', 2),
                random_seed=42,
            )
        elif req.wrapper_mode == "greedyv3":
            env = GreedyV3DecisionAdapter(
                k=wrapper_params.get('k', 50),
                quant_step=wrapper_params.get('quant_step', 10.0),
                oversample_factor=wrapper_params.get('oversample_factor', 2),
                edge_ratio=wrapper_params.get('edge_ratio', 0.8),
                random_seed=42,
            )
        elif req.wrapper_mode == "alphachip":
            env = AlphaChipDecisionAdapter(
                coarse_grid=wrapper_params.get('coarse_grid', 128),
                rot=0,
            )
        elif req.wrapper_mode == "maskplace":
            env = MaskPlaceDecisionAdapter(
                grid=wrapper_params.get('grid', 224),
                rot=0,
                soft_coefficient=wrapper_params.get('soft_coefficient', 1.0),
            )
        else:
            raise ValueError(f"Unknown wrapper_mode: {req.wrapper_mode}")
        
        # Create agent with dynamic params
        agent_params = {k.replace('agent_', ''): v 
                       for k, v in params.items() if k.startswith('agent_')}
        
        if req.agent_mode == "greedy":
            agent = GreedyAgent(
                prior_temperature=agent_params.get('prior_temperature', 1.0)
            )
        elif req.agent_mode == "alphachip":
            if req.wrapper_mode != "alphachip":
                raise ValueError(
                    f"AlphaChipAgent requires wrapper_mode='alphachip', got '{req.wrapper_mode}'"
                )
            agent = AlphaChipAgent(
                coarse_grid=wrapper_params.get('coarse_grid', 128),
                checkpoint_path=agent_params.get('checkpoint_path'),
                device=device,
            )
        elif req.agent_mode == "maskplace":
            if req.wrapper_mode != "maskplace":
                raise ValueError(
                    f"MaskPlaceAgent requires wrapper_mode='maskplace', got '{req.wrapper_mode}'"
                )
            agent = MaskPlaceAgent(
                device=device,
                grid=wrapper_params.get('grid', 224),
                soft_coefficient=wrapper_params.get('soft_coefficient', 1.0),
                checkpoint_path=agent_params.get('checkpoint_path'),
            )
        else:
            raise ValueError(f"Unknown agent_mode: {req.agent_mode}")
        
        # Create search with dynamic params
        search_params = {k.replace('search_', ''): v 
                        for k, v in params.items() if k.startswith('search_')}
        
        if req.search_mode == "none":
            search = None
        elif req.search_mode == "mcts":
            search = MCTSSearch(
                config=MCTSConfig(
                    num_simulations=search_params.get('num_simulations', 50),
                    c_puct=search_params.get('c_puct', 2.0),
                    rollout_enabled=search_params.get('rollout_enabled', True),
                    rollout_depth=search_params.get('rollout_depth', 5),
                    dirichlet_epsilon=search_params.get('dirichlet_epsilon', 0.2),
                    dirichlet_concentration=search_params.get('dirichlet_concentration', 0.5),
                    temperature=search_params.get('temperature', 0.0),
                    pw_enabled=search_params.get('pw_enabled', False),
                    pw_c=search_params.get('pw_c', 1.5),
                    pw_alpha=search_params.get('pw_alpha', 0.5),
                    pw_min_children=search_params.get('pw_min_children', 1),
                )
            )
        elif req.search_mode == "beam":
            search = BeamSearch(
                config=BeamConfig(
                    beam_width=search_params.get('beam_width', 8),
                    depth=search_params.get('depth', 5),
                    expansion_topk=search_params.get('expansion_topk', 16),
                )
            )
        else:
            raise ValueError(f"Unknown search_mode: {req.search_mode}")

        ordering_mode = str(params.get("ordering_mode", "none"))
        if ordering_mode == "none":
            ordering_agent = None
        elif ordering_mode == "difficulty":
            ordering_agent = DifficultyOrderingAgent()
        else:
            raise ValueError(f"Unknown ordering_mode: {ordering_mode}")

        pipeline = DecisionPipeline(
            agent=agent,
            engine=engine,
            adapter=env,
            search=search,
            ordering_agent=ordering_agent,
        )
        env.bind(engine)
        
        session = Session(
            sid=sid,
            env=env,
            agent=agent,
            search=search,
            pipeline=pipeline,
            device=device,
            reset_kwargs=loaded.reset_kwargs,
        )
        
        # Reset environment
        _obs_core, _ = engine.reset(options=loaded.reset_kwargs)
        session.obs = env.build_observation(_obs_core)
        session._update_candidates()
        session._save_to_history()
        
        async with self._lock:
            self._sessions[sid] = session
        
        return session
    
    async def get_session(self, sid: str) -> Session:
        """Get session by ID."""
        async with self._lock:
            if sid not in self._sessions:
                raise KeyError(f"Session not found: {sid}")
            return self._sessions[sid]
    
    async def delete_session(self, sid: str) -> None:
        """Delete a session."""
        async with self._lock:
            if sid in self._sessions:
                del self._sessions[sid]
    
    async def list_sessions(self) -> List[str]:
        """List all session IDs."""
        async with self._lock:
            return list(self._sessions.keys())


# Global session manager
manager = SessionManager()
