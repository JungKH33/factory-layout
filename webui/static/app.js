/**
 * Factory Layout WebUI - Main Application
 */

// State
let sessionId = null;
let socket = null;
let currentState = null;
let hoveredCandidate = null;
let selectedCandidate = null;
let isDarkMode = false;

// Canvas
const canvas = document.getElementById('layout-canvas');
const ctx = canvas.getContext('2d');

// Settings
const settings = {
    showCandidates: true,
    showFlow: true,
    showForbidden: false,
    showPlacementZones: false,
    showWeightZones: false,
    showDryZones: false,
    showHeightZones: false,
    showScores: true,
    showVisits: false,
    showLabels: true,
    showGrid: false,
};

// Colors (theme-aware)
function getColors() {
    if (isDarkMode) {
        return {
            background: '#161b22',
            grid: '#30363d',
            placed: {
                fill: 'rgba(255, 165, 0, 0.6)',
                stroke: '#ffa500',
            },
            candidate: {
                valid: '#4ade80',
                invalid: '#f87171',
                hovered: '#60a5fa',
                selected: '#ff7f0e',
            },
            text: '#ffffff',
            flow: 'rgba(96, 165, 250, 0.6)',
            forbidden: 'rgba(255, 100, 100, 0.2)',
            placementZone: 'rgba(30, 144, 255, 0.15)',
            weightZone: 'rgba(31, 119, 180, 0.12)',
            dryZone: 'rgba(44, 160, 44, 0.10)',
            heightZone: 'rgba(127, 127, 127, 0.08)',
        };
    }
    return {
        background: '#ffffff',
        grid: '#e0e0e0',
        placed: {
            fill: 'rgba(255, 165, 0, 0.6)',
            stroke: '#000000',
        },
        candidate: {
            valid: '#2ca02c',
            invalid: '#d62728',
            hovered: '#1f77b4',
            selected: '#ff7f0e',
        },
        text: '#000000',
        flow: 'rgba(31, 119, 180, 0.5)',
        forbidden: 'rgba(255, 0, 0, 0.15)',
        placementZone: 'rgba(30, 144, 255, 0.12)',
        weightZone: 'rgba(31, 119, 180, 0.08)',
        dryZone: 'rgba(44, 160, 44, 0.06)',
        heightZone: 'rgba(127, 127, 127, 0.04)',
    };
}

// Shortcut for current colors
let COLORS = getColors();

// ============================================================
// Initialization
// ============================================================

function init() {
    setupEventListeners();
    loadConfigs();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        toggleTheme();
    }
}

function toggleTheme() {
    isDarkMode = !isDarkMode;
    COLORS = getColors();
    
    if (isDarkMode) {
        document.documentElement.setAttribute('data-theme', 'dark');
        document.getElementById('theme-toggle').textContent = '☀️';
        localStorage.setItem('theme', 'dark');
    } else {
        document.documentElement.removeAttribute('data-theme');
        document.getElementById('theme-toggle').textContent = '🌙';
        localStorage.setItem('theme', 'light');
    }
    
    render();
}

function setupEventListeners() {
    // Theme toggle
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    
    // Session controls
    document.getElementById('btn-create').addEventListener('click', createSession);
    document.getElementById('btn-reset').addEventListener('click', resetSession);
    document.getElementById('btn-undo').addEventListener('click', undoStep);
    document.getElementById('btn-redo').addEventListener('click', redoStep);
    document.getElementById('btn-search').addEventListener('click', runSearch);

    // Layer toggles
    const layerToggles = [
        ['show-candidates', 'showCandidates'],
        ['show-flow', 'showFlow'],
        ['show-forbidden', 'showForbidden'],
        ['show-placement-zones', 'showPlacementZones'],
        ['show-weight-zones', 'showWeightZones'],
        ['show-dry-zones', 'showDryZones'],
        ['show-height-zones', 'showHeightZones'],
        ['show-scores', 'showScores'],
        ['show-visits', 'showVisits'],
        ['show-labels', 'showLabels'],
        ['show-grid', 'showGrid'],
    ];
    
    layerToggles.forEach(([id, setting]) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', (e) => {
                settings[setting] = e.target.checked;
                render();
            });
        }
    });

    // Canvas interactions
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('mouseleave', onCanvasMouseLeave);
}

async function loadConfigs() {
    try {
        const res = await fetch('/api/configs');
        const data = await res.json();
        const select = document.getElementById('env-config');
        select.innerHTML = '';
        data.configs.forEach(config => {
            const opt = document.createElement('option');
            opt.value = config;
            opt.textContent = config;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load configs:', e);
    }
}

function resizeCanvas() {
    const container = canvas.parentElement;
    const maxW = container.clientWidth - 40;
    const maxH = container.clientHeight - 40;
    
    if (currentState) {
        const aspectRatio = currentState.grid_width / currentState.grid_height;
        if (maxW / aspectRatio <= maxH) {
            canvas.width = maxW;
            canvas.height = maxW / aspectRatio;
        } else {
            canvas.height = maxH;
            canvas.width = maxH * aspectRatio;
        }
    } else {
        canvas.width = Math.min(maxW, 800);
        canvas.height = Math.min(maxH, 600);
    }
    
    render();
}

// ============================================================
// Session Management
// ============================================================

async function createSession() {
    const req = {
        env_json: document.getElementById('env-config').value,
        wrapper_mode: document.getElementById('wrapper-mode').value,
        agent_mode: document.getElementById('agent-mode').value,
        search_mode: document.getElementById('search-mode').value,
        topk_k: parseInt(document.getElementById('topk-k').value),
    };

    try {
        const res = await fetch('/api/session/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req),
        });
        const data = await res.json();
        
        if (data.session_id) {
            sessionId = data.session_id;
            document.getElementById('session-id').textContent = sessionId;
            updateState(data.state);
            connectWebSocket();
            enableControls(true);
        }
    } catch (e) {
        console.error('Failed to create session:', e);
        alert('Failed to create session: ' + e.message);
    }
}

async function resetSession() {
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/api/session/${sessionId}/reset`, { method: 'POST' });
        const data = await res.json();
        updateState(data.state);
    } catch (e) {
        console.error('Failed to reset:', e);
    }
}

async function undoStep() {
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/api/session/${sessionId}/undo`, { method: 'POST' });
        const data = await res.json();
        updateState(data.state);
    } catch (e) {
        console.error('Failed to undo:', e);
    }
}

async function redoStep() {
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/api/session/${sessionId}/redo`, { method: 'POST' });
        const data = await res.json();
        updateState(data.state);
    } catch (e) {
        console.error('Failed to redo:', e);
    }
}

async function stepAction(action) {
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/api/session/${sessionId}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action }),
        });
        const data = await res.json();
        updateState(data.state);
    } catch (e) {
        console.error('Failed to step:', e);
    }
}

async function runSearch() {
    if (!sessionId) return;
    
    const sims = parseInt(document.getElementById('search-sims').value);
    const interval = parseInt(document.getElementById('search-interval').value);
    
    document.getElementById('btn-search').disabled = true;
    const progressBar = document.getElementById('search-progress');
    progressBar.style.display = 'block';
    
    try {
        const res = await fetch(`/api/session/${sessionId}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ simulations: sims, broadcast_interval: interval }),
        });
        const data = await res.json();
        console.log('Search result:', data);
    } catch (e) {
        console.error('Failed to run search:', e);
    } finally {
        document.getElementById('btn-search').disabled = false;
        progressBar.style.display = 'none';
    }
}

function enableControls(enabled) {
    document.getElementById('btn-reset').disabled = !enabled;
    document.getElementById('btn-search').disabled = !enabled;
    updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
    if (currentState) {
        document.getElementById('btn-undo').disabled = !currentState.can_undo;
        document.getElementById('btn-redo').disabled = !currentState.can_redo;
    }
}

// ============================================================
// WebSocket
// ============================================================

function connectWebSocket() {
    if (socket) {
        socket.close();
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`);
    
    socket.onopen = () => {
        document.getElementById('connection-status').className = 'status-connected';
    };
    
    socket.onclose = () => {
        document.getElementById('connection-status').className = 'status-disconnected';
    };
    
    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === 'state') {
            updateState(msg.state);
        } else if (msg.type === 'search_progress') {
            updateSearchProgress(msg.progress);
        }
    };
    
    socket.onerror = (e) => {
        console.error('WebSocket error:', e);
    };
}

// ============================================================
// State Updates
// ============================================================

function updateState(state) {
    currentState = state;
    resizeCanvas();
    render();
    updateInfoPanel();
    updateCandidateList();
    updateUndoRedoButtons();
}

function updateSearchProgress(progress) {
    // Update progress bar
    const percent = Math.round((progress.simulation / progress.total) * 100);
    const progressBar = document.getElementById('search-progress');
    progressBar.querySelector('.progress-fill').style.width = `${percent}%`;
    progressBar.querySelector('.progress-text').textContent = `${percent}% (${progress.simulation}/${progress.total})`;
    
    // Update candidates with visits/values
    if (currentState && progress.candidates) {
        currentState.candidates = progress.candidates;
        render();
        updateCandidateList();
    }
}

function updateInfoPanel() {
    if (!currentState) return;
    
    document.getElementById('info-step').textContent = currentState.step;
    document.getElementById('info-cost').textContent = currentState.cost.toFixed(2);
    document.getElementById('info-value').textContent = currentState.value.toFixed(4);
    document.getElementById('info-remaining').textContent = currentState.remaining.length;
    document.getElementById('info-current').textContent = currentState.current_gid || '-';
}

function updateCandidateList() {
    if (!currentState) return;
    
    const list = document.getElementById('candidate-list');
    const countSpan = document.getElementById('candidate-count');
    
    const validCount = currentState.candidates.filter(c => c.valid).length;
    countSpan.textContent = `(${validCount}/${currentState.candidates.length})`;
    
    list.innerHTML = '';
    
    // Sort by score descending
    const sorted = [...currentState.candidates].sort((a, b) => b.score - a.score);
    
    sorted.forEach(cand => {
        const item = document.createElement('div');
        item.className = 'candidate-item' + (cand.valid ? '' : ' invalid');
        if (selectedCandidate === cand.index) item.classList.add('selected');
        
        const scoreClass = cand.score > 0.5 ? 'score-high' : cand.score > 0.2 ? 'score-mid' : 'score-low';
        
        item.innerHTML = `
            <span class="candidate-index">#${cand.index}</span>
            <span class="candidate-pos">(${cand.x.toFixed(0)}, ${cand.y.toFixed(0)})</span>
            <span class="candidate-score ${scoreClass}">${cand.score.toFixed(3)}</span>
            ${settings.showVisits ? `<span class="candidate-visits">${cand.visits}</span>` : ''}
        `;
        
        item.addEventListener('click', () => {
            if (cand.valid) {
                stepAction(cand.index);
            }
        });
        
        item.addEventListener('mouseenter', () => {
            hoveredCandidate = cand.index;
            render();
        });
        
        item.addEventListener('mouseleave', () => {
            hoveredCandidate = null;
            render();
        });
        
        list.appendChild(item);
    });
}

// ============================================================
// Canvas Rendering
// ============================================================

function render() {
    if (!ctx) return;
    
    // Clear
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (!currentState) {
        // Draw placeholder
        ctx.fillStyle = '#666';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Create a session to start', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    const scaleX = canvas.width / currentState.grid_width;
    const scaleY = canvas.height / currentState.grid_height;
    
    // Draw grid
    if (settings.showGrid) {
        drawGrid(scaleX, scaleY);
    }
    
    // Draw zones (back to front)
    if (settings.showHeightZones) drawZones(scaleX, scaleY, currentState.height_zones, COLORS.heightZone, '#7f7f7f', 'h≤');
    if (settings.showDryZones) drawZones(scaleX, scaleY, currentState.dry_zones, COLORS.dryZone, '#2ca02c', 'dry≥');
    if (settings.showWeightZones) drawZones(scaleX, scaleY, currentState.weight_zones, COLORS.weightZone, '#1f77b4', 'w≤');
    if (settings.showPlacementZones) drawZones(scaleX, scaleY, currentState.placement_zones, COLORS.placementZone, '#1e90ff', '');
    if (settings.showForbidden) drawZones(scaleX, scaleY, currentState.forbidden_areas, COLORS.forbidden, '#d62728', '');
    
    // Draw candidates (before placed so they appear behind)
    if (settings.showCandidates) {
        drawCandidates(scaleX, scaleY);
    }
    
    // Draw placed facilities
    drawPlaced(scaleX, scaleY);
    
    // Draw flow arrows (on top)
    if (settings.showFlow) {
        drawFlow(scaleX, scaleY);
    }
}

function drawGrid(scaleX, scaleY) {
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    
    // Draw every 50 grid units
    const step = 50;
    
    for (let x = 0; x <= currentState.grid_width; x += step) {
        ctx.beginPath();
        ctx.moveTo(x * scaleX, 0);
        ctx.lineTo(x * scaleX, canvas.height);
        ctx.stroke();
    }
    
    for (let y = 0; y <= currentState.grid_height; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, canvas.height - y * scaleY);
        ctx.lineTo(canvas.width, canvas.height - y * scaleY);
        ctx.stroke();
    }
}

function drawZones(scaleX, scaleY, zones, fillColor, strokeColor, prefix) {
    if (!zones || zones.length === 0) return;
    
    zones.forEach(zone => {
        const x = zone.x0 * scaleX;
        const y = canvas.height - zone.y1 * scaleY;
        const w = (zone.x1 - zone.x0) * scaleX;
        const h = (zone.y1 - zone.y0) * scaleY;
        
        // Fill
        ctx.fillStyle = fillColor;
        ctx.fillRect(x, y, w, h);
        
        // Stroke
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, w, h);
        
        // Label
        if (settings.showLabels && (zone.value !== null || zone.id !== null)) {
            let label = '';
            if (zone.id) label = zone.id;
            else if (zone.value !== null) label = prefix + zone.value;
            
            if (label) {
                ctx.fillStyle = strokeColor;
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, x + w / 2, y + h / 2);
            }
        }
    });
}

function drawFlow(scaleX, scaleY) {
    if (!currentState.flow_edges) return;
    
    ctx.strokeStyle = COLORS.flow;
    ctx.lineWidth = 1.5;
    
    currentState.flow_edges.forEach(edge => {
        // Only draw if both endpoints are placed
        if (edge.src_x === null || edge.dst_x === null) return;
        
        const sx = edge.src_x * scaleX;
        const sy = canvas.height - edge.src_y * scaleY;
        const dx = edge.dst_x * scaleX;
        const dy = canvas.height - edge.dst_y * scaleY;
        
        // Draw line
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(dx, dy);
        ctx.stroke();
        
        // Draw arrowhead
        const angle = Math.atan2(dy - sy, dx - sx);
        const arrowLen = 10;
        ctx.beginPath();
        ctx.moveTo(dx, dy);
        ctx.lineTo(
            dx - arrowLen * Math.cos(angle - Math.PI / 6),
            dy - arrowLen * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(dx, dy);
        ctx.lineTo(
            dx - arrowLen * Math.cos(angle + Math.PI / 6),
            dy - arrowLen * Math.sin(angle + Math.PI / 6)
        );
        ctx.stroke();
    });
}

function drawPlaced(scaleX, scaleY) {
    currentState.placed.forEach(fac => {
        const x = fac.x * scaleX;
        const y = canvas.height - (fac.y + fac.h) * scaleY;
        const w = fac.w * scaleX;
        const h = fac.h * scaleY;
        
        // Fill
        ctx.fillStyle = COLORS.placed.fill;
        ctx.fillRect(x, y, w, h);
        
        // Stroke
        ctx.strokeStyle = COLORS.placed.stroke;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x, y, w, h);
        
        // Label
        if (settings.showLabels) {
            ctx.fillStyle = COLORS.text;
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            // Truncate long names
            let label = fac.gid;
            if (label.length > 15) {
                label = label.substring(0, 12) + '..';
            }
            ctx.fillText(label, x + w / 2, y + h / 2);
        }
    });
}

function drawCandidates(scaleX, scaleY) {
    const candidates = currentState.candidates;
    if (!candidates || candidates.length === 0) return;
    
    // Find score range for color mapping
    const scores = candidates.map(c => c.score);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    const scoreRange = maxScore - minScore || 1;
    
    candidates.forEach(cand => {
        if (!cand.valid && !settings.showVisits) return;
        
        const cx = cand.x * scaleX;
        const cy = canvas.height - cand.y * scaleY;
        
        // Determine color
        let color;
        if (cand.index === hoveredCandidate) {
            color = COLORS.candidate.hovered;
        } else if (cand.index === selectedCandidate) {
            color = COLORS.candidate.selected;
        } else if (!cand.valid) {
            color = COLORS.candidate.invalid;
        } else {
            // Color by score (green = high, red = low)
            const t = (cand.score - minScore) / scoreRange;
            color = scoreToColor(t);
        }
        
        // Size by visits if showing
        let radius = 5;
        if (settings.showVisits && cand.visits > 0) {
            radius = Math.min(15, 5 + Math.log(cand.visits + 1) * 2);
        }
        
        // Draw point
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        
        // Highlight hovered/selected
        if (cand.index === hoveredCandidate || cand.index === selectedCandidate) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

function scoreToColor(t) {
    // t: 0 (low) -> 1 (high)
    // Red (low) -> Yellow (mid) -> Green (high)
    const r = t < 0.5 ? 255 : Math.round(255 * (1 - (t - 0.5) * 2));
    const g = t < 0.5 ? Math.round(255 * t * 2) : 255;
    const b = 50;
    return `rgb(${r}, ${g}, ${b})`;
}

// ============================================================
// Canvas Interactions
// ============================================================

function onCanvasMouseMove(e) {
    if (!currentState) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const scaleX = canvas.width / currentState.grid_width;
    const scaleY = canvas.height / currentState.grid_height;
    
    // Find nearest candidate
    let nearest = null;
    let minDist = 20; // Max distance to consider
    
    currentState.candidates.forEach(cand => {
        const cx = cand.x * scaleX;
        const cy = canvas.height - cand.y * scaleY;
        const dist = Math.sqrt((mouseX - cx) ** 2 + (mouseY - cy) ** 2);
        
        if (dist < minDist) {
            minDist = dist;
            nearest = cand;
        }
    });
    
    if (nearest !== null && nearest.index !== hoveredCandidate) {
        hoveredCandidate = nearest.index;
        showHoverInfo(e.clientX, e.clientY, nearest);
        render();
    } else if (nearest === null && hoveredCandidate !== null) {
        hoveredCandidate = null;
        hideHoverInfo();
        render();
    }
}

function onCanvasClick(e) {
    if (!currentState || hoveredCandidate === null) return;
    
    const cand = currentState.candidates.find(c => c.index === hoveredCandidate);
    if (cand && cand.valid) {
        stepAction(cand.index);
    }
}

function onCanvasMouseLeave() {
    hoveredCandidate = null;
    hideHoverInfo();
    render();
}

function showHoverInfo(x, y, cand) {
    const info = document.getElementById('hover-info');
    info.style.display = 'block';
    info.style.left = `${x + 15}px`;
    info.style.top = `${y + 15}px`;
    info.innerHTML = `
        <div><strong>#${cand.index}</strong> ${cand.valid ? '✓' : '✗'}</div>
        <div>Pos: (${cand.x.toFixed(1)}, ${cand.y.toFixed(1)})</div>
        <div>Score: ${cand.score.toFixed(4)}</div>
        ${cand.visits > 0 ? `<div>Visits: ${cand.visits}</div>` : ''}
        ${cand.q_value !== 0 ? `<div>Q: ${cand.q_value.toFixed(4)}</div>` : ''}
    `;
}

function hideHoverInfo() {
    document.getElementById('hover-info').style.display = 'none';
}

// ============================================================
// Start
// ============================================================

init();
