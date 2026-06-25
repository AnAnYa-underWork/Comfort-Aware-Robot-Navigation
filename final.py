import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patheffects import withStroke
from scipy import interpolate
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import networkx as nx
import heapq

# --- Reproducibility ---
np.random.seed(7)
random.seed(7)

# ===========================================================
# Moving Obstacles
# ===========================================================

class MovingObstacle:
    def __init__(self, path_points, speed, radius, color, label):
        self.path_points = np.array(path_points, dtype=float)
        self.speed = speed
        self.radius = radius
        self.color = color
        self.label = label
        self.curr_target = 1
        self.position = self.path_points[0].copy()
        self.prev_position = self.position.copy()

    def update(self, dt):
        self.prev_position = self.position.copy()
        target = self.path_points[self.curr_target]
        direction = target - self.position
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            self.curr_target = (self.curr_target + 1) % len(self.path_points)
            return
        direction /= max(dist, 1e-9)
        move = self.speed * dt
        if move >= dist:
            self.position = target.copy()
            self.curr_target = (self.curr_target + 1) % len(self.path_points)
        else:
            self.position += direction * move

    def shape(self):
        return Point(self.position).buffer(self.radius)

# ===========================================================
# Environment (CLASSROOM layout)
# ===========================================================

class Environment:
    def __init__(self):
        self.width, self.height = 10, 10
        self.static_obs = []
        self.static_drawables = []
        self.dynamic_obs = []
        self.create_obstacles()

    def _add_static(self, poly, color, label=None):
        self.static_obs.append(poly)
        self.static_drawables.append((poly, color, label))

    def create_obstacles(self):
        COL_FLOOR = "#f2e6d6"
        COL_BOARD = "#2e7d32"
        COL_DESK  = "#f4b400"
        COL_SEAT  = "#e64a19"

        # Board
        board = Polygon([(1.0, 9.1), (9.0, 9.1), (9.0, 9.7), (1.0, 9.7)])
        self._add_static(board, COL_BOARD, "Board")

        # Desks & Seats
        desk_w, desk_h = 1.25, 0.75
        seat_w, seat_h = 0.9, 0.35
        row_y = [7.0, 5.2, 3.4]
        col_x = [2.4, 5.0, 7.6]

        for y in row_y:
            for x in col_x:
                desk = Polygon([(x - desk_w/2, y - desk_h/2),
                                (x + desk_w/2, y - desk_h/2),
                                (x + desk_w/2, y + desk_h/2),
                                (x - desk_w/2, y + desk_h/2)])
                self._add_static(desk, COL_DESK)
                seat_y = y - (desk_h/2 + 0.25 + seat_h/2)
                seat = Polygon([(x - seat_w/2, seat_y - seat_h/2),
                                (x + seat_w/2, seat_y - seat_h/2),
                                (x + seat_w/2, seat_y + seat_h/2),
                                (x - seat_w/2, seat_y + seat_h/2)])
                self._add_static(seat, COL_SEAT)

        # Dynamic obstacles (teachers)
        aisle_mid_y  = 4.0
        aisle_high_y = 6.6

        self.dynamic_obs = [
            MovingObstacle([(1.0, aisle_mid_y), (9.0, aisle_mid_y)],
                           speed=0.6, radius=0.28, color="#1e88e5", label="Teacher A"),
            MovingObstacle([(9.0, aisle_high_y), (1.0, aisle_high_y)],
                           speed=0.55, radius=0.28, color="#fb8c00", label="Teacher B"),
        ]

        self.COL_FLOOR = COL_FLOOR

    def obstacles_union(self, clearance=0.25):
        static_expanded = [obs.buffer(clearance) for obs in self.static_obs]
        dynamic_expanded = [mob.shape().buffer(clearance) for mob in self.dynamic_obs]
        return unary_union(static_expanded + dynamic_expanded)

    def in_obstacle(self, p):
        return self.obstacles_union().contains(Point(p))


# ===========================================================
# RRT* + Smoothing
# ===========================================================

class Node:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.parent = None
        self.cost = 0.0

def dist(a, b): return np.linalg.norm(a - b)

def steer(from_node, to_pos, step):
    d = to_pos - from_node.pos
    L = np.linalg.norm(d)
    return from_node.pos + step * d / max(L, 1e-9)

def collision_free_basic(p1, p2, env):
    """Baseline: no fuzzy comfort check."""
    line = LineString([p1, p2])
    return not line.intersects(env.obstacles_union(clearance=0.2))

def rrt_star_basic(start, goal, env, step=0.4, radius=0.8, n_iter=1500):
    """Standard RRT* without fuzzy comfort logic."""
    nodes = [Node(start)]
    goal_node = Node(goal)
    for _ in range(n_iter):
        rand = np.array([random.uniform(0, env.width), random.uniform(0, env.height)])
        if env.in_obstacle(rand): continue
        nearest = min(nodes, key=lambda n: dist(n.pos, rand))
        new_pos = steer(nearest, rand, step)
        if env.in_obstacle(new_pos) or not collision_free_basic(nearest.pos, new_pos, env): continue
        new_node = Node(new_pos)
        near_nodes = [n for n in nodes if dist(n.pos, new_pos) < radius and collision_free_basic(n.pos, new_pos, env)]
        if near_nodes:
            costs = [n.cost + dist(n.pos, new_pos) for n in near_nodes]
            best = near_nodes[np.argmin(costs)]
            new_node.parent = best
            new_node.cost = min(costs)
        nodes.append(new_node)
        for n in near_nodes:
            c = new_node.cost + dist(n.pos, new_node.pos)
            if c < n.cost and collision_free_basic(n.pos, new_node.pos, env):
                n.parent = new_node; n.cost = c
        if dist(new_node.pos, goal) < step and collision_free_basic(new_node.pos, goal, env):
            goal_node.parent = new_node; break
    path, node = [], goal_node
    while node:
        path.append(node.pos); node = node.parent
    path.reverse()
    return np.array(path)

def collision_free(p1, p2, env):
    line = LineString([p1, p2])
    if line.intersects(env.obstacles_union(clearance=0.2)):
        return False
    # Comfort gate kept (your original behavior)
    mid = np.array(line.interpolate(0.5, normalized=True).coords[0])
    comfort_here = fuzzy_comfort(mid, env)
    if comfort_here < 0.4:
        return False
    return True

def rrt_star(start, goal, env, step=0.4, radius=0.8, n_iter=1500):
    nodes = [Node(start)]
    goal_node = Node(goal)
    for _ in range(n_iter):
        rand = np.array([random.uniform(0, env.width), random.uniform(0, env.height)])
        if env.in_obstacle(rand): continue
        nearest = min(nodes, key=lambda n: dist(n.pos, rand))
        new_pos = steer(nearest, rand, step)
        if env.in_obstacle(new_pos) or not collision_free(nearest.pos, new_pos, env): continue
        new_node = Node(new_pos)
        near_nodes = [n for n in nodes if dist(n.pos, new_pos) < radius and collision_free(n.pos, new_pos, env)]
        if near_nodes:
            costs = [n.cost + dist(n.pos, new_pos) for n in near_nodes]
            best = near_nodes[np.argmin(costs)]
            new_node.parent = best
            new_node.cost = min(costs)
        nodes.append(new_node)
        for n in near_nodes:
            c = new_node.cost + dist(n.pos, new_node.pos)
            if c < n.cost and collision_free(n.pos, new_node.pos, env):
                n.parent = new_node; n.cost = c
        if dist(new_node.pos, goal) < step and collision_free(new_node.pos, goal, env):
            goal_node.parent = new_node; break
    path, node = [], goal_node
    while node:
        path.append(node.pos); node = node.parent
    path.reverse()
    return np.array(path)

def smooth_path(path):
    if len(path) < 4:
        return path
    t = np.linspace(0, 1, len(path))
    splx = interpolate.make_interp_spline(t, path[:, 0], k=3)
    sply = interpolate.make_interp_spline(t, path[:, 1], k=3)
    t_smooth = np.linspace(0, 1, 300)
    return np.vstack((splx(t_smooth), sply(t_smooth))).T

# ===========================================================
# Fuzzy Comfort (unchanged helper used by collision_free & planner)
# ===========================================================

def setup_fuzzy_system():
    dist_var = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'distance')
    vel_var = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'relative_velocity')
    comfort_var = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'comfort')

    dist_var['close'] = fuzz.trapmf(dist_var.universe, [0, 0, 0.8, 1.5])
    dist_var['medium'] = fuzz.trimf(dist_var.universe, [1.0, 2.5, 4.0])
    dist_var['far'] = fuzz.trapmf(dist_var.universe, [3.0, 4.0, 5.0, 5.0])

    vel_var['approaching'] = fuzz.trapmf(vel_var.universe, [-1, -1, -0.3, 0])
    vel_var['still'] = fuzz.trimf(vel_var.universe, [-0.2, 0, 0.2])
    vel_var['leaving'] = fuzz.trapmf(vel_var.universe, [0, 0.3, 1, 1])

    comfort_var['unsafe'] = fuzz.trapmf(comfort_var.universe, [0, 0, 0.3, 0.5])
    comfort_var['neutral'] = fuzz.trimf(comfort_var.universe, [0.4, 0.6, 0.8])
    comfort_var['comfortable'] = fuzz.trapmf(comfort_var.universe, [0.7, 0.9, 1, 1])

    rules = [
        ctrl.Rule(dist_var['close'] & vel_var['approaching'], comfort_var['unsafe']),
        ctrl.Rule(dist_var['medium'] & vel_var['approaching'], comfort_var['neutral']),
        ctrl.Rule(dist_var['far'] & vel_var['approaching'], comfort_var['neutral']),
        ctrl.Rule(dist_var['close'] & vel_var['still'], comfort_var['neutral']),
        ctrl.Rule(dist_var['medium'] & vel_var['still'], comfort_var['comfortable']),
        ctrl.Rule(dist_var['far'] & vel_var['still'], comfort_var['comfortable']),
        ctrl.Rule(dist_var['close'] & vel_var['leaving'], comfort_var['neutral']),
        ctrl.Rule(dist_var['medium'] & vel_var['leaving'], comfort_var['comfortable']),
        ctrl.Rule(dist_var['far'] & vel_var['leaving'], comfort_var['comfortable'])
    ]
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)

def fuzzy_comfort(pos, env):
    p = Point(pos)
    d_static = min([p.distance(obs) for obs in env.static_obs]) if env.static_obs else 10.0
    d_dynamic = min([np.linalg.norm(mob.position - pos) for mob in env.dynamic_obs]) if env.dynamic_obs else 10.0
    d_min = min(d_static, d_dynamic)
    comfort = np.clip(d_min / 2.0, 0, 1)
    comfort = 0.7 * comfort + 0.3 * np.random.uniform(0.9, 1.0)
    return comfort

# ===========================================================
# Lightweight Roadmap Planner (for quick reroutes)
# ===========================================================

class LazyPlanner:
    def __init__(self, env, n_samples=200):
        self.env = env
        self.nodes = []
        self.edges = {}
        self.build_roadmap(n_samples)

    def build_roadmap(self, n_samples):
        for _ in range(n_samples):
            p = np.random.uniform([0, 0], [self.env.width, self.env.height])
            if not self.env.in_obstacle(p):
                self.nodes.append(tuple(p))
        for i, a in enumerate(self.nodes):
            for j, b in enumerate(self.nodes):
                if i < j and np.linalg.norm(np.array(a) - np.array(b)) < 1.5:
                    if collision_free(np.array(a), np.array(b), self.env):
                        self.edges.setdefault(a, []).append(b)
                        self.edges.setdefault(b, []).append(a)

    def shortest_path(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        if self.env.in_obstacle(start) or self.env.in_obstacle(goal):
            return []
        # attach start/goal to nearest nodes
        if start not in self.nodes:
            self.nodes.append(start)
        if goal not in self.nodes:
            self.nodes.append(goal)

        for n in sorted(self.nodes, key=lambda p: np.linalg.norm(np.array(p)-np.array(start)))[:8]:
            if collision_free(np.array(n), np.array(start), self.env):
                self.edges.setdefault(start, []).append(n)
                self.edges.setdefault(n, []).append(start)
        for n in sorted(self.nodes, key=lambda p: np.linalg.norm(np.array(p)-np.array(goal)))[:8]:
            if collision_free(np.array(n), np.array(goal), self.env):
                self.edges.setdefault(goal, []).append(n)
                self.edges.setdefault(n, []).append(goal)

        pq = [(0, start, [])]
        visited = set()
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited: continue
            visited.add(node)
            path = path + [node]
            if node == goal:
                return np.array(path)
            for nxt in self.edges.get(node, []):
                if nxt not in visited:
                    dist_cost = np.linalg.norm(np.array(node) - np.array(nxt))
                    # retain fuzzy penalty as in your original
                    fuzzy_penalty = 1.2 - 0.5 * fuzzy_comfort(np.array(nxt), self.env)
                    heapq.heappush(pq, (cost + dist_cost * fuzzy_penalty, nxt, path))
        return []

# ===========================================================
# Visualization & Animation
# ===========================================================
import time

def compute_path_length(path):
    if len(path) < 2: return 0
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def compute_smoothness(path):
    if len(path) < 3: return 0
    vecs = np.diff(path, axis=0)
    angles = []
    for i in range(1, len(vecs)):
        dot = np.dot(vecs[i-1], vecs[i])
        denom = np.linalg.norm(vecs[i-1]) * np.linalg.norm(vecs[i])
        if denom > 0:
            angles.append(np.arccos(np.clip(dot / denom, -1, 1)))
    return 1 - np.mean(np.abs(np.diff(angles))) / np.pi if len(angles) > 1 else 0.0

from scipy.spatial.distance import directed_hausdorff

def compare_paths(old_path, new_path, comfort_old, comfort_new):
    """Compare rerouted path vs previous planned path."""
    if len(old_path) < 2 or len(new_path) < 2:
        return {}

    old_len = compute_path_length(old_path)
    new_len = compute_path_length(new_path)
    old_smooth = compute_smoothness(old_path)
    new_smooth = compute_smoothness(new_path)
    old_comf = np.mean(comfort_old) if len(comfort_old) else 0
    new_comf = np.mean(comfort_new) if len(comfort_new) else 0

    # Compute spatial deviation (Hausdorff distance)
    dev = max(
        directed_hausdorff(old_path, new_path)[0],
        directed_hausdorff(new_path, old_path)[0]
    )

    # Relative differences
    delta_len = new_len - old_len
    delta_comf = new_comf - old_comf
    delta_smooth = new_smooth - old_smooth

    # A composite "replan gain" score (weighted)
    replan_gain = (0.6 * delta_comf + 0.3 * delta_smooth - 0.1 * (delta_len / max(old_len, 1)))

    return {
        "old_len": old_len, "new_len": new_len,
        "delta_len": delta_len,
        "old_comf": old_comf, "new_comf": new_comf,
        "delta_comf": delta_comf,
        "old_smooth": old_smooth, "new_smooth": new_smooth,
        "delta_smooth": delta_smooth,
        "deviation": dev,
        "replan_gain": replan_gain
    }
# ---------- Deterministic comfort for metrics (no randomness) ----------
def fuzzy_comfort_det(pos, env):
    p = Point(pos)
    d_static = min([p.distance(obs) for obs in env.static_obs]) if env.static_obs else 10.0
    d_dynamic = min([np.linalg.norm(mob.position - pos) for mob in env.dynamic_obs]) if env.dynamic_obs else 10.0
    d_min = min(d_static, d_dynamic)
    return float(np.clip(d_min / 2.0, 0, 1))  # 0..1

# ---------- Robust metrics helpers ----------
def compute_path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def compute_smoothness(path):
    # 1.0 is very smooth, lower means jerkier turns
    if path is None or len(path) < 3:
        return 1.0
    v1 = path[1:-1] - path[:-2]
    v2 = path[2:] - path[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-9
    n2 = np.linalg.norm(v2, axis=1) + 1e-9
    cosang = np.sum(v1*v2, axis=1)/(n1*n2)
    cosang = np.clip(cosang, -1, 1)
    return float(np.mean((cosang+1)/2))  # 0..1, higher=straighter

def resample_polyline(path, n=200):
    if path is None or len(path) < 2:
        return path
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    L = np.sum(seg)
    if L < 1e-9:
        return np.repeat(path[:1], n, axis=0)
    t = np.concatenate([[0], np.cumsum(seg)/L])
    t_new = np.linspace(0, 1, n)
    x = np.interp(t_new, t, path[:,0])
    y = np.interp(t_new, t, path[:,1])
    return np.column_stack([x,y])

def comfort_along(path, env):
    if path is None or len(path) == 0:
        return 0.0
    vals = [fuzzy_comfort_det(p, env) for p in path]
    return float(np.mean(vals))

# Symmetric Hausdorff on resampled paths (same density)
from scipy.spatial.distance import directed_hausdorff
def path_deviation(a, b):
    if a is None or b is None or len(a)<2 or len(b)<2:
        return 0.0
    A = resample_polyline(a, 200); B = resample_polyline(b, 200)
    d1 = directed_hausdorff(A, B)[0]
    d2 = directed_hausdorff(B, A)[0]
    return float(max(d1, d2))

# Compare remaining baseline vs new reroute (both from current->goal)
def compare_paths(baseline_suffix, new_path, env):
    bl = resample_polyline(baseline_suffix, 200)
    nw = resample_polyline(new_path, 200)
    old_len = compute_path_length(bl)
    new_len = compute_path_length(nw)
    old_smooth = compute_smoothness(bl)
    new_smooth = compute_smoothness(nw)
    old_comf = comfort_along(bl, env)
    new_comf = comfort_along(nw, env)
    dev = path_deviation(bl, nw)
    delta_len = new_len - old_len
    delta_comf = new_comf - old_comf
    delta_smooth = new_smooth - old_smooth
    # Prioritize comfort (you can tune weights)
    replan_gain = 0.75*delta_comf + 0.2*delta_smooth - 0.05*(delta_len/max(old_len,1e-6))
    return {
        "old_len": old_len, "new_len": new_len, "delta_len": delta_len,
        "old_comf": old_comf, "new_comf": new_comf, "delta_comf": delta_comf,
        "old_smooth": old_smooth, "new_smooth": new_smooth, "delta_smooth": delta_smooth,
        "deviation": dev, "replan_gain": replan_gain
    }


def main():
    env = Environment()
    fuzzy_sys = setup_fuzzy_system()  # kept for completeness (used indirectly)
    planner = LazyPlanner(env)

    GOAL_RADIUS = 0.20           # ← added: stop condition
    start, goal = np.array([5.0, 0.8]), np.array([5.0, 8.8])
    dt = 0.1

    # Global path
    path_rrt = rrt_star(start, goal, env)
    path_comfort = smooth_path(path_rrt)

    # ← ensure last point equals goal exactly (prevents “stops early”)
    if len(path_comfort) > 0:
        path_comfort[-1] = goal

    previous_paths = []

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    fig.patch.set_facecolor(env.COL_FLOOR)
    ax.set_facecolor("#eee7dc")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Fuzzy Comfort-Aware RRT* Path Planning (Classroom)", fontsize=13, weight='bold')

    for poly, color, label in env.static_drawables:
        coords = np.array(poly.exterior.coords)
        ax.add_patch(MplPolygon(coords, closed=True, facecolor=color, edgecolor="#5a5245", lw=1.0))
        if label:
            cx, cy = np.mean(coords[:-1,0]), np.mean(coords[:-1,1])
            ax.text(cx, cy+0.25, label, ha='center', va='bottom', fontsize=9, weight='bold')

    rrt_line, = ax.plot(path_rrt[:, 0], path_rrt[:, 1], color='#264653', lw=2.4, label='RRT* Path')
    comfort_line, = ax.plot(path_comfort[:, 0], path_comfort[:, 1], color='#E63946', lw=3.0, alpha=0.95, label='Comfort Path')

    robot_dot, = ax.plot([], [], 'o', color='#E63946', markersize=8, zorder=5)
    mob_dots = []
    for mob in env.dynamic_obs:
        dot, = ax.plot([], [], 'o', ms=15, color=mob.color, alpha=0.95)
        mob_dots.append(dot)
        ax.text(mob.path_points[0][0], mob.path_points[0][1] + 0.35, mob.label, ha='center', va='bottom', fontsize=9)

    ax.legend(frameon=False, loc='lower right', fontsize=10)
    replan_flash = [0]

    current_path = [path_comfort.copy()]
    path_index = [0]
    reroute_points = []
    replan_cooldown = [0]
    reroute_preview, = ax.plot([], [], '--', color='#0077b6', lw=2.0, alpha=0.7, zorder=1.5)

    reached_goal = [False]
    comfort_values = []
    min_clearances = []
    replan_count = [0]
    replan_times = []
    path_lengths = []
    smoothness_scores = []
    # --- Baseline RRT* (no fuzzy comfort) ---
    print("\nRunning baseline RRT* ...")
    path_rrt_base = rrt_star_basic(start, goal, env)
    path_base_smooth = smooth_path(path_rrt_base)
    base_length = compute_path_length(path_base_smooth)
    base_smoothness = compute_smoothness(path_base_smooth)

    # --- Proposed Fuzzy Comfort RRT* ---
    print("Running fuzzy comfort-aware RRT* ...")
    path_rrt = rrt_star(start, goal, env)
    path_comfort = smooth_path(path_rrt)
    proposed_length = compute_path_length(path_comfort)
    proposed_smoothness = compute_smoothness(path_comfort)
    segment_comfort = []  # comfort values for current segment
    last_path = None  # keeps track of previous path for comparison
    reroute_metrics = []

    def update(frame):
        nonlocal path_comfort, segment_comfort, last_path

        dt_local = dt

        # Move teachers
        for mob in env.dynamic_obs:
            mob.update(dt_local)
        for i, mob in enumerate(env.dynamic_obs):
            mob_dots[i].set_data([mob.position[0]], [mob.position[1]])

        if replan_cooldown[0] > 0:
            replan_cooldown[0] -= 1

        # Follow current path
        active_path = current_path[-1]
        i = path_index[0]

        # ← don’t exit when path ends; hold at last point
        if i >= len(active_path):
            robot_pos = active_path[-1]
        else:
            robot_pos = active_path[i]
            path_index[0] += 1

        # Goal snap/stop
        if np.linalg.norm(robot_pos - goal) <= GOAL_RADIUS:
            robot_pos = goal.copy()
            reached_goal[0] = True

        # Check proximity to dynamic obstacles; quick reroute preview
        if (not reached_goal[0]) and replan_cooldown[0] == 0:
            trigger = any(np.linalg.norm(mob.position - robot_pos) < 0.8 for mob in env.dynamic_obs)
            if trigger:
                new_path = planner.shortest_path(robot_pos, goal)
                if len(new_path) > 4:
                    new_smooth = smooth_path(new_path)
                    new_smooth[-1] = goal.copy()

                    # Show reroute preview visually
                    reroute_preview.set_data(new_smooth[:, 0], new_smooth[:, 1])
                    current_path.append(new_smooth)
                    path_index[0] = 0
                    replan_cooldown[0] = 12

                    # --- Determine comparison baseline ---
                    if last_path is None:
                        baseline_path = path_comfort  # first reroute: compare with original global path
                    else:
                        baseline_path = last_path  # later reroutes: compare with previous reroute

                    # robot_pos already computed, and you've built: new_path -> new_smooth (ending at goal)

                    # Determine the proper baseline: the remaining suffix of the current path from the robot's index
                    active_path = current_path[-1]
                    i = path_index[0] if path_index[0] < len(active_path) else len(active_path) - 1
                    baseline_suffix = active_path[i:]  # compare fair "remaining to goal" vs new reroute to goal

                    # If this is the very first replan and you want to compare against the original RRT* smooth:
                    # (baseline_suffix is already the suffix of that original path at first replan)
                    baseline_name = "Original Path" if last_path is None else "Previous Reroute"

                    cmpres = compare_paths(baseline_suffix, new_smooth, env)
                    reroute_metrics.append({"baseline": baseline_name, **cmpres})

                    print("\n🔁 REROUTE COMPARISON")
                    print(f"Baseline: {baseline_name}")
                    print(f"ΔPath Length:  {cmpres['delta_len']:.3f}")
                    print(f"ΔComfort:      {cmpres['delta_comf']:.3f}")
                    print(f"ΔSmoothness:   {cmpres['delta_smooth']:.3f}")
                    print(f"Deviation:     {cmpres['deviation']:.3f}")
                    print(f"Replan Gain:   {cmpres['replan_gain']:.3f}")

                    last_path = new_smooth.copy()
                    segment_comfort = []  # reset collector


                else:
                    reroute_preview.set_data([], [])

        # Draw all paths (history + current)
        for p in current_path[:-1]:
            ax.plot(p[:, 0], p[:, 1], color='gray', lw=1.0, alpha=0.35, zorder=1)
        ax.plot(current_path[-1][:, 0], current_path[-1][:, 1],
                color='#E63946', lw=2.8, alpha=0.9, zorder=2)

        robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
        # Record comfort and safety metrics
        comfort_score = fuzzy_comfort(robot_pos, env)
        segment_comfort.append(comfort_score)

        comfort_values.append(comfort_score)
        if "segment_comfort" not in locals():
            segment_comfort = []
        segment_comfort.append(comfort_score)

        dist_to_dyn = min(np.linalg.norm(mob.position - robot_pos) for mob in env.dynamic_obs)
        min_clearances.append(dist_to_dyn)

        return [robot_dot, *mob_dots, reroute_preview]

    # ← give extra frames so it can reach and sit at the goal
    total_frames = len(path_comfort) + 200
    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=80, blit=True, repeat=False)

    plt.tight_layout(); plt.show()
    # --- Post-Run Analysis ---
    if comfort_values:
        avg_comfort = np.mean(comfort_values)
        min_clearance = np.mean(min_clearances)
        avg_replan_time = np.mean(replan_times) if replan_times else 0
        avg_path_len = np.mean(path_lengths) if path_lengths else compute_path_length(path_comfort)
        avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else compute_smoothness(path_comfort)

        print("\n===== BASELINE vs PROPOSED COMPARISON =====")
        print(f"Path Length (RRT*):              {base_length:.3f}")
        print(f"Path Length (Fuzzy-RRT*):        {proposed_length:.3f}")
        print(f"Smoothness (RRT*):               {base_smoothness:.3f}")
        print(f"Smoothness (Fuzzy-RRT*):         {proposed_smoothness:.3f}")
        print(f"Average Comfort (Fuzzy-RRT*):    {np.mean(comfort_values):.3f}")
        print(f"Replans Triggered (Fuzzy-RRT*):  {replan_count[0]}")
        print("===========================================")

        # --- Visualization Comparison ---
        plt.figure(figsize=(6, 6))
        plt.plot(path_base_smooth[:, 0], path_base_smooth[:, 1], '--', color='gray', label='Baseline RRT*')
        plt.plot(path_comfort[:, 0], path_comfort[:, 1], '-', color='red', label='Fuzzy Comfort-RRT*')
        plt.scatter(*start, c='green', label='Start', s=60)
        plt.scatter(*goal, c='blue', label='Goal', s=60)
        plt.legend();
        plt.title("Baseline vs Proposed Path Comparison")
        plt.xlabel("X");
        plt.ylabel("Y")
        plt.show()

        print("\n===== PERFORMANCE METRICS =====")
        print(f"Average Comfort Score:     {avg_comfort:.3f}")
        print(f"Average Clearance:         {min_clearance:.3f} m")
        print(f"Replans Triggered:         {replan_count[0]}")
        print(f"Average Replan Time:       {avg_replan_time:.3f} s")
        print(f"Average Path Length:       {avg_path_len:.3f} units")
        print(f"Average Smoothness Score:  {avg_smoothness:.3f}")
        print("===============================")

        # --- Plot Performance Graphs ---
        plt.figure(figsize=(8, 5))
        plt.subplot(2, 1, 1)
        plt.plot(comfort_values, label='Comfort Score', color='#E63946')
        plt.ylabel("Comfort");
        plt.legend();
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(min_clearances, label='Min Clearance', color='#219EBC')
        plt.ylabel("Distance to Dynamic Obstacle")
        plt.xlabel("Frame");
        plt.legend();
        plt.grid(True, alpha=0.3)

        plt.tight_layout();
        plt.show()

        # --- Comfort Heatmap Visualization ---
        grid_res = 100
        xv, yv = np.meshgrid(np.linspace(0, 10, grid_res), np.linspace(0, 10, grid_res))
        comfort_grid = np.zeros_like(xv)

        for i in range(grid_res):
            for j in range(grid_res):
                comfort_grid[j, i] = fuzzy_comfort([xv[j, i], yv[j, i]], env)

        print(comfort_grid)
        plt.figure(figsize=(6, 6))
        plt.imshow(comfort_grid, extent=(0, 10, 0, 10), origin='lower', cmap='coolwarm', alpha=0.85)
        plt.colorbar(label="Comfort Level")
        plt.title("Fuzzy Comfort Field of Classroom Environment")
        plt.show()

        if reroute_metrics:
            print("\n===== REROUTE SUMMARY =====")
            for k, m in enumerate(reroute_metrics, 1):
                print(f"#{k:02d} [{m['baseline']}] "
                      f"ΔLen={m['delta_len']:.3f}, ΔComf={m['delta_comf']:.3f}, "
                      f"ΔSmooth={m['delta_smooth']:.3f}, Dev={m['deviation']:.3f}, "
                      f"Gain={m['replan_gain']:.3f}")


if __name__ == "__main__":
    main()
