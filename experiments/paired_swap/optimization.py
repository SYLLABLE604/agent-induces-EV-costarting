"""Exact small-component signed-cut search and unit-power minimum-cost flow."""
from __future__ import annotations

from collections import deque
from itertools import product

from protocol import bin_id, minima


def coordinate(ids, weights, limit=22):
    adjacency = {i: set() for i in ids}
    for (i, j), w in weights.items():
        if w:
            adjacency[i].add(j)
            adjacency[j].add(i)
    unseen, result, components = set(ids), {}, []
    while unseen:
        root = min(unseen)
        nodes, stack = set(), [root]
        while stack:
            i = stack.pop()
            if i not in nodes:
                nodes.add(i)
                stack.extend(adjacency[i] - nodes)
        unseen -= nodes
        nodes = sorted(nodes)
        if len(nodes) > limit:
            raise ValueError(f"Component size {len(nodes)} > exact limit {limit}; no heuristic advertised as exact")
        edges = [(i, j, w) for (i, j), w in weights.items() if i in nodes and j in nodes and w]
        best_value, best = None, None
        for tail in product((-1, 1), repeat=len(nodes) - 1):
            z = dict(zip(nodes, (1,) + tail))
            value = sum(w * z[i] * z[j] for i, j, w in edges)
            if best_value is None or value < best_value:
                best_value, best = value, z
        result.update(best)
        components.append({"tasks": nodes, "objective": float(best_value), "exact": True})
    return result, components


def repair(tasks, starts, capacity):
    """A_i = frozen M_i, not rebuilt relative to a selected output; integer capacities."""
    graph = []

    def node():
        graph.append([])
        return len(graph) - 1

    def edge(u, v, cap, cost):
        a = [v, cap, cost, len(graph[v])]
        b = [u, 0, -cost, len(graph[u])]
        graph[u].append(a)
        graph[v].append(b)
        return a

    source = node()
    task_nodes = {t["task_id"]: node() for t in tasks}
    slots = sorted({bin_id(o["start_time_utc"]) for t in tasks for o in minima(t)})
    slot_nodes = {k: node() for k in slots}
    sink = node()
    choices = {}
    for t in tasks:
        tid = t["task_id"]
        edge(source, task_nodes[tid], 1, 0)
        per_bin = {}
        for o in minima(t):
            s = o["start_time_utc"]
            k = bin_id(s)
            if k not in per_bin or s == starts.get(tid):
                per_bin[k] = s
        choices[tid] = [(s, edge(task_nodes[tid], slot_nodes[k], 1, int(s != starts.get(tid))))
                        for k, s in per_bin.items()]
    for k in slots:
        edge(slot_nodes[k], sink, capacity, 0)
    flow = cost = 0
    while flow < len(tasks):
        distance = [float("inf")] * len(graph)
        parent = [None] * len(graph)
        distance[source] = 0
        queue, queued = deque([source]), {source}
        while queue:
            u = queue.popleft()
            queued.remove(u)
            for index, (v, cap, price, _) in enumerate(graph[u]):
                if cap and distance[v] > distance[u] + price:
                    distance[v], parent[v] = distance[u] + price, (u, index)
                    if v not in queued:
                        queued.add(v)
                        queue.append(v)
        if parent[sink] is None:
            y = [tid for tid, u in task_nodes.items() if distance[u] < float("inf")]
            neighbors = [k for k, v in slot_nodes.items() if distance[v] < float("inf")]
            return {"feasible": False, "certificate_tasks": y, "neighbor_bins": neighbors,
                    "demand": len(y), "capacity": capacity * len(neighbors)}
        v = sink
        while v != source:
            u, index = parent[v]
            arc = graph[u][index]
            arc[1] -= 1
            graph[v][arc[3]][1] += 1
            v = u
        flow += 1
        cost += int(distance[sink])
    final = {tid: next(s for s, e in arcs if e[1] == 0) for tid, arcs in choices.items()}
    bad = sum(starts.get(t["task_id"]) not in {o["start_time_utc"] for o in minima(t)} for t in tasks)
    return {"feasible": True, "modified": cost, "mandatory_quality_changes": bad,
            "additional_capacity_changes": cost - bad, "starts": final}
