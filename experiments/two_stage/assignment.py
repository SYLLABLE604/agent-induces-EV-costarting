"""Exact budget feasibility and lexicographic minimum-change assignment."""
from __future__ import annotations

from collections import defaultdict, deque


def allowed_bins(tasks, budget):
    return {t.id: sorted({c.bin for c in t.candidates if c.regret <= budget}) for t in tasks}


def feasible(tasks, capacity, budget):
    if capacity < 1 or budget < 0:
        raise ValueError("Capacity must be positive and budget nonnegative")
    adj = allowed_bins(tasks, budget)
    loads, owner = defaultdict(list), {}

    def augment(task, seen):
        for k in adj[task]:
            if k in seen:
                continue
            seen.add(k)
            if len(loads[k]) < capacity:
                loads[k].append(task)
                owner[task] = k
                return True
            for other in list(loads[k]):
                if augment(other, seen):
                    loads[k].remove(other)
                    loads[k].append(task)
                    owner[task] = k
                    return True
        return False

    for task in sorted(adj):
        augment(task, set())
    if len(owner) == len(tasks):
        starts = {}
        for t in tasks:
            choices = [c for c in t.candidates if c.bin == owner[t.id] and c.regret <= budget]
            starts[t.id] = min(choices, key=lambda c: (c.regret, c.start)).start
        return {"feasible": True, "starts": starts}

    reached = set(adj) - set(owner)
    queue, neighbors = list(reached), set()
    for task in queue:
        for k in adj[task]:
            if k in neighbors:
                continue
            neighbors.add(k)
            for other in loads[k]:
                if other not in reached:
                    reached.add(other)
                    queue.append(other)
    actual_neighbors = {k for task in reached for k in adj[task]}
    assert actual_neighbors == neighbors
    assert len(reached) > capacity * len(neighbors)
    return {"feasible": False, "certificate": {
        "task_ids": sorted(reached), "bins": sorted(neighbors),
        "demand": len(reached), "capacity": capacity * len(neighbors),
        "budget_micro": budget, "per_bin_capacity": capacity,
    }}


def minimum_budget(tasks, capacity):
    thresholds = sorted({0} | {c.regret for t in tasks for c in t.candidates})
    all_options = feasible(tasks, capacity, thresholds[-1])
    if not all_options["feasible"]:
        return {"capacity": capacity, "feasible": False, "full_set_certificate": all_options["certificate"]}
    lo, hi = 0, len(thresholds) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(tasks, capacity, thresholds[mid])["feasible"]:
            hi = mid
        else:
            lo = mid + 1
    result = {"capacity": capacity, "feasible": True, "budget_micro": thresholds[lo],
              "witness_starts": feasible(tasks, capacity, thresholds[lo])["starts"]}
    if lo:
        previous = feasible(tasks, capacity, thresholds[lo - 1])
        assert not previous["feasible"]
        result["previous_threshold_micro"] = thresholds[lo - 1]
        result["lower_certificate"] = previous["certificate"]
    return result


def minimum_peak(tasks, budget):
    lo, hi = 1, len(tasks)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(tasks, mid, budget)["feasible"]:
            hi = mid
        else:
            lo = mid + 1
    return lo


class PlanSolver:
    """Primary: changed tasks. Secondary: total additional tariff cost.

    Budget and capacity are hard constraints. Full feasible-set support is fixed
    before looking at the original plan. Components share no admissible bins.
    """

    def __init__(self, tasks, capacity, budget):
        self.tasks = {t.id: t for t in tasks}
        self.capacity, self.budget = capacity, budget
        self.allowed = {t.id: tuple(c for c in t.candidates if c.regret <= budget) for t in tasks}
        self.validation = feasible(tasks, capacity, budget)
        self.cache = {}
        self.multiplier = len(tasks) * budget + 1
        by_bin = defaultdict(set)
        for i, choices in self.allowed.items():
            for c in choices:
                by_bin[c.bin].add(i)
        unseen, components = set(self.tasks), []
        while unseen:
            first = min(unseen)
            stack, group = [first], {first}
            for i in stack:
                for c in self.allowed[i]:
                    for j in by_bin[c.bin]:
                        if j not in group:
                            group.add(j)
                            stack.append(j)
            unseen -= group
            components.append(tuple(sorted(group)))
        self.components = components

    def solve(self, starts):
        if not self.validation["feasible"]:
            return self.validation
        final = {}
        for group in self.components:
            key = (group, tuple(starts.get(i) for i in group))
            if key not in self.cache:
                self.cache[key] = self._component(group, starts)
            final.update(self.cache[key])
        modified = sum(final[i] != starts.get(i) for i in self.tasks)
        mandatory = sum(starts.get(i) not in {c.start for c in cs} for i, cs in self.allowed.items())
        regrets = [self.tasks[i].by_start[s].regret for i, s in final.items()]
        counts = defaultdict(int)
        for i, s in final.items():
            counts[self.tasks[i].by_start[s].bin] += 1
        assert len(final) == len(self.tasks)
        assert max(counts.values(), default=0) <= self.capacity
        assert max(regrets, default=0) <= self.budget
        assert modified >= mandatory
        return {"feasible": True, "starts": final, "modified": modified,
                "mandatory_admissibility_changes": mandatory,
                "additional_capacity_changes": modified - mandatory,
                "max_regret_micro": max(regrets, default=0), "total_regret_micro": sum(regrets),
                "peak": max(counts.values(), default=0)}

    def _component(self, group, starts):
        graph = []

        def node():
            graph.append([])
            return len(graph) - 1

        def edge(u, v, cap, cost):
            a, b = [v, cap, cost, len(graph[v])], [u, 0, -cost, len(graph[u])]
            graph[u].append(a)
            graph[v].append(b)
            return a

        source = node()
        task_nodes = {i: node() for i in group}
        bins = sorted({c.bin for i in group for c in self.allowed[i]})
        slot_nodes = {k: node() for k in bins}
        sink = node()
        selected = {}
        for i in group:
            edge(source, task_nodes[i], 1, 0)
            per_bin = {}
            for c in self.allowed[i]:
                objective = self.multiplier * (c.start != starts.get(i)) + c.regret
                value = (objective, c.start)
                if c.bin not in per_bin or value < per_bin[c.bin]:
                    per_bin[c.bin] = value
            selected[i] = [(s, edge(task_nodes[i], slot_nodes[k], 1, obj))
                           for k, (obj, s) in sorted(per_bin.items())]
        for k in bins:
            edge(slot_nodes[k], sink, self.capacity, 0)
        for _ in group:
            distance = [float("inf")] * len(graph)
            parent = [None] * len(graph)
            distance[source] = 0
            queue, queued = deque([source]), {source}
            while queue:
                u = queue.popleft()
                queued.remove(u)
                for index, (v, cap, price, _) in enumerate(graph[u]):
                    if cap and distance[v] > distance[u] + price:
                        distance[v] = distance[u] + price
                        parent[v] = (u, index)
                        if v not in queued:
                            queued.add(v)
                            queue.append(v)
            if parent[sink] is None:
                raise AssertionError("Component infeasible after global feasibility check")
            v = sink
            while v != source:
                u, index = parent[v]
                arc = graph[u][index]
                arc[1] -= 1
                graph[v][arc[3]][1] += 1
                v = u
        result = {}
        for i, choices in selected.items():
            used = [s for s, arc in choices if arc[1] == 0]
            assert len(used) == 1
            result[i] = used[0]
        return result
