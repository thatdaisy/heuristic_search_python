"""8-puzzle solver (A*) with Manhattan or Hamming heuristic.

Usage: python eight_puzzle.py --heuristic manhattan --runs 10 --showPath
"""
import argparse
import heapq
import itertools
import random
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1) ARRAY REPRESENTATION OF THE GOAL STATE
# The 3x3 board is stored as a flat array of 9 cells (index 0..8, row by row):
#     index:  0 1 2      value:  1 2 3
#             3 4 5              4 5 6
#             6 7 8              7 8 _   (0 = blank)
# A tuple is used because it is immutable and can be a dictionary key.
# ---------------------------------------------------------------------------
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GOAL_POS = {tile: i for i, tile in enumerate(GOAL)}   # tile -> goal index


# ---------------------------------------------------------------------------
# 2) DATA STRUCTURE FOR A STATE (NODE)
# ---------------------------------------------------------------------------
@dataclass
class Node:
    state: tuple                                # board: array of 9 cells
    g: int = 0                                  # cost so far: +1 per move
    h: int = 0                                  # estimated cost to goal
    path: list = field(default_factory=list)    # states visited so far

    @property
    def f(self):                                # total estimated cost
        return self.g + self.h


# ---------------------------------------------------------------------------
# 3) VALID TRANSITIONS
# For every possible blank position the table lists the cells it may swap
# with (up, down, left, right - but never over the edge of the board).
# It is built once; make_a_move() can only ever use moves from this table.
# ---------------------------------------------------------------------------
def build_valid_moves():
    table = {}
    for i in range(9):
        row, col = divmod(i, 3)
        table[i] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < 3 and 0 <= c < 3:       # stay inside the board
                table[i].append(r * 3 + c)
    return table


VALID_MOVES = build_valid_moves()   # e.g. {0: [3, 1], 4: [1, 7, 3, 5], ...}

stats = []  # one dict per run

def calculate_heuristic(node, heuristic):
    """Estimated cost to reach GOAL: 'manhattan' or 'hamming'."""
    tiles = [(i, t) for i, t in enumerate(node.state) if t != 0]
    if heuristic == "hamming":
        return sum(1 for i, t in tiles if t != GOAL[i])
    total = 0
    for i, t in tiles:
        goal_row, goal_col = divmod(GOAL_POS[t], 3)
        row, col = divmod(i, 3)
        total += abs(row - goal_row) + abs(col - goal_col)
    return total

def make_a_move(node, heuristic):
    """Slide the blank only along valid transitions; return the new nodes.
    Each new node has g + 1 and the taken path saved."""
    blank = node.state.index(0)
    children = []
    for target in VALID_MOVES[blank]:           # only legal swaps
        s = list(node.state)
        s[blank], s[target] = s[target], s[blank]
        child = Node(tuple(s), node.g + 1, 0, node.path + [tuple(s)])
        child.h = calculate_heuristic(child, heuristic)
        children.append(child)
    return children


def is_solvable(node):
    """Solvable if the number of inversions is even (3x3 board)."""
    tiles = [t for t in node.state if t != 0]
    inversions = sum(a > b for a, b in itertools.combinations(tiles, 2))
    return inversions % 2 == 0


def is_solved(node):
    return node.state == GOAL


def random_start_node(heuristic):
    """Random solvable start node."""
    while True:
        s = list(GOAL)
        random.shuffle(s)
        node = Node(tuple(s), 0, 0, [tuple(s)])
        if is_solvable(node):
            node.h = calculate_heuristic(node, heuristic)
            return node


# ---------------------------------------------------------------------------
# 4) HEAPQ AS PRIORITY QUEUE
# heapq keeps the list ordered so that heappop() always returns the entry
# with the smallest first value. Entries are (f, counter, node):
#   f       -> lowest estimated total cost is expanded first
#   counter -> unique tie-breaker, so two nodes are never compared directly
# ---------------------------------------------------------------------------
def orchestrate_search(node, heuristic=):
    """A*: repeatedly expand the lowest f = g + h node until solved."""
    start = time.perf_counter()
    counter = itertools.count()
    frontier = []                                            # the priority queue
    heapq.heappush(frontier, (node.f, next(counter), node))
    best_g = {node.state: 0}                                 # cheapest g per board
    expanded = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)              # best node first
        if is_solved(current):
            stats.append({"moves": current.g, "expanded": expanded,
                          "time": time.perf_counter() - start})
            return current
        expanded += 1
        for child in make_a_move(current, heuristic):
            if child.g < best_g.get(child.state, float("inf")):
                best_g[child.state] = child.g
                heapq.heappush(frontier, (child.f, next(counter), child))
    return None


def calc_statistics():
    """Return a printable summary of all runs."""
    if not stats:
        return "No runs."
    n = len(stats)
    lines = [f"Runs: {n}"]
    for key, label in (("moves", "Solution length"), ("expanded", "Nodes expanded"),
                       ("time", "Time (s)")):
        vals = [s[key] for s in stats]
        lines.append(f"{label:16}: avg {sum(vals) / n:.3f} | min {min(vals):.3f} | max {max(vals):.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="8-puzzle A* solver")
    parser.add_argument("--heuristic", choices=["manhattan", "hamming"], default="manhattan")
    parser.add_argument("--runs", type=int, default=10, help="number of test runs")
    parser.add_argument("--showPath", action="store_true", help="show last solution path")
    args = parser.parse_args()

    print("\n### Introduction to AI and Data Science - Task 1: Heuristic Search ###")
    print(f">>> Heuristic: {args.heuristic} - Executions: {args.runs}\n")

    for i in range(1, args.runs + 1):
        start = random_start_node(args.heuristic)
        solved = orchestrate_search(start, args.heuristic)
        s = stats[-1]
        print(f"Run {i}: start={start.state} moves={s['moves']} expanded={s['expanded']}")
    print("\n" + calc_statistics())
    if args.showPath:
        print("\nLast solution path:")
        for step in solved.path:
            print(step[:3], step[3:6], step[6:], sep="\n", end="\n\n")


if __name__ == '__main__':
    main()