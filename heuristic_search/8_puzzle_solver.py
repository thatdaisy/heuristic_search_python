"""8-puzzle solver (A*) with Manhattan or Hamming heuristic.

Usage: python 8_puzzle_solver.py --heuristic manhattan --runs 10 --showPath
       python 8_puzzle_solver.py --compare

Overview of the program flow:
    main() -> random_start_node() -> orchestrate_search() -> make_a_move() -> calculate_heuristic() -> calc_statistics()
"""
import argparse
import heapq
import itertools
import random
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# ARRAY REPRESENTATION OF THE GOAL STATE
# The 3x3 board is stored as a flat array of 9 cells (index 0..8, row by row):
#     index:  0 1 2      value:  1 2 3
#             3 4 5              4 5 6
#             6 7 8              7 8 _   (0 = blank)
# A tuple is used because it is immutable and can be a dictionary key.
# ---------------------------------------------------------------------------
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GOAL_POS = {tile: i for i, tile in enumerate(GOAL_STATE)}   # tile -> goal index

COMPARE_RUNS = 100  # random searches per heuristic in --compare mode


# ---------------------------------------------------------------------------
# DATA STRUCTURE FOR A STATE (NODE)
# ---------------------------------------------------------------------------
@dataclass
class Node:
    """
    Function: Bundles everything the search needs to know about one board.
    Fields:   state - tuple of 9 cells, the board (0 = blank)
              g     - int, cost so far (number of moves, +1 per move)
              h     - int, estimated cost to the goal (heuristic value)
              path  - list of states, all boards visited from the start to here
    Property: f     - int, g + h, the total estimated cost (used for ranking)
    """
    state: tuple
    g: int = 0
    h: int = 0
    path: list = field(default_factory=list)

    @property
    def f(self):
        return self.g + self.h


# ---------------------------------------------------------------------------
# VALID TRANSITIONS
# For every possible blank position the table lists the cells it may swap
# with (up, down, left, right - but never over the edge of the board).
# It is built once; make_a_move() can only ever use moves from this table.
# ---------------------------------------------------------------------------
def build_valid_moves():
    """
    Function: Builds the table of legal blank moves for a 3x3 board. For each
              blank position it checks the four directions and keeps only
              those that stay inside the board.
    Input:    none
    Output:   dict {blank index (0-8): list of indices it may swap with},
              e.g. {0: [3, 1], 4: [1, 7, 3, 5], ...}
    """
    table = {}
    for i in range(9):
        row, col = divmod(i, 3)
        table[i] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < 3 and 0 <= c < 3:       # stay inside the board
                table[i].append(r * 3 + c)
    return table


VALID_MOVES = build_valid_moves()   # built once, used by make_a_move()

stats = []  # global list, one dict per finished search (filled by orchestrate_search)


def calculate_heuristic(node, heuristic):
    """
    Function: Estimates the remaining cost h(n) from a board to GOAL_STATE.
              Selects the Hamming or Manhattan calculation. The blank is
              excluded because it is not a real tile.
    Input:    node      - Node, the board to evaluate
              heuristic - str, "hamming" or "manhattan"
    Output:   int, estimated number of moves still needed
    """
    tiles = [(i, t) for i, t in enumerate(node.state) if t != 0]
    if heuristic == "hamming":
        return calculate_hamming(tiles)
    return calculate_manhattan(tiles)


# ---------------------------------------------------------------------------
# HAMMING DISTANCE
# ---------------------------------------------------------------------------
def calculate_hamming(tiles):
    """
    Function: Counts the tiles that are not on their goal square.
    Input:    tiles - list of (position, tile number) pairs, without the blank
    Output:   int, number of misplaced tiles
    """
    return sum(1 for i, t in tiles if t != GOAL_STATE[i])


# ---------------------------------------------------------------------------
# MANHATTAN DISTANCE
# ---------------------------------------------------------------------------
def calculate_manhattan(tiles):
    """
    Function: For every tile, sums up the horizontal plus vertical steps
              between its current square and its goal square.
    Input:    tiles - list of (position, tile number) pairs, without the blank
    Output:   int, total number of steps all tiles are away from home
    """
    total = 0
    for i, t in tiles:
        goal_row, goal_col = divmod(GOAL_POS[t], 3)
        row, col = divmod(i, 3)
        total += abs(row - goal_row) + abs(col - goal_col)
    return total


def make_a_move(node, heuristic):
    """
    Function: Expands a node: slides the blank along every valid transition
              (see VALID_MOVES) and creates one new node per legal move. Each
              new node has g + 1, its own h value, and the path taken so far.
    Input:    node      - Node, the board to expand
              heuristic - str, "hamming" or "manhattan" (for the new nodes' h)
    Output:   list of Node, the 2-4 successor boards
    """
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
    """
    Function: Checks whether the board can reach the goal at all. On a 3x3
              board this is the case if the number of inversions (pairs of
              tiles in the wrong order, blank ignored) is even.
    Input:    node - Node, the board to check
    Output:   bool, True if solvable
    """
    tiles = [t for t in node.state if t != 0]
    inversions = sum(a > b for a, b in itertools.combinations(tiles, 2))
    return inversions % 2 == 0


def is_solved(node):
    """
    Function: Checks whether the board equals the goal state.
    Input:    node - Node, the board to check
    Output:   bool, True if the board is solved
    """
    return node.state == GOAL_STATE


def random_start_node(heuristic):
    """
    Function: Shuffles the tiles until the board is solvable and wraps it
              into a start node (g = 0, h calculated, path = [start board]).
    Input:    heuristic - str, "hamming" or "manhattan" (for the start node's h)
    Output:   Node, a random solvable start node
    """
    while True:
        s = list(GOAL_STATE)
        random.shuffle(s)
        node = Node(tuple(s), 0, 0, [tuple(s)])
        if is_solvable(node):
            node.h = calculate_heuristic(node, heuristic)
            return node


# ---------------------------------------------------------------------------
# HEAPQ AS PRIORITY QUEUE
# heapq keeps the list ordered so that heappop() always returns the entry
# with the smallest first value. Entries are (f, counter, node):
#   f       -> lowest estimated total cost is expanded first
#   counter -> unique tie-breaker, so two nodes are never compared directly
# ---------------------------------------------------------------------------
def orchestrate_search(node, heuristic):
    """
    Function: A* search. Repeatedly takes the node with the lowest f = g + h
              from the priority queue; if it is the goal the search ends,
              otherwise its successors (make_a_move) are added to the queue.
              A board is only re-added if it was reached with lower cost than
              before (best_g). On success, moves, expanded nodes and time are
              appended to the global stats list.
    Input:    node      - Node, the start node
              heuristic - str, "hamming" or "manhattan"
    Output:   Node, the solved node (g = solution length, path = all boards
              from start to goal); None if no solution exists
    """
    start = time.perf_counter()
    counter = itertools.count()
    priority_queue = []
    heapq.heappush(priority_queue, (node.f, next(counter), node))
    best_g = {node.state: 0}                                 # cheapest g per board
    expanded = 0
    while priority_queue:
        _, _, current = heapq.heappop(priority_queue)        # best node first
        if is_solved(current):
            stats.append({"moves": current.g, "expanded": expanded,
                          "time": time.perf_counter() - start})
            return current
        expanded += 1
        for child in make_a_move(current, heuristic):
            if child.g < best_g.get(child.state, float("inf")):
                best_g[child.state] = child.g
                heapq.heappush(priority_queue, (child.f, next(counter), child))
    return None


def calc_statistics(results=None):
    """
    Function: Summarises the finished searches: average, minimum and maximum
              of solution length, expanded nodes (memory) and time.
    Input:    results - list of dicts with keys "moves", "expanded", "time";
                        optional, defaults to the global stats list
    Output:   str, a printable multi-line summary ("No runs." if empty)
    """
    results = stats if results is None else results
    if not results:
        return "No runs."
    n = len(results)
    lines = [f"Runs: {n}"]
    for key, label in (("moves", "Solution length"), ("expanded", "Nodes expanded"),
                       ("time", "Time (s)")):
        vals = [s[key] for s in results]
        lines.append(f"{label:16}: avg {sum(vals) / n:.3f} | min {min(vals):.3f} | max {max(vals):.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# COMPARISON OF THE TWO HEURISTICS
# Both heuristics solve the SAME random start boards, so the comparison is fair.
#   memory usage     -> number of expanded nodes
#   computation time -> seconds per search
# ---------------------------------------------------------------------------
def compare_heuristics(n=COMPARE_RUNS):
    """
    Function: Generates n random solvable boards once, solves each of them
              with Hamming and with Manhattan, prints the statistics per
              heuristic and a comparison table (avg expanded nodes = memory,
              avg seconds = time, plus the ratio Hamming / Manhattan).
    Input:    n - int, number of random searches per heuristic (default 100)
    Output:   none (prints to the console; clears and refills the global stats)
    """
    boards = [random_start_node("manhattan").state for _ in range(n)]
    results = {}
    for heuristic in ("hamming", "manhattan"):
        stats.clear()
        for state in boards:
            node = Node(state, 0, 0, [state])
            node.h = calculate_heuristic(node, heuristic)   # h depends on the heuristic
            orchestrate_search(node, heuristic)
        results[heuristic] = list(stats)
        print(f"--- {heuristic.upper()} ---\n{calc_statistics(results[heuristic])}\n")

    def avg(heuristic, key):
        """Average of one measure (key) over all runs of one heuristic."""
        return sum(s[key] for s in results[heuristic]) / n

    print(f"=== Comparison ({n} identical random start boards each) ===")
    print(f"{'':28}{'Hamming':>12}{'Manhattan':>12}{'Ratio':>9}")
    for key, label in (("expanded", "Memory: avg nodes expanded"), ("time", "Time: avg seconds")):
        ham, man = avg("hamming", key), avg("manhattan", key)
        print(f"{label:28}{ham:>12.3f}{man:>12.3f}{ham / man:>8.1f}x")


def main():
    """
    Function: Entry point. Reads the command-line arguments, then either runs
              the heuristic comparison (--compare) or the requested number of
              random searches with one heuristic, and prints the results.
    Input:    command-line arguments:
                --heuristic manhattan|hamming  (default manhattan)
                --runs N                       (default 10)
                --showPath                     print the last solution path
                --compare                      compare both heuristics
    Output:   none (prints to the console)
    """
    parser = argparse.ArgumentParser(description="8-puzzle A* solver")
    parser.add_argument("--heuristic", choices=["manhattan", "hamming"], default="manhattan")
    parser.add_argument("--runs", type=int, default=10, help="number of test runs")
    parser.add_argument("--showPath", action="store_true", help="show last solution path")
    parser.add_argument("--compare", action="store_true",
                        help=f"compare both heuristics ({COMPARE_RUNS} random searches each)")
    args = parser.parse_args()

    print("\n### Introduction to AI and Data Science - Task 1: Heuristic Search ###")
    if args.compare:
        print(f">>> Comparing Hamming vs. Manhattan - {COMPARE_RUNS} executions each\n")
        compare_heuristics()
        return
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