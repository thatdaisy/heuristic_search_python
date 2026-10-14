import random
import heapq
import time
import statistics
import csv

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GOAL_POSITIONS = {value: divmod(index, 3) for index, value in enumerate(GOAL_STATE)}


def print_state(state):
    """Print a puzzle state in 3x3 format."""
    for i in range(0, 9, 3):
        print(state[i:i+3])
    print()


def count_inversions(state):
    """
    Count inversions in the puzzle state.
    The blank tile (0) is ignored.
    Input: tuple of length 9
    Output: integer inversion count
    """
    tiles = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(state):
    """
    Check if an 8-puzzle state is solvable.
    For a 3x3 puzzle, solvable iff inversion count is even.
    Input: tuple of length 9
    Output: True/False
    """
    return count_inversions(state) % 2 == 0


def generate_random_state():
    """
    Generate a random solvable 8-puzzle state.
    Input: none
    Output: solvable tuple state
    """
    while True:
        state = list(range(9))
        random.shuffle(state)
        state = tuple(state)
        if is_solvable(state):
            return state


def get_neighbors(state):
    """
    Generate all valid neighbor states by moving the blank tile.
    Input: tuple state
    Output: list of neighboring tuple states
    """
    neighbors = []
    zero_index = state.index(0)
    row, col = divmod(zero_index, 3)

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in moves:
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_zero_index = new_row * 3 + new_col
            new_state = list(state)
            new_state[zero_index], new_state[new_zero_index] = new_state[new_zero_index], new_state[zero_index]
            neighbors.append(tuple(new_state))

    return neighbors


def heuristic_hamming(state):
    """
    Hamming distance:
    Number of misplaced tiles, excluding the blank.
    Input: tuple state
    Output: integer heuristic value
    """
    misplaced = 0
    for i in range(9):
        if state[i] != 0 and state[i] != GOAL_STATE[i]:
            misplaced += 1
    return misplaced


def heuristic_manhattan(state):
    """
    Manhattan distance:
    Sum of row/column distances of each tile from its goal position.
    Input: tuple state
    Output: integer heuristic value
    """
    distance = 0
    for index, tile in enumerate(state):
        if tile != 0:
            current_row, current_col = divmod(index, 3)
            goal_row, goal_col = GOAL_POSITIONS[tile]
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance


def reconstruct_path(came_from, current):
    """
    Reconstruct solution path from start to goal.
    Input: parent dictionary, goal state
    Output: list of states from start to goal
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star_search(start_state, heuristic_func):
    """
    Solve the 8-puzzle using A* search.

    Inputs:
    - start_state: tuple
    - heuristic_func: function

    Outputs:
    - solution_path: list of states from start to goal
    - nodes_expanded: integer
    - solution_depth: integer
    """
    open_heap = []
    heap_counter = 0

    g_score = {start_state: 0}
    came_from = {}
    closed_set = set()
    nodes_expanded = 0

    start_h = heuristic_func(start_state)
    heapq.heappush(open_heap, (start_h, heap_counter, start_state))

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        if current == GOAL_STATE:
            path = reconstruct_path(came_from, current)
            return path, nodes_expanded, len(path) - 1

        closed_set.add(current)
        nodes_expanded += 1

        current_g = g_score[current]

        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue

            tentative_g = current_g + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_func(neighbor)
                heap_counter += 1
                heapq.heappush(open_heap, (f_score, heap_counter, neighbor))

    return None, nodes_expanded, None


def benchmark_heuristic(test_states, heuristic_func, heuristic_name):
    """
    Run benchmark for a heuristic over multiple test states.

    Inputs:
    - test_states: list of tuple states
    - heuristic_func: heuristic function
    - heuristic_name: string

    Outputs:
    - dictionary with raw results and statistics
    """
    nodes_list = []
    times_list = []
    solution_lengths = []

    print(f'Running benchmark for {heuristic_name}...')

    for i, state in enumerate(test_states, start=1):
        start_time = time.perf_counter()
        path, nodes_expanded, solution_depth = a_star_search(state, heuristic_func)
        end_time = time.perf_counter()

        runtime = end_time - start_time

        nodes_list.append(nodes_expanded)
        times_list.append(runtime)
        solution_lengths.append(solution_depth)

        print(f'Test {i:3d}: nodes={nodes_expanded:6d}, time={runtime:.6f}s, depth={solution_depth}')

    return {
        'heuristic': heuristic_name,
        'nodes': nodes_list,
        'times': times_list,
        'depths': solution_lengths,
        'mean_nodes': statistics.mean(nodes_list),
        'std_nodes': statistics.stdev(nodes_list) if len(nodes_list) > 1 else 0,
        'mean_time': statistics.mean(times_list),
        'std_time': statistics.stdev(times_list) if len(times_list) > 1 else 0,
        'mean_depth': statistics.mean(solution_lengths),
        'std_depth': statistics.stdev(solution_lengths) if len(solution_lengths) > 1 else 0,
    }


def print_summary(result):
    """Print summary statistics for one heuristic."""
    print(f'=== {result["heuristic"]} SUMMARY ===')
    print(f'Mean expanded nodes : {result["mean_nodes"]:.2f}')
    print(f'Std dev nodes       : {result["std_nodes"]:.2f}')
    print(f'Mean runtime (s)    : {result["mean_time"]:.6f}')
    print(f'Std dev runtime (s) : {result["std_time"]:.6f}')
    print(f'Mean solution depth : {result["mean_depth"]:.2f}')
    print(f'Std dev depth       : {result["std_depth"]:.2f}')


def compare_results(hamming_result, manhattan_result):
    """Print a comparison table between heuristics."""
    print('=== COMPARISON TABLE ===')
    print(f'{"Metric":25s} {"Hamming":>15s} {"Manhattan":>15s}')
    print('-' * 60)
    print(f'{"Mean expanded nodes":25s} {hamming_result["mean_nodes"]:15.2f} {manhattan_result["mean_nodes"]:15.2f}')
    print(f'{"Std dev nodes":25s} {hamming_result["std_nodes"]:15.2f} {manhattan_result["std_nodes"]:15.2f}')
    print(f'{"Mean runtime (s)":25s} {hamming_result["mean_time"]:15.6f} {manhattan_result["mean_time"]:15.6f}')
    print(f'{"Std dev runtime (s)":25s} {hamming_result["std_time"]:15.6f} {manhattan_result["std_time"]:15.6f}')
    print(f'{"Mean solution depth":25s} {hamming_result["mean_depth"]:15.2f} {manhattan_result["mean_depth"]:15.2f}')


def save_results_csv(hamming_result, manhattan_result, filename='benchmark_results.csv'):
    """Save summary results to CSV."""
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Hamming', 'Manhattan'])
        writer.writerow(['Mean expanded nodes', hamming_result['mean_nodes'], manhattan_result['mean_nodes']])
        writer.writerow(['Std dev nodes', hamming_result['std_nodes'], manhattan_result['std_nodes']])
        writer.writerow(['Mean runtime (s)', hamming_result['mean_time'], manhattan_result['mean_time']])
        writer.writerow(['Std dev runtime (s)', hamming_result['std_time'], manhattan_result['std_time']])
        writer.writerow(['Mean solution depth', hamming_result['mean_depth'], manhattan_result['mean_depth']])


def main():
    """
    Main program:
    1. Generate 100 solvable random states
    2. Run A* with Hamming heuristic
    3. Run A* with Manhattan heuristic
    4. Print statistical comparison
    5. Save summary CSV
    """
    num_tests = 100

    print(f'Generating {num_tests} solvable random start states...')
    test_states = [generate_random_state() for _ in range(num_tests)]

    hamming_result = benchmark_heuristic(test_states, heuristic_hamming, 'Hamming Distance')
    manhattan_result = benchmark_heuristic(test_states, heuristic_manhattan, 'Manhattan Distance')

    print_summary(hamming_result)
    print_summary(manhattan_result)
    compare_results(hamming_result, manhattan_result)
    save_results_csv(hamming_result, manhattan_result)

    print('Results saved to benchmark_results.csv')
    print('Done.')


if __name__ == '__main__':
    main()
