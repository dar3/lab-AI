import heapq
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import time
import time
from functools import wraps
import pandas as pd


@dataclass
class TimeInformation:
    """ class for keeping info about time"""
    hour: int
    minute: int
    minAfterOO: int

    def __init__(self, str: str):
        self.hour, self.minute = int(str[0:2]), int(str[3:5])
        self.minAfterOO = self.hour * 60 + self.minute

    def __str__(self):
        return f"{self.hour:02d}:{self.minute:02d}:00"

    def __lt__(self, other):
        return self.minAfterOO < other.minAfterOO

    def __le__(self, other):
        return self.minAfterOO <= other.minAfterOO

    def __gt__(self, other):
        return self.minAfterOO > other.minAfterOO

    def __ge__(self, other):
        return self.minAfterOO >= other.minAfterOO


@dataclass
class VerticleInformation:
    name: str
    latitiude: float
    longitude: float


@dataclass
class EdgeInformation:
    line: str
    departure_time: TimeInformation
    arrival_time: TimeInformation
    start_stop: str
    end_stop: str


class Node:
    def __init__(self, geo_info: VerticleInformation):
        self.geo_info = geo_info
        self.edges = list()  # lista krawedzi do sasiednich wezlow
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0.0
        self.parent_name = ""
        self.parent_edge = None  # Track the edge used to reach this node

    def __str__(self):
        return f"Node {self.geo_info.name}"

    def set_neighbour(self, ride: EdgeInformation):
        """ creates a list of neighbours to the node"""
        self.edges.append(ride)
        return self

    @staticmethod
    def edge_time_cost(edg: EdgeInformation):
        return edg.arrival_time.minAfterOO - edg.departure_time.minAfterOO

def create_graph(filename: str):
    graph: dict[str, Node] = {}

    df = pd.read_csv(filename,
                     dtype={"id": int, "company": str, "line": str, "departure_time": str, "arrival_time": str
                         , "start_stop": str, "end_stop": str, "start_stop_lat": float, "start_stop_lon": float,
                            "end_stop_lat": float, "end_stop_lon": float})
    df.sort_values(['line', 'departure_time', 'start_stop'])

    nodes_created = 0
    nodes_count = 0
    print("Creating nodes")
    for index, row in df.iterrows():
        ride_info = EdgeInformation(
            row['line'],
            TimeInformation(row['departure_time']),
            TimeInformation(row['arrival_time']),
            row['start_stop'],
            row['end_stop']
        )

        start_info = VerticleInformation(
            row['start_stop'],
            float(row['start_stop_lat']),
            float(row['start_stop_lon'])
        )

        stop_info = VerticleInformation(
            row['end_stop'],
            float(row['end_stop_lat']),
            float(row['end_stop_lon'])
        )

        if row['start_stop'] not in graph:
            graph[row['start_stop']] = Node(start_info)
            nodes_created += 1
            print(".", end='')

        if row['end_stop'] not in graph:
            graph[row['end_stop']] = Node(stop_info)
            nodes_created += 1
            print(".", end='')

        graph[row['start_stop']].set_neighbour(ride_info)

        if nodes_created % 100 == 0 and nodes_created > nodes_count:
            print(f"{nodes_created} nodes created")
        nodes_count = nodes_created

    print(f"\n{nodes_created} nodes created totally")
    return graph


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds', file=sys.stderr)
        return result

    return timeit_wrapper


def get_valid_edges(node: Node, current_time: TimeInformation):
    """Returns edges that depart after the current time"""
    valid_edges = []
    for edge in node.edges:
        # Handle overnight trips (departure next day)
        if edge.departure_time.minAfterOO >= current_time.minAfterOO:
            valid_edges.append(edge)
        elif edge.departure_time.minAfterOO + 24 * 60 >= current_time.minAfterOO:
            # Create a copy with adjusted times for overnight trips
            adjusted_edge = EdgeInformation(
                edge.line,
                TimeInformation(f"{edge.departure_time.hour:02d}:{edge.departure_time.minute:02d}:00"),
                TimeInformation(f"{edge.arrival_time.hour:02d}:{edge.arrival_time.minute:02d}:00"),
                edge.start_stop,
                edge.end_stop
            )
            adjusted_edge.departure_time.minAfterOO += 24 * 60
            adjusted_edge.arrival_time.minAfterOO += 24 * 60
            valid_edges.append(adjusted_edge)
    return valid_edges


def neighbours(graph: dict[str, Node], node: Node, current_time: TimeInformation):
    """Returns neighboring nodes reachable from the current node after current_time"""
    neighbors = set()
    for edge in get_valid_edges(node, current_time):
        neighbors.add(graph[edge.end_stop])
    return list(neighbors)


def cost_fun_for_time(current_node: Node, next_node: Node, current_time: TimeInformation, prev_line: str):
    """Cost function for time minimization"""
    min_cost = float('inf')
    best_edge = None

    for edge in get_valid_edges(current_node, current_time):
        if edge.end_stop != next_node.geo_info.name:
            continue

        # Calculate waiting time + travel time
        wait_time = edge.departure_time.minAfterOO - current_time.minAfterOO
        if wait_time < 0:
            wait_time += 24 * 60

        travel_time = edge.arrival_time.minAfterOO - edge.departure_time.minAfterOO
        total_cost = wait_time + travel_time

        if total_cost < min_cost:
            min_cost = total_cost
            best_edge = edge

    return min_cost, best_edge


def cost_fun_for_switch(current_node: Node, next_node: Node, current_time: TimeInformation, prev_line: str):
    """Cost function that strongly prefers direct connections"""
    min_cost = float('inf')
    best_edge = None

    for edge in get_valid_edges(current_node, current_time):
        if edge.end_stop != next_node.geo_info.name:
            continue

        # Calculate waiting time
        wait_time = edge.departure_time.minAfterOO - current_time.minAfterOO
        if wait_time < 0:
            wait_time += 24 * 60

        # Check if this is a transfer
        is_transfer = prev_line and edge.line != prev_line

        # Base cost components
        base_cost = wait_time * 0.1  # Small penalty for waiting

        # Massive penalty for transfers (10000 per transfer)
        transfer_penalty = 10000 if is_transfer else 0
        # transfer_penalty = 999999999 if is_transfer else 0

        # Additional bonus for routes that go directly to final destination
        final_destination_bonus = 0
        if edge.end_stop == end_stop_name:
            final_destination_bonus = -5000

        total_cost = base_cost + transfer_penalty + final_destination_bonus

        if total_cost < min_cost:
            min_cost = total_cost
            best_edge = edge

    return min_cost, best_edge


def cost_fun_for_switch_modified(current_node: Node, next_node: Node, current_time: TimeInformation, prev_line: str):
    """Modified version that considers distance and direct connections"""
    min_cost = float('inf')
    best_edge = None

    for edge in get_valid_edges(current_node, current_time):
        if edge.end_stop != next_node.geo_info.name:
            continue

        wait_time = edge.departure_time.minAfterOO - current_time.minAfterOO
        if wait_time < 0:
            wait_time += 24 * 60

        is_transfer = prev_line and edge.line != prev_line

        # Distance-based components
        distance = h(current_node, next_node)
        distance_cost = distance * 0.01  # Small distance penalty

        # Transfer penalty scales with distance
        transfer_penalty = 5000 * distance if is_transfer else 0

        # Bonus for direct connections to final destination
        final_dest_bonus = -2000 * distance if edge.end_stop == end_stop_name else 0

        total_cost = (wait_time * 0.1) + distance_cost + transfer_penalty + final_dest_bonus

        if total_cost < min_cost:
            min_cost = total_cost
            best_edge = edge

    return min_cost, best_edge


def h(curr: Node, goal: Node):
    """Heuristic function - Euclidean distance"""
    return math.sqrt((curr.geo_info.latitiude - goal.geo_info.latitiude) ** 2 +
                     (curr.geo_info.longitude - goal.geo_info.longitude) ** 2)
    return 0


@timeit
def astar_search(graph, start, goal, cost_fn, start_time_str):
    """A* search algorithm with proper edge tracking"""
    start_time = TimeInformation(start_time_str)

    # Reset graph state
    for node in graph.values():
        node.g = float('inf')
        node.f = float('inf')
        node.parent_name = ""
        node.parent_edge = None

    # Initialize start node
    start_node = graph[start]
    start_node.g = 0
    start_node.h = h(start_node, graph[goal])
    start_node.f = start_node.g + start_node.h

    open_set = []
    heapq.heappush(open_set, (start_node.f, start))

    # visited = set()

    while open_set:
        current_f, current_name = heapq.heappop(open_set)
        current_node = graph[current_name]

        if current_name == goal:
            return graph

        # if current_name in visited:
        #     continue
        # visited.add(current_name)

        # Get current time based on how we reached this node
        if current_node.parent_edge is None:
            current_time = start_time  # Starting at initial time
            prev_line = ""
        else:
            current_time = current_node.parent_edge.arrival_time
            prev_line = current_node.parent_edge.line

        # Explore neighbors
        for neighbor in neighbours(graph, current_node, current_time):
            # Calculate cost to neighbor
            cost, edge = cost_fn(current_node, neighbor, current_time, prev_line)

            if edge is None:
                continue  # Skip if no valid edge found

            tentative_g = current_node.g + cost

            if tentative_g < neighbor.g:
                neighbor.g = tentative_g
                neighbor.h = h(neighbor, graph[goal])
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent_name = current_name
                neighbor.parent_edge = edge  # Track the edge used
                heapq.heappush(open_set, (neighbor.f, neighbor.geo_info.name))

    return None


@timeit
def dijkstra(graph, start, goal, cost_fn, start_time_str):
    """Dijkstra's algorithm with proper edge tracking"""
    start_time = TimeInformation(start_time_str)

    # Reset graph state
    for node in graph.values():
        node.g = float('inf')
        node.parent_name = ""
        node.parent_edge = None

    # Initialize start node
    start_node = graph[start]
    start_node.g = 0

    open_set = []
    heapq.heappush(open_set, (start_node.g, start))

    # visited = set()

    while open_set:
        current_g, current_name = heapq.heappop(open_set)
        current_node = graph[current_name]

        if current_name == goal:
            return graph

        # if current_name in visited:
        #     continue
        # visited.add(current_name)

        # Get current time based on how we reached this node
        if current_node.parent_edge is None:
            current_time = start_time  # Starting at initial time
            prev_line = ""
        else:
            current_time = current_node.parent_edge.arrival_time
            prev_line = current_node.parent_edge.line

        # Explore neighbors
        for neighbor in neighbours(graph, current_node, current_time):
            # Calculate cost to neighbor
            cost, edge = cost_fn(current_node, neighbor, current_time, prev_line)

            if edge is None:
                continue  # Skip if no valid edge found

            tentative_g = current_node.g + cost

            if tentative_g < neighbor.g:
                neighbor.g = tentative_g
                neighbor.parent_name = current_name
                neighbor.parent_edge = edge  # Track the edge used
                heapq.heappush(open_set, (neighbor.g, neighbor.geo_info.name))

    return None


def read_path(graph: dict[str, Node], end: str):
    """Reconstructs the path from end to start"""
    path = []
    current_node = graph[end]

    while current_node.parent_name:
        path.append(current_node)
        current_node = graph[current_node.parent_name]

    path.append(current_node)  # Add the start node
    path.reverse()
    return path


def path_with_information(path: list[Node], start_time_str: str):
    """Prints the journey information"""
    if not path or len(path) < 2:
        print("No valid path found")
        return

    start_time = TimeInformation(start_time_str)
    current_time = start_time
    total_time = 0
    num_changes = 0
    prev_line = None

    print("\nJourney details:")

    try:
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            # Find the edge used to get to next_node
            edge = next_node.parent_edge

            if edge is None:
                print(f"Error: Missing connection between {current_node.geo_info.name} and {next_node.geo_info.name}")
                return

            # Calculate waiting time
            wait_time = edge.departure_time.minAfterOO - current_time.minAfterOO
            if wait_time < 0:
                wait_time += 24 * 60

            travel_time = edge.arrival_time.minAfterOO - edge.departure_time.minAfterOO
            total_time += wait_time + travel_time

            # Print segment information
            print(f"Line {edge.line}:")
            print(f"  Board at {edge.start_stop} at {edge.departure_time}")
            print(f"  Get off at {edge.end_stop} at {edge.arrival_time}")
            print(f"  Travel time: {travel_time} minutes")
            if wait_time > 0:
                print(f"  Wait time: {wait_time} minutes")

            # Check for transfers
            if prev_line is not None and edge.line != prev_line:
                num_changes += 1
                print(f"  * Transfer from {prev_line} to {edge.line}")

            prev_line = edge.line
            current_time = edge.arrival_time

        # Handle total time that might span midnight
        if current_time.minAfterOO < start_time.minAfterOO:
            total_time = (24 * 60 - start_time.minAfterOO) + current_time.minAfterOO
        else:
            total_time = current_time.minAfterOO - start_time.minAfterOO

        print("\nSummary:")
        print(f"Total journey time: {total_time} minutes")
        print(f"Ilość przesiadek: {num_changes}")
        print(f"Total cost: {path[-1].g}", file=sys.stderr)

    except AttributeError as e:
        print(f"Error reconstructing path: {e}", file=sys.stderr)
        return


def save_graph(graph, filename):
    with open(filename, 'wb') as file:
        pickle.dump(graph, file)


def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def get_user_input():
    # start, end, criterion, departure_time = "Renoma", "PORT LOTNICZY", "t", "16:48:00"
    start, end, criterion, departure_time = "Renoma", "PORT LOTNICZY", "t", "10:00:00"
    print("Wyszukiwanie trasy")
    return start, end, criterion, departure_time
    start = input("Przystanek początkowy: ").strip()
    end = input("Przystanek końcowy: ").strip()
    departure_time = input("Godzina odjazdu HH:MM:SS : ").strip()
    while True:
        criterion = input("t - minimalizacja czasu dojazdu, p - minimalizacja liczby przesiadek: ").strip().lower()
        if criterion in ['t', 'p']:
            break
        print("Proszę podać 't' dla czasu lub 'p' dla przesiadek")
    return start, end, criterion, departure_time



# Global variable for destination stop name
end_stop_name = ""


def main():
    global end_stop_name

    GRAPH_FILE = "graph_data.pickle"

    # Try to load existing graph
    try:
        graph = load_graph(GRAPH_FILE)
        print("Graph loaded successfully from file.")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print("Creating new graph from CSV file...")
        try:
            graph = create_graph("connection_graph.csv")
            save_graph(graph, GRAPH_FILE)
            print("Graph created and saved to file.")
        except Exception as e:
            print(f"Error creating graph: {e}", file=sys.stderr)
            return

    # Get user input
    start, end, criterion, departure_time = get_user_input()
    end_stop_name = end  # Set the global destination

    # Verify stops exist in graph
    if start not in graph:
        print(f"Error: Starting stop '{start}' not found in graph.", file=sys.stderr)
        return
    if end not in graph:
        print(f"Error: Destination stop '{end}' not found in graph.", file=sys.stderr)
        return

    print(f"\nSearching route from {start} to {end} at {departure_time}")

    if criterion == 't':
        # Time optimization
        print("\n=== Dijkstra Algorithm (Time Optimization) ===")
        time_start = time.perf_counter()
        result = dijkstra(graph, start, end, cost_fun_for_time, departure_time)
        time_end = time.perf_counter()

        if result:
            path = read_path(result, end)
            path_with_information(path, departure_time)
            print(f"Algorithm execution time: {(time_end - time_start):.4f}s", file=sys.stderr)
        else:
            print("No path found with Dijkstra")

        print("\n=== A* Algorithm (Time Optimization) ===")
        time_start = time.perf_counter()
        result = astar_search(graph, start, end, cost_fun_for_time, departure_time)
        time_end = time.perf_counter()

        if result:
            path = read_path(result, end)
            path_with_information(path, departure_time)
            print(f"Algorithm execution time: {(time_end - time_start):.4f}s", file=sys.stderr)
        else:
            print("No path found with A*")

    else:
        # Transfer optimization
        print("\n=== A* Algorithm (Transfer Optimization) ===")
        time_start = time.perf_counter()
        result = astar_search(graph, start, end, cost_fun_for_switch, departure_time)
        time_end = time.perf_counter()

        if result:
            path = read_path(result, end)
            path_with_information(path, departure_time)
            print(f"Algorithm execution time: {(time_end - time_start):.4f}s", file=sys.stderr)
        else:
            print("No path found with A* (Transfer Optimization)")

        print("\n=== Modified A* Algorithm (Smart Transfer Optimization) ===")
        time_start = time.perf_counter()
        result = astar_search(graph, start, end, cost_fun_for_switch_modified, departure_time)
        time_end = time.perf_counter()

        if result:
            path = read_path(result, end)
            path_with_information(path, departure_time)
            print(f"Algorithm execution time: {(time_end - time_start):.4f}s", file=sys.stderr)
        else:
            print("No path found with Modified A*")



if __name__ == "__main__":
    main()