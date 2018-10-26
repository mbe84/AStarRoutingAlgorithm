from search import *
import math, heapq
from itertools import *

class RoutingGraph(Graph):
    """Implementation of the RoutingGraph class."""

    def __init__(self, map_str):
        self.directions = [('N', -1, 0),
                  ('NE', -1, 1),
                  ('E', 0, 1),
                  ('SE', 1, 1),
                  ('S', 1, 0),
                  ('SW', 1, -1),
                  ('W', 0, -1),
                  ('NW', -1, -1)]
        self.map = map_str.split('\n')
        self.answer = []
        self.goal = []
        for y in self.map:
            count = 0
            for x in y.strip():
                if (x == 'G'):
                    self.goal.append((self.map.index(y), count))
                count += 1
        for item in self.map:
            self.answer.append(list(item.strip()))
        self.starting_place = self.starting_nodes()


    def is_goal(self, node):
        """Returns true if the given node is a goal state, false otherwise."""
        x1, y1, c1 = node;
        for y in self.map:
            for x in y.strip():
                if(x=='G'):
                    if((self.map.index(y), y.strip().index(x)) == (x1, y1)):
                        return True
        return False


    def starting_nodes(self):
        """Returns a sequence of starting nodes. Often there is only one
        starting node but even then the function returns a sequence
        with one element. It can be implemented as an iterator if
        needed.

        """
        for y in self.map:
            count = 0
            for x in y.strip():
                if(x=='S'):
                    yield (self.map.index(y), count, math.inf)
                elif(x.isnumeric()):
                     yield (self.map.index(y), count, int(x))
                count += 1


    def outgoing_arcs(self, tail_node):
        """Given a node it returns a sequence of arcs (Arc objects)
        which correspond to the actions that can be taken in that
        state (node)."""
        row_pos, col_pos, fuel = tail_node
        if fuel > 0:
            for direction in self.directions:
                direct, row_dir, col_dir = direction
                valid = [' ', 'S', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'G', 'F']
                if self.map[row_pos + row_dir].strip()[col_pos + col_dir] in valid:
                    yield Arc(tail_node, (row_pos + row_dir, col_pos+ col_dir, fuel-1), direct, 2)
        if self.map[row_pos].strip()[col_pos] == 'F' and fuel < 9:
            # if(fuel-1+5 >= 9):
            #     fuel = 9
            # else:
            #     fuel = fuel - 1 + 5
            yield Arc(tail_node, (row_pos, col_pos, 9), 'Fuel up', 5)

    def estimated_cost_to_goal(self, node):
        """Return the estimated cost to a goal node from the given
        state. This function is usually implemented when there is a
        single goal state. The function is used as a heuristic in
        search. The implementation should make sure that the heuristic
        meets the required criteria for heuristics."""
        ny, nx, label = node
        gy, gx = self.goal[0]

        return 2 * (abs(gy - ny) + abs(gx - nx)) + (2 - 2 * 2) * min(abs(gy - ny), abs(gx - nx))


class AStarFrontier(Frontier):
    """Implementation of the A*Frontier class."""

    def __init__(self, map_graph):
        """The constructor takes no argument. It initialises the
                container to an empty list."""
        self.myheapq = []
        self.map = map_graph
        self.visited = set()
        self.number = count()

    def add(self, path):
        """Adding a path to the container list"""
        cost = 0
        for i in path:
            cost += i.cost
        coord = (path[-1].head[0], path[-1].head[1])
        if coord not in self.visited:
            cost += self.map.estimated_cost_to_goal(path[-1].head)
            heapq.heappush(self.myheapq, (cost,next(self.number) , path))

    def __iter__(self):
        """Iterate throw the heapq"""

        while len(self.myheapq) != 0:
            cost, _, path = heapq.heappop(self.myheapq)
            if(path[-1]) not in self.visited:
                self.visited.add(path[-1])
                yield path



def print_map(RoutingGraph, Frontier, Solution):
    """Print out all the things"""
    map1 = RoutingGraph.answer
    map = RoutingGraph.answer

    for point in Frontier.visited:
        row, column = point[1][:2]
        map_point = map[row][column]
        if map_point != 'S' and map_point != 'G':
            map[row][column] = '.'
    if Solution is not None:
        for point in Solution:
                ans = map[point.head[0]][point.head[1]]
                if ans != 'S' and ans != 'G':
                    map[point.head[0]][point.head[1]] = '*'
    else:
        map = map1
    for i in map:
        for x in i:
            print(x, end="")
        print()


map_str = """\
+---------------+
|    G          |
|XXXXXXXXXXXX   |
|           X   |
|  XXXXXX   X   |
|  X S  X   X   |
|  X        X   |
|  XXXXXXXXXX   |
|               |
+---------------+
"""

map_graph = RoutingGraph(map_str)
frontier = AStarFrontier(map_graph)
solution = next(generic_search(map_graph, frontier), None)
print_map(map_graph, frontier, solution)


map_str = """\
+-------------+
|             |
|             |
|     S       |
|             |
| G           |
+-------------+
"""

map_graph = RoutingGraph(map_str)
frontier = AStarFrontier(map_graph)
solution = next(generic_search(map_graph, frontier), None)
print_map(map_graph, frontier, solution)

map_str = """\
+------------+
|         X  |
| S       X G|
|         X  |
|         X  |
|         X  |
+------------+
"""

map_graph = RoutingGraph(map_str)
frontier = AStarFrontier(map_graph)
solution = next(generic_search(map_graph, frontier), None)
print_map(map_graph, frontier, solution)


