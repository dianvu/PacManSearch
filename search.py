# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    # node <- NODE(problem.INITIAL)
    startNode = {'state': problem.getStartState(), 'action': (), 'cost': 0}  # Action as tuple
    # if problem.IS-GOAL(node.STATE) then return node
    if problem.isGoalState(startNode['state']):
        return []
    # frontier <- a LIFO stack with node as an element
    frontier = util.Stack()
    frontier.push(startNode)
    # reached <- {problem.INITAL}
    reached = set()
    # while not IS-EMPTY(frontier) do
    while not frontier.isEmpty():
        # node <- POP(frontier)
        node = frontier.pop()
        # if problem.IS-GOAL(node.STATE) then return node action
        if problem.isGoalState(node['state']):
            actions = [] # Create empty list
            actions.append(node['action']) # Append node action (tuple) to actions
            print([item for action in actions for item in action])
            return [item for action in actions for item in action]  # Convert list of tuple actions to list as in game.Directions
        # for each child in EXPAND(problem, node) do
        successors = problem.getSuccessors(node['state'])
        for successor in successors:
            # child <- CHILD-NODE(state, action, cost)
            child = {'state': successor[0],'action': successor[1], 'cost': successor[2]}
            # if child.STATE is not in reached
            if child['state'] not in reached:
                # add child.STATE to reached
                reached.add(child['state'])
                child['action'] = tuple(node['action']) + (child['action'],) # Concatenate actions
                child['cost'] = node['cost'] + child['cost'] # Add cost
                # add child to frontier
                frontier.push(child)
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # node <- NODE(problem.INITIAL)
    startNode = {'state': problem.getStartState(), 'action': (), 'cost': 0}  # Convert action list to tuple
    # if problem.IS-GOAL(node.STATE) then return node
    if problem.isGoalState(startNode['state']):
        return []
    # frontier <- a FIFO stack with node as an element
    frontier = util.Queue()
    frontier.push(startNode)
    # reached <- {problem.INITAL}
    reached = set()
    # while not IS-EMPTY(frontier) do
    while not frontier.isEmpty():
        # node <- POP(frontier)
        node = frontier.pop()
        # if problem.IS-GOAL(node.STATE) then return node action
        if problem.isGoalState(node['state']):
            actions = [] # Convert actions to game.Directions
            actions.append(node['action']) # Convert actions to game.Directions
            return [item for action in actions for item in action]  # Convert actions to game.Directions
        # if node.STATE is not in reached
        if node['state'] not in reached:
            # add node.STATE to reached
            reached.add(node['state']) # In BFS, check for node state in reached before expanding because if node has been reached, there is no need to expand it again
            # for each child in EXPAND(problem, node) do
            successors = problem.getSuccessors(node['state'])
            for successor in successors:
                # child <- CHILD-NODE(state, action, cost)
                child = {'state': successor[0],'action': successor[1], 'cost': successor[2]}
                child['action'] = tuple(node['action']) + (child['action'],) # Concatenate actions
                child['cost'] = node['cost'] + child['cost'] # Add cost
                # add child to frontier
                frontier.push(child)
                # print(frontier.__repr__())
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    # node ← a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
    startNode = {'state': problem.getStartState(), 'action': (), 'cost': 0}
    # frontier ← a priority queue ordered by PATH-COST, with node as the only element
    frontier = util.PriorityQueue()
    frontier.push(startNode, 0)
    # reached <- an empty dict 
    reached = {} # reached dictionary to store state:cost of reached node
    # loop do
    while True:
        # if EMPTY?(frontier) then return failure
        if frontier.isEmpty():
            raise Exception('No solution found!')
        # node<- POP(frontier) /* chooses the lowest-cost node in frontier */
        node = frontier.pop()
        # if node state is not in reached or node cost is less than reached node cost
        if (node['state'] not in reached) or (node['cost'] < reached[node['state']]):
            # add node.STATE to explored
            reached[node['state']] = node['cost']
            print(reached)
            # if problem.GOAL-TEST(node.STATE) then return SOLUTION(node)
            if problem.isGoalState(node['state']):
                actions = []
                actions.append(node['action'])
                return [item for action in actions for item in action]
            else:
                # for each action in problem.ACTIONS(node.STATE) do
                successors = problem.getSuccessors(node['state']) #list of successors with state, action, cost
                for successor in successors:
                    # child <- CHILD-NODE(problem,node,action)
                    child = {'state': successor[0], 'action': successor[1], 'cost': successor[2]}
                    # Update CHILD-NODE(action, cost)
                    child['action'] = tuple(node['action']) + (child['action'],)
                    child['cost'] = node['cost'] + child['cost']
                    # print(child)
                    # replace that frontier node with child with higher PATH-COST
                    frontier.update(child, child['cost'])
    return actions
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    # implementation of A* search algorithm is similar to uniform cost search with Heuristic function
    frontier = util.PriorityQueue()
    reached = []
    startNode = {'state': problem.getStartState(), 'action': (), 'cost': 0}
    frontier.push(startNode, 0)
    while True:
        if frontier.isEmpty():
            raise Exception('No solution found!')
        node = frontier.pop()
        reached.append((node['state'], node['cost']))
        
        if problem.isGoalState(node['state']):
            actions = []
            actions.append(node['action'])
            return [item for action in actions for item in action]
        else:
            successors = problem.getSuccessors(node['state'])
            for successor in successors:
                child = {'state': successor[0], 'action': successor[1], 'cost': successor[2]}
                child['action'] = tuple(node['action']) + (child['action'],)
                child['cost'] = node['cost'] + child['cost']
                alreadyReached = False
                for reachState, reachCost in reached:
                    if child['state'] == reachState and child['cost'] >= reachCost:
                        alreadyReached = True
                if not alreadyReached:
                    frontier.push(child, child['cost'] + heuristic(child['state'], problem))
                    reached.append((child['state'], child['cost']))
    util.raiseNotDefined()


#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
