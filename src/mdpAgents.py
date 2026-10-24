# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The skeleton of the agent here was written by Simon Parsons, based on the
# code in pacmanAgents.py. The MDP model and solver below are our own work.

# Approach
# --------
# Every move Pacman builds an MDP of the current world and solves it with
# Value Iteration:
#   States  S : every non-wall grid cell.
#   Actions A : North, South, East, West.
#   P(s'|s,a) : the api motion model (0.8 intended, 0.1 each side, walls mean stay).
#   R(s)      : rebuilt each move from food, capsules and scared ghosts.
#   Discount  : GAMMA.
# Long-term values come from Value Iteration on the static map. Active ghosts are
# handled by a short finite-horizon MDP over (cell, time): the reward at each time
# is penalised by the predicted chance a ghost is there. The action played is the
# argmax of expected utility at Pacman's cell.

from collections import deque

from pacman import Directions
from game import Agent
import api

# Motion model (matches api.directionProb).
MOVE_PROB = 0.8

# Discount factor and step cost when ghosts are present: short-sighted and urgent.
GAMMA = 0.7
STEP_REWARD = -1.0

# Same two values for ghost-free mazes: far-sighted so distant food still pulls Pacman.
MAZE_GAMMA = 0.95
MAZE_STEP_REWARD = -0.04

# Value Iteration stopping rules.
EPSILON = 0.001
MAX_ITERATIONS = 300

# Rewards.
FOOD_REWARD = 5.0            # Value of a food cell.
LAST_FOOD_REWARD = 30.0      # Value of the final food, so the game is finished.
CAPSULE_REWARD = 20.0        # Value of a capsule cell.
GHOST_PENALTY = -100.0       # Penalty scaled by the chance an active ghost is in a cell.
SCARED_GHOST_REWARD = 50.0   # Value of a cell holding a scared (edible) ghost.

# Ghost prediction: the finite-horizon MDP looks this many moves ahead.
GHOST_HORIZON = 10

# Unit step for each action, and the two sideways slips for each action.
STEP = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}
SLIPS = {Directions.NORTH: (Directions.WEST, Directions.EAST),
         Directions.SOUTH: (Directions.EAST, Directions.WEST),
         Directions.EAST: (Directions.NORTH, Directions.SOUTH),
         Directions.WEST: (Directions.SOUTH, Directions.NORTH)}
REVERSE = {Directions.NORTH: Directions.SOUTH, Directions.SOUTH: Directions.NORTH,
           Directions.EAST: Directions.WEST, Directions.WEST: Directions.EAST}
ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]


class MDPAgent(Agent):
    """Pacman agent that picks moves by solving an MDP with Value Iteration."""

    def __init__(self):
        """Set up empty per-game state."""
        self.resetGameState()

    def resetGameState(self):
        """Forget everything learned about the current game."""
        self.cells = []          # Non-wall cells, indexed by position in this list.
        self.index = {}          # Cell (x, y) -> index in self.cells.
        self.openMoves = []      # Per cell: list of (action, next index) for open directions.
        self.previousGhosts = [] # Cell index of each ghost on the previous move.
        self.gamma = GAMMA       # Discount factor for this game.
        self.stepReward = STEP_REWARD  # Cost of each move for this game.
        self.transitions = []    # Per cell, per action: list of (prob, next index).
        self.utilities = []      # Utility of each cell from the latest solve.

    def registerInitialState(self, state):
        """Build the parts of the MDP that never change (states, transitions)."""
        self.resetGameState()
        self.buildStates(state)
        self.buildTransitions()
        self.chooseProfile(state)
        self.utilities = [0.0] * len(self.cells)

    def final(self, state):
        """Clear per-game state so the next game starts fresh."""
        self.resetGameState()

    def getAction(self, state):
        """Solve the MDP for the current world and return the best legal move."""
        legal = [a for a in api.legalActions(state) if a != Directions.STOP]
        if not legal:
            return api.makeMove(Directions.STOP, legal)
        here = self.index.get(api.whereAmI(state))
        if here is None:
            return api.makeMove(legal[0], legal)
        rewards = self.buildRewards(state)
        self.solve(rewards)
        values = self.planAhead(rewards, self.activeGhosts(state))
        return api.makeMove(self.bestAction(here, legal, values), legal)

    # ---- Model construction (once per game) ----

    def buildStates(self, state):
        """Collect every non-wall cell inside the grid."""
        width = api.corners(state)[3][0] + 1
        height = api.corners(state)[3][1] + 1
        walls = set(api.walls(state))
        self.cells = [(x, y) for x in range(width) for y in range(height)
                      if (x, y) not in walls]
        self.index = {cell: i for i, cell in enumerate(self.cells)}

    def chooseProfile(self, state):
        """Pick discount and step cost: a layout with no ghosts is a pure path-finding maze."""
        if not api.ghosts(state):
            self.gamma, self.stepReward = MAZE_GAMMA, MAZE_STEP_REWARD

    def move(self, cell, action):
        """Return the index reached from cell by action, or cell itself if blocked."""
        dx, dy = STEP[action]
        target = (cell[0] + dx, cell[1] + dy)
        return self.index.get(target, self.index[cell])

    def buildTransitions(self):
        """Precompute P(s'|s,a) as (probability, next index) lists."""
        slip = (1.0 - MOVE_PROB) / 2.0
        self.transitions = []
        self.openMoves = []
        for cell in self.cells:
            options = []
            for action in ACTIONS:
                left, right = SLIPS[action]
                options.append([(MOVE_PROB, self.move(cell, action)),
                                (slip, self.move(cell, left)),
                                (slip, self.move(cell, right))])
            self.transitions.append(options)
            opened = [(a, self.move(cell, a)) for a in ACTIONS]
            self.openMoves.append([(a, j) for a, j in opened if j != self.index[cell]])

    # ---- Reward construction (every move) ----

    def ghostCells(self, position):
        """Return indices of the cells a (possibly fractional) ghost position touches."""
        xs = {int(position[0]), int(position[0] + 0.5)}
        ys = {int(position[1]), int(position[1] + 0.5)}
        return [self.index[(x, y)] for x in xs for y in ys if (x, y) in self.index]

    def nearestCell(self, position):
        """Return the index of the cell nearest to a (possibly fractional) position."""
        rounded = (int(position[0] + 0.5), int(position[1] + 0.5))
        floored = (int(position[0]), int(position[1]))
        return self.index.get(rounded, self.index.get(floored))

    def heading(self, previous, current):
        """Return the action a ghost took between two cells, or None if unknown."""
        if previous is None or current is None:
            return None
        for action, j in self.openMoves[previous]:
            if j == current:
                return action
        return None

    def ghostStep(self, cell, heading):
        """Return the ghost's next (cell, heading) options: no stopping, no turning back."""
        options = self.openMoves[cell]
        ahead = [(a, j) for a, j in options if a != REVERSE.get(heading)]
        return ahead if ahead else options

    def ghostOccupancy(self, cell, heading):
        """Return, per time step, the probability of the ghost being in each cell."""
        belief = {(cell, heading): 1.0}
        steps = []
        for _ in range(GHOST_HORIZON + 1):
            occupancy = {}
            for (i, h), p in belief.items():
                occupancy[i] = occupancy.get(i, 0.0) + p
            steps.append(occupancy)
            nextBelief = {}
            for (i, h), p in belief.items():
                options = self.ghostStep(i, h)
                for a, j in options:
                    nextBelief[(j, a)] = nextBelief.get((j, a), 0.0) + p / len(options)
            belief = nextBelief
        return steps

    def activeGhosts(self, state):
        """Return (cell, heading) for each non-scared ghost; heading comes from the last move."""
        ghosts, current = [], []
        for n, (position, timer) in enumerate(api.ghostStatesWithTimes(state)):
            cell = self.nearestCell(position)
            current.append(cell)
            if timer <= 0 and cell is not None:
                before = self.previousGhosts[n] if n < len(self.previousGhosts) else None
                ghosts.append((cell, self.heading(before, cell)))
        self.previousGhosts = current
        return ghosts

    def buildRewards(self, state):
        """Build R(s) from food, capsules and scared ghosts in the current world."""
        rewards = [self.stepReward] * len(self.cells)
        food = [self.index[f] for f in api.food(state) if f in self.index]
        foodReward = LAST_FOOD_REWARD if len(food) == 1 else FOOD_REWARD
        for i in food:
            rewards[i] += foodReward
        for c in api.capsules(state):
            if c in self.index:
                rewards[self.index[c]] += CAPSULE_REWARD
        for position, timer in api.ghostStatesWithTimes(state):
            if timer > 0:
                for i in self.ghostCells(position):
                    rewards[i] += SCARED_GHOST_REWARD
        return rewards

    # ---- Solver ----

    def backup(self, i, reward, values):
        """Bellman backup for cell i: reward plus discounted best expected next value."""
        best = max(sum(p * values[j] for p, j in option)
                   for option in self.transitions[i])
        return reward + self.gamma * best

    def solve(self, rewards):
        """Run in-place Value Iteration on the static map, warm-started from last move."""
        utilities = self.utilities
        for _ in range(MAX_ITERATIONS):
            change = 0.0
            for i in range(len(self.cells)):
                updated = self.backup(i, rewards[i], utilities)
                change = max(change, abs(updated - utilities[i]))
                utilities[i] = updated
            if change < EPSILON:
                break

    def planAhead(self, rewards, ghosts):
        """Finite-horizon Value Iteration over (cell, time) with ghost collision risk.

        Backs up from the static utilities at the horizon to time 1 and returns the
        utility of each cell one move from now.
        """
        if not ghosts:
            return self.utilities
        occupancy = [self.ghostOccupancy(cell, heading) for cell, heading in ghosts]
        values = list(self.utilities)
        for t in range(GHOST_HORIZON - 1, 0, -1):
            timed = list(rewards)
            for steps in occupancy:
                for i in set(steps[t - 1]) | set(steps[t]):
                    risk = steps[t - 1].get(i, 0.0) + steps[t].get(i, 0.0)
                    timed[i] += GHOST_PENALTY * min(1.0, risk)
            values = [self.backup(i, timed[i], values) for i in range(len(self.cells))]
        return values

    def bestAction(self, here, legal, values):
        """Return the legal action with the highest expected utility from here."""
        best, bestValue = legal[0], None
        for action in legal:
            value = sum(p * values[j]
                        for p, j in self.transitions[here][ACTIONS.index(action)])
            if bestValue is None or value > bestValue:
                best, bestValue = action, value
        return best
