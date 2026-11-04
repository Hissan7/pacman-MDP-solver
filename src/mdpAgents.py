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

# COMMENT : The skeleton of the agent here was written by Simon Parsons, based on the
# COMMENT : code in pacmanAgents.py. The MDP model and solver below are our own work.

# COMMENT : Approach
# COMMENT : --------
# COMMENT : Every move Pacman builds an MDP of the current world and solves it with
# COMMENT : Value Iteration:
# COMMENT :   States  S : every non-wall grid cell.
# COMMENT :   Actions A : North, South, East, West.
# COMMENT :   P(s'|s,a) : the api motion model (0.8 intended, 0.1 each side, walls mean stay).
# COMMENT :   R(s)      : rebuilt each move from food, capsules and scared ghosts.
# COMMENT :   Discount  : set by the layout's parameter profile (see PROFILES).
# COMMENT : Long-term values come from Value Iteration on the static map. Active ghosts are
# COMMENT : handled by a short finite-horizon MDP over (cell, time): the reward at each time
# COMMENT : is penalised by the predicted chance a ghost is there. The action played is the
# COMMENT : argmax of expected utility at Pacman's cell.

from collections import namedtuple

from pacman import Directions
from game import Agent
import api

# COMMENT : Motion model (matches api.directionProb).
MOVE_PROB = 0.8

# COMMENT : Value Iteration stopping rules.
EPSILON = 0.001
MAX_ITERATIONS = 300

# COMMENT : A layout with ghosts and at most this many open cells counts as a small grid.
SMALL_GRID_CELLS = 40

# COMMENT : Tunable model parameters, one set per kind of layout.
Profile = namedtuple('Profile', [
    'gamma',            # COMMENT : Discount factor.
    'stepReward',       # COMMENT : Cost of every move, encourages short paths.
    'foodReward',       # COMMENT : Value of a food cell.
    'lastFoodReward',   # COMMENT : Value of the final food, so the game is finished.
    'capsuleReward',    # COMMENT : Value of a capsule cell.
    'scaredReward',     # COMMENT : Value of a cell holding a scared (edible) ghost.
    'ghostPenalty',     # COMMENT : Penalty scaled by the chance an active ghost is in a cell.
    'horizon'])         # COMMENT : Moves ahead the ghosts are predicted.

# COMMENT : Values were chosen by random search over many games on fixed seeds, then
# COMMENT : confirmed on fresh seeds. Values that suit a small grid lose badly on a large
# COMMENT : one and vice versa, which is why there are separate profiles.
PROFILES = {
    # COMMENT : No ghosts: far-sighted so distant food still pulls Pacman through the maze.
    'maze': Profile(0.95, -0.04, 5.0, 30.0, 20.0, 50.0, -100.0, 10),
    # COMMENT : Tiny arena with a ghost: short-sighted and urgent.
    'small': Profile(0.7, -1.0, 5.0, 30.0, 20.0, 50.0, -100.0, 10),
    # COMMENT : Large arena with several ghosts: longer view, strong ghost penalty.
    'large': Profile(0.9, -1.0, 5.0, 10.0, 5.0, 0.0, -1000.0, 10),
}

# COMMENT : Unit step for each action, and the two sideways slips for each action.
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
    """COMMENT : Pacman agent that picks moves by solving an MDP with Value Iteration."""

    def __init__(self):
        """COMMENT : Set up empty per-game state."""
        self.resetGameState()

    def resetGameState(self):
        """COMMENT : Forget everything learned about the current game."""
        self.cells = []           # COMMENT : Non-wall cells, indexed by position in this list.
        self.index = {}           # COMMENT : Cell (x, y) -> index in self.cells.
        self.openMoves = []       # COMMENT : Per cell: (action, next index) for open directions.
        self.transitions = []     # COMMENT : Per cell, per action: list of (prob, next index).
        self.utilities = []       # COMMENT : Utility of each cell from the latest solve.
        self.previousGhosts = []  # COMMENT : Cell index of each ghost on the previous move.
        self.profile = PROFILES['large']  # COMMENT : Parameters for this game's layout.

    def registerInitialState(self, state):
        """COMMENT : Build the parts of the MDP that never change (states, transitions)."""
        self.resetGameState()
        self.buildStates(state)
        self.buildTransitions()
        self.chooseProfile(state)
        self.utilities = [0.0] * len(self.cells)

    def final(self, state):
        """COMMENT : Clear per-game state so the next game starts fresh."""
        self.resetGameState()

    def getAction(self, state):
        """COMMENT : Solve the MDP for the current world and return the best legal move."""
        legal = [a for a in api.legalActions(state) if a != Directions.STOP]
        if not legal:
            return api.makeMove(Directions.STOP, legal)
        here = self.index.get(api.whereAmI(state))
        if here is None:  # COMMENT : Defensive: Pacman is off the known grid, so no model.
            return api.makeMove(legal[0], legal)
        rewards = self.buildRewards(state)
        self.solve(rewards)
        values = self.planAhead(rewards, self.activeGhosts(state))
        return api.makeMove(self.bestAction(here, legal, values), legal)

    # COMMENT : ---- Model construction (once per game) ----

    def buildStates(self, state):
        """COMMENT : Collect every non-wall cell inside the grid."""
        width = api.corners(state)[3][0] + 1
        height = api.corners(state)[3][1] + 1
        walls = set(api.walls(state))
        self.cells = [(x, y) for x in range(width) for y in range(height)
                      if (x, y) not in walls]
        self.index = {cell: i for i, cell in enumerate(self.cells)}

    def chooseProfile(self, state):
        """COMMENT : Pick a parameter profile from what the api shows: ghosts present, grid size."""
        if not api.ghosts(state):
            self.profile = PROFILES['maze']
        elif len(self.cells) <= SMALL_GRID_CELLS:
            self.profile = PROFILES['small']
        else:
            self.profile = PROFILES['large']

    def move(self, cell, action):
        """COMMENT : Return the index reached from cell by action, or cell itself if blocked."""
        dx, dy = STEP[action]
        target = (cell[0] + dx, cell[1] + dy)
        return self.index.get(target, self.index[cell])

    def buildTransitions(self):
        """COMMENT : Precompute P(s'|s,a) as (probability, next index) lists."""
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

    # COMMENT : ---- Reward construction (every move) ----

    def ghostCells(self, position):
        """COMMENT : Return indices of the cells a (possibly fractional) ghost position touches."""
        xs = {int(position[0]), int(position[0] + 0.5)}
        ys = {int(position[1]), int(position[1] + 0.5)}
        return [self.index[(x, y)] for x in xs for y in ys if (x, y) in self.index]

    def nearestCell(self, position):
        """COMMENT : Return the index of the cell nearest to a (possibly fractional) position."""
        rounded = (int(position[0] + 0.5), int(position[1] + 0.5))
        floored = (int(position[0]), int(position[1]))
        return self.index.get(rounded, self.index.get(floored))

    def heading(self, previous, current):
        """COMMENT : Return the action a ghost took between two cells, or None if unknown."""
        if previous is None or current is None:
            return None
        for action, j in self.openMoves[previous]:
            if j == current:
                return action
        return None

    def ghostStep(self, cell, heading):
        """COMMENT : Return a ghost's possible next moves as (action, cell) pairs.

        Models the game's ghost rule: ghosts never stop and never turn back,
        except at a dead end. Each option is equally likely.
        """
        options = self.openMoves[cell]
        ahead = [(a, j) for a, j in options if a != REVERSE.get(heading)]
        return ahead if ahead else options

    def ghostOccupancy(self, cell, heading):
        """COMMENT : Return, per time step, the probability of the ghost being in each cell."""
        belief = {(cell, heading): 1.0}
        steps = []
        for _ in range(self.profile.horizon + 1):
            occupancy = {}
            for (i, _), p in belief.items():
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
        """COMMENT : Return (cell, heading) for each non-scared ghost.

        The heading is inferred from the ghost's cell on the previous move.
        """
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
        """COMMENT : Build R(s) from food, capsules and scared ghosts in the current world."""
        rewards = [self.profile.stepReward] * len(self.cells)
        food = [self.index[f] for f in api.food(state) if f in self.index]
        foodReward = (self.profile.lastFoodReward if len(food) == 1
                      else self.profile.foodReward)
        for i in food:
            rewards[i] += foodReward
        for c in api.capsules(state):
            if c in self.index:
                rewards[self.index[c]] += self.profile.capsuleReward
        for position, timer in api.ghostStatesWithTimes(state):
            if timer > 0:
                for i in self.ghostCells(position):
                    rewards[i] += self.profile.scaredReward
        return rewards

    # COMMENT : ---- Solver ----

    def backup(self, i, reward, values):
        """COMMENT : Bellman backup for cell i: reward plus discounted best expected next value."""
        best = max(sum(p * values[j] for p, j in option)
                   for option in self.transitions[i])
        return reward + self.profile.gamma * best

    def solve(self, rewards):
        """COMMENT : Run in-place Value Iteration on the static map, warm-started from last move."""
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
        """COMMENT : Finite-horizon Value Iteration over (cell, time) with ghost collision risk.

        Backs up from the static utilities at the horizon to time 1 and returns the
        utility of each cell one move from now.
        """
        if not ghosts:
            return self.utilities
        occupancy = [self.ghostOccupancy(cell, heading) for cell, heading in ghosts]
        values = list(self.utilities)
        for t in range(self.profile.horizon - 1, 0, -1):
            timed = list(rewards)
            for steps in occupancy:
                for i in set(steps[t - 1]) | set(steps[t]):
                    risk = steps[t - 1].get(i, 0.0) + steps[t].get(i, 0.0)
                    timed[i] += self.profile.ghostPenalty * min(1.0, risk)
            values = [self.backup(i, timed[i], values) for i in range(len(self.cells))]
        return values

    def bestAction(self, here, legal, values):
        """COMMENT : Return the legal action with the highest expected utility from here."""
        best, bestValue = legal[0], None
        for action in legal:
            value = sum(p * values[j]
                        for p, j in self.transitions[here][ACTIONS.index(action)])
            if bestValue is None or value > bestValue:
                best, bestValue = action, value
        return best
