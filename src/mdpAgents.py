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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

# PLAGIARISM STATEMENT : I declare that this script was fully written by myself and no external
# sources of code were used. I seek full credit of this work. 

from collections import namedtuple

from pacman import Directions
from game import Agent
import api

# Motion model 
MOVE_PROB = 0.8

# Value Iteration stopping rules.
EPSILON = 0.001
MAX_ITERATIONS = 300

# A layout with ghosts and at most this many open cells counts as a small grid.
SMALL_GRID_CELLS = 40

# Tunable model parameters
Profile = namedtuple('Profile', [
    'gamma',            
    'stepReward',       
    'foodReward',       
    'lastFoodReward',   
    'capsuleReward',    
    'scaredReward',     
    'ghostPenalty',     
    'horizon'])         


PROFILES = {
    # No ghosts
    'maze': Profile(0.95, -0.04, 5.0, 30.0, 20.0, 50.0, -100.0, 10),
    # Tiny arena with a ghost
    'small': Profile(0.7, -1.0, 5.0, 30.0, 20.0, 50.0, -100.0, 10),
    # Large arena with several ghosts
    'large': Profile(0.9, -1.0, 5.0, 10.0, 5.0, 0.0, -1000.0, 10),
}

# Unit step for each action & the two sideways slips for each action.
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
    #Pacman agent that picks moves by solving an MDP with Value Iteration

    # Set up empty per-game state
    def __init__(self):
        self.resetGameState()

    def resetGameState(self):
        
        self.cells = []          
        self.index = {}           
        self.openMoves = []       
        self.transitions = []     
        self.utilities = []       
        self.previousGhosts = []  
        self.profile = PROFILES['large']  

    # Build the parts of the MDP that are constant
    def registerInitialState(self, state):
        self.resetGameState()
        self.buildStates(state)
        self.buildTransitions()
        self.chooseProfile(state)
        self.utilities = [0.0] * len(self.cells)

    # Clear game state so next game starts fresh
    def final(self, state):
        self.resetGameState()

    # Solve the MDP for the current world & give the best move
    def getAction(self, state):
        legal = [a for a in api.legalActions(state) if a != Directions.STOP]
        if not legal:
            return api.makeMove(Directions.STOP, legal)
        here = self.index.get(api.whereAmI(state))
        if here is None:  # Pacman is off the grid so no model.
            return api.makeMove(legal[0], legal)
        rewards = self.buildRewards(state)
        self.solve(rewards)
        values = self.planAhead(rewards, self.activeGhosts(state))
        return api.makeMove(self.bestAction(here, legal, values), legal)

    # Collect every cell in the grid that isnt a wall 
    def buildStates(self, state):
        width = api.corners(state)[3][0] + 1
        height = api.corners(state)[3][1] + 1
        walls = set(api.walls(state))
        self.cells = [(x, y) for x in range(width) for y in range(height)
                      if (x, y) not in walls]
        self.index = {cell: i for i, cell in enumerate(self.cells)}

    def chooseProfile(self, state):
        if not api.ghosts(state):
            self.profile = PROFILES['maze']
        elif len(self.cells) <= SMALL_GRID_CELLS:
            self.profile = PROFILES['small']
        else:
            self.profile = PROFILES['large']

    # Return index reached from cell by action
    def move(self, cell, action):
        dx, dy = STEP[action]
        target = (cell[0] + dx, cell[1] + dy)
        return self.index.get(target, self.index[cell])

    # Precompute P(s'|s,a)
    def buildTransitions(self):
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

    # Return indices of the cells a ghost position touches
    def ghostCells(self, position):
        xs = {int(position[0]), int(position[0] + 0.5)}
        ys = {int(position[1]), int(position[1] + 0.5)}
        return [self.index[(x, y)] for x in xs for y in ys if (x, y) in self.index]

    # Return index of cell nearest to a position
    def nearestCell(self, position):
        rounded = (int(position[0] + 0.5), int(position[1] + 0.5))
        floored = (int(position[0]), int(position[1]))
        return self.index.get(rounded, self.index.get(floored))

    # Return the action a ghost took between two cells else None if unknown.
    def heading(self, previous, current):
        if previous is None or current is None:
            return None
        for action, j in self.openMoves[previous]:
            if j == current:
                return action
        return None

    # Return a ghost's possible next moves as action, cell pairs.
    def ghostStep(self, cell, heading):
        options = self.openMoves[cell]
        ahead = [(a, j) for a, j in options if a != REVERSE.get(heading)]
        return ahead if ahead else options

    def ghostOccupancy(self, cell, heading):
        # Return P(ghost being in each cell)
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

    # Return cell & heading for each non scared ghost.
    def activeGhosts(self, state):
        ghosts, current = [], []
        for n, (position, timer) in enumerate(api.ghostStatesWithTimes(state)):
            cell = self.nearestCell(position)
            current.append(cell)
            if timer <= 0 and cell is not None:
                before = self.previousGhosts[n] if n < len(self.previousGhosts) else None
                ghosts.append((cell, self.heading(before, cell)))
        self.previousGhosts = current
        return ghosts

    # Build R(s) from food, capsules and scared ghosts in the current world
    def buildRewards(self, state):
        rewards = [self.profile.stepReward] * len(self.cells)
        food = [self.index[f] for f in api.food(state) if f in self.index]
        foodReward = (self.profile.lastFoodReward if len(food) == 1
                      else self.profile.foodReward)
        for i in food:
            rewards[i] = rewards[i] + foodReward
        for c in api.capsules(state):
            if c in self.index:
                rewards[self.index[c]] = rewards[self.index[c]] + self.profile.capsuleReward
        for position, timer in api.ghostStatesWithTimes(state):
            if timer > 0:
                for i in self.ghostCells(position):
                    rewards[i] = rewards[i] + self.profile.scaredReward
        return rewards

    # Bellman for cell i (reward + discounted best expected next value)
    def backup(self, i, reward, values):
        best = max(sum(p * values[j] for p, j in option)
                   for option in self.transitions[i])
        return reward + self.profile.gamma * best

    # Run in-place Value Iteration on the static map
    def solve(self, rewards):
        utilities = self.utilities
        for _ in range(MAX_ITERATIONS):
            change = 0.0
            for i in range(len(self.cells)):
                updated = self.backup(i, rewards[i], utilities)
                change = max(change, abs(updated - utilities[i]))
                utilities[i] = updated
            if change < EPSILON:
                break
    
    # Finite-horizon Value Iteration over (cell, time) with ghost collision risk.
    def planAhead(self, rewards, ghosts):
        if not ghosts:
            return self.utilities
        occupancy = [self.ghostOccupancy(cell, heading) for cell, heading in ghosts]
        values = list(self.utilities)
        for t in range(self.profile.horizon - 1, 0, -1):
            timed = list(rewards)
            for steps in occupancy:
                for i in set(steps[t - 1]) | set(steps[t]):
                    risk = steps[t - 1].get(i, 0.0) + steps[t].get(i, 0.0)
                    timed[i] = timed[i] + self.profile.ghostPenalty * min(1.0, risk)
            values = [self.backup(i, timed[i], values) for i in range(len(self.cells))]
        return values

    # Return the legal action with the highest expected utility
    def bestAction(self, here, legal, values):
        best, bestValue = legal[0], None
        for action in legal:
            value = sum(p * values[j]
                        for p, j in self.transitions[here][ACTIONS.index(action)])
            if bestValue is None or value > bestValue:
                best, bestValue = action, value
        return best
