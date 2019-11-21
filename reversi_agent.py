"""
This module contains agents that play reversi.

Version 3.1
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value

import numpy as np
import gym
import boardgame2 as bg2

_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.
        
        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.
        
        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)    
            p = Process(
                target=self.search,
                args=(
                    self._color, board, valid_actions,
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.
        
        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains 
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for 
            `output_move_row.value` and `output_move_column.value` 
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


class AgentCopy(ReversiAgent):
    """An agent that move randomly."""

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.

        # return move
        try:
            # while True:
            #     pass
            # time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]

            # find better move
            # alpha beta search

            action, value = self.max_value(board, self.player, float('-inf'), float('inf'), 0)  # (state,alpha,beta, depth)
            output_move_row.value = action[0]
            output_move_column.value = action[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def max_value(self, board, player, alpha, beta, depth):
        v = float('-inf')
        c_action = None # Collected Action
        if self.terminal_test(board, player):
            return None, self.utility(board, player)  # return action, utility
        if depth >= 10:
            return None, self.evaluate(board, player)
        for current_action in self.actions(board, player):
            # v = a utility value
            new_board = transition(board, player, current_action)
            new_player = -1 * player
            ignore_action, new_val = self.min_value(new_board, new_player, alpha, beta, depth+1)

            if v < new_val: # new_val = max (at that time)
                v = new_val
                c_action = current_action
            if v >= beta:
                return c_action, v
            alpha = max(alpha, v)
        return c_action, v

    def evaluate(self, board, player):


        


        return self.utility(board, player)

    def min_value(self, board, player, alpha, beta, depth):
        v = float('inf')
        c_action = None  # Collected Action
        if self.terminal_test(board, player):
            return None, self.utility(board, player)  # return action, utility
        if depth >= 10:
            return None, self.evaluate(board, player)
        for current_action in self.actions(board, player):
            # v = a utility value
            new_board = transition(board, player, current_action)
            new_player = -1 * player
            ignore_action, new_val = self.max_value(new_board, new_player, alpha, beta, depth+1)
            if v > new_val:  # new_val = min (at that time)
                v = new_val
                c_action = current_action
            if v <= alpha:
                return c_action, v
            beta = min(beta, v)
        return c_action, v

    def actions(self, board, player):
        # TODO: return list
        # return list
        valids = _ENV.get_valid((board, player))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

    def terminal_test(self, board, player):
        winner = _ENV.get_winner((board, player))
        if winner is not None:
            return True
        else:
            return False

    def utility(self, board, player):
        return np.count_nonzero(board == self.player)