"""
Implement a game strategy for dominoes game using Adversarial search.
"""

import copy
import random
import sys

############################################################
# Section 1: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    board = [[False] * cols for i in xrange(rows)]
    return DominoesGame(board)


class DominoesGame(object):

    leaf = 0
    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.best_move = None

    def get_board(self):
        return self.board

    def reset(self):
        self.board = [[False] * self.cols for i in xrange(self.rows)]

    def is_legal_move(self, row, col, vertical):
        board = self.board
        occupy = ((row, col), (row + 1, col)) if vertical else ((row, col), (row, col + 1))
        # check out-of-bound and not empty
        first = occupy[0]
        second = occupy[1]
        if first[0] < 0 or first[0] >= self.rows or first[1] < 0 or first[1] >= self.cols or board[first[0]][first[1]]:
            return False
        if second[0] < 0 or second[0] >= self.rows or second[1] < 0 or second[1] >= self.cols or board[second[0]][second[1]]:
            return False
        return True

    def legal_moves(self, vertical):
        for row in xrange(self.rows):
            for col in xrange(self.cols):
                if self.is_legal_move(row, col, vertical):
                    yield (row, col)

    def perform_move(self, row, col, vertical):
        # assuming caller will check for legal_move before calling perform_move
        occupy = ((row, col), (row + 1, col)) if vertical else ((row, col), (row, col + 1))
        self.board[occupy[0][0]][occupy[0][1]] = True
        self.board[occupy[1][0]][occupy[1][1]] = True

    def game_over(self, vertical):
        for move in self.legal_moves(vertical):
            if move:
                return False
        return True

    def copy(self):
        return copy.deepcopy(self)

    def successors(self, vertical):
        for move in self.legal_moves(vertical):
            newboard = self.copy()
            newboard.perform_move(move[0], move[1], vertical)
            yield move, newboard

    def get_random_move(self, vertical):
        legal_moves = list(self.legal_moves(vertical))
        return random.choice(legal_moves)

    # Required
    def utility(self, vertical):
        return len(list(self.legal_moves(vertical))) - len(list(self.legal_moves(not vertical)))

    def get_best_move(self, vertical, limit):
        DominoesGame.leaf = 0

        best_val = -sys.maxint
        best_val = self.max_value(-sys.maxint, sys.maxint, vertical, limit)

        return (self.best_move, best_val, DominoesGame.leaf)

    def max_value(self, alpha, beta, vertical, limit):
        if limit == 0 or self.game_over(vertical):
            DominoesGame.leaf += 1
            return self.utility(vertical)

        max_val = -sys.maxint
        for move, successor in self.successors(vertical):
            curr_val = successor.min_value(alpha, beta, not vertical, limit - 1)
            if curr_val > max_val:
                max_val = curr_val
                self.best_move = move

            if max_val >= beta:
                return max_val

            alpha = max(alpha, max_val)
        return max_val

    def min_value(self, alpha, beta, vertical, limit):
        if limit == 0 or self.game_over(vertical):
            DominoesGame.leaf += 1
            return self.utility(not vertical)

        min_val = sys.maxint
        for move, successor in self.successors(vertical):
            curr_val = successor.max_value(alpha, beta, not vertical, limit - 1)
            if curr_val < min_val:
                min_val = curr_val
                self.best_move = move

            if min_val <= alpha:
                return min_val
            beta = min(beta, min_val)
        return min_val
