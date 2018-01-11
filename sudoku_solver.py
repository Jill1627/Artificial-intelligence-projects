"""
Implement three different inference algorithm to solve Sudoku games
"""

import Queue
from collections import deque
import copy

############################################################
# Section 1: Sudoku
############################################################

def sudoku_cells():
    return [(i, j) for j in xrange(9) for i in xrange(9)]

def sudoku_arcs():
    arcs = list()
    exist = set()

    # arcs in all rows and cols
    for i in xrange(9):
        for j in xrange(9):
            for d in xrange(9):
                if j != d:
                    arcs.append(((i, j), (i, d)))
                if i != d:
                    arcs.append(((i, j), (d, j)))

    # arcs in all 3 * 3 block
    for i in xrange(3):
        for j in xrange(3):
            for r in xrange(3):
                row = i * 3 + r
                for c in xrange(3):
                    col = j * 3 + c
                    for drow in xrange(1, 3):
                        row_neigh = i * 3 + (row + drow) % 3
                        for dcol in xrange(1, 3):
                            col_neigh = j * 3 + (col + dcol) % 3
                            tup = ((row, col), (row_neigh, col_neigh))
                            arcs.append(tup)
    return arcs


def read_board(path):
    with open(path) as sudoku_file:
        matrix = [list(line.rstrip('\n').rstrip('\r')) for line in sudoku_file]

    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    board = dict()
    for i in xrange(len(matrix)):
        for j in xrange(len(matrix[0])):
            if matrix[i][j] != '*':
                board[(i, j)] = set([int(matrix[i][j])])
            else:
                board[(i, j)] = set(candidates[:])
    return board

class Sudoku(object):

    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()

    def __init__(self, board):
        self.board = board

    def get_values(self, cell):
        return self.board[cell]

    def remove_inconsistent_values(self, cell1, cell2):
        cell1_candidates = self.board[cell1]
        cell2_candidates = self.board[cell2]

        if len(cell2_candidates) != 1:
            return False

        # Assume! prior to calling this method, a check of existing arc between cell1 and cell2 has been done
        conflict = next(iter(self.board[cell2]))
        if conflict in cell1_candidates:
            self.board[cell1].remove(conflict)
            return True
        return False

    def get_neighbors(self, cell1, cell2):
        row = cell1[0]
        col = cell1[1]
        # same col
        for i in xrange(9):
            if i not in xrange((row / 3) * 3,((row / 3) * 3) + 3):
                if (i, col) != cell2:
                    yield ((i, col), (row, col))
        # same row
        for j in xrange(9):
            if j not in xrange((col / 3) * 3,((col / 3) * 3) + 3):
                if (row, j) != cell2:
                    yield ((row, j), (row, col))
        # same block
        for i in range((row / 3) * 3,((row / 3) * 3) + 3):
            for j in range((col / 3) * 3,((col / 3) * 3) + 3):
                if i != row or j != col:
                    if (i, j) != cell2:
                        yield ((i, j), (row, col))

    def print_board(self):
        for i in xrange(9):
            prt = list()
            for j in xrange(9):
                prt.extend(list(self.board[(i, j)]))
            print prt

    def in_row(self, cell, val):
        for j in xrange(9):
            if j != cell[1]:
                if val in self.get_values((cell[0], j)):
                    return True
        return False

    def in_col(self, cell, val):
        for i in xrange(9):
            if i != cell[0]:
                if val in self.get_values((i, cell[1])):
                    return True
        return False

    def in_block(self, cell, val):
        row_base = cell[0] / 3 * 3
        col_base = cell[1] / 3 * 3
        for i in xrange(row_base, row_base + 3):
            for j in xrange(col_base, col_base + 3):
                if i != cell[0] or j != cell[1]:
                    if val in self.get_values((i, j)):
                        return True
        return False

    def infer_ac3(self):
        # assuming all input boards are solvable
        allarcs = Sudoku.ARCS[:]
        q = Queue.Queue()
        q.queue = deque(allarcs)

        while not q.empty():
            arc = q.get()
            if self.is_solved():
                return True
            if self.remove_inconsistent_values(arc[0], arc[1]):
                if len(self.board[arc[0]]) == 0:
                    return False

                for neighbor in self.get_neighbors(arc[0], arc[1]):
                    q.put(neighbor)
        return True

    def is_solved(self):
        for cell in Sudoku.CELLS:
            if len(self.board[cell]) != 1:
                return False
        return True

    def successors(self, cell):
        candidates = list(self.get_values(cell))
        for val in candidates:
            succ = copy.deepcopy(self)
            succ.board[cell] = {val}
            yield succ

    def pick_cell(self):
        # arbitrarily choose any unassigned cell, improve to use heuristics later if necessary
        for cell in Sudoku.CELLS:
            if len(self.get_values(cell)) > 1:
                return cell
        return None

    def solvable(self):
        for cell in Sudoku.CELLS:
            if len(self.get_values(cell)) == 0:
                return False
        return True

    def infer_improved(self):
        unfinished = True
        while unfinished:
            unfinished = False
            self.infer_ac3()
            if self.is_solved():
                return True
            for i in xrange(9):
                for j in xrange(9):
                    cell = (i, j)
                    candidates = self.get_values(cell)
                    if len(candidates) > 1:
                        for val in candidates:
                            if not self.in_row(cell, val):
                                self.board[cell] = {val}
                                unfinished = True
                            if not self.in_col(cell, val):
                                self.board[cell] = {val}
                                unfinished = True
                            if not self.in_block(cell, val):
                                self.board[cell] = {val}
                                unfinished = True
        return False

    def infer_with_guessing(self):
        self.infer_improved()
        if self.is_solved():
            return True
        if not self.solvable():
            return False
        cell = self.pick_cell()
        if cell:
            for succ in self.successors(cell):
                if succ.infer_with_guessing():
                    self.board = succ.board
                    return True
        return False
