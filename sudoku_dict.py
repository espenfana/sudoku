'dict based sudoku solver'
# A (possibly) improved version of the two-algortihm sudoku solver, better utilizing 
# the dictionary and set classes in python. 

# Instead of manipulating an array of the board itself, in this version we're working from 
# a dictionary where the key is the cell coordinate and the value is a set of the valid numbers 
# that can go into the cell. This dictionary is updated every time a number is inserted into a
# cell. This should eliminate a lot of repetetive tasks. The algorithm is aided by lists of 
# coordinates representing the different blocks, columns and rows (thus a block is functually no
# different from a row/column)

# Algorithms (in increasing number of operations): 
#  Insertion algorithms (finds non-ambigous values to insert into cells):
#     only_val - find a cell where only one value is valid (len(set) == 1)
#     only_cell - find a value which has only one legal coordinate within a shape (block/row/col)
#  Option reducing algorithms (modifies the dictionary of legal values for each cell)
#  [x]val_iso - for one value in a series, if the set of legal cells are also within a different series,
#        this value can't be in any other cells in that series
#  [x] cell_union - for a set of values (within a shape): if the UNION of legal cells is the same length as the 
#        set of values, no other values can be legal in these cells
#  [x] val_union - if a set of cells (within a shape) share a set of values (of the same len), these 
#        values can't be legal for any other cells within that shape
#     brute - save the current state and insert a value in a cell. If the recursor returns 
#        without the board being solved the guess must have been wrong

# v1 - functional algorithms, brute not implemented, list of insertions not implemented
#  Success - This is a lot (at least 10x) faster than the array-based method. Needs brute to solve hard boards

# v2 - in order to solve the hardest boards, the two basic algorithms are not enough
#  [x] implement a brute guessing algorithm
#     [x] add error detection to algorithms
#  [x] allow algorithms to return multiple insertions
#  [x] add recipe

# v3 - polish and generalization
#  [x] convert solver into a class
#  [x] lose numpy dependency
#  [x] generalize to include non block boards
#  [x] make the board value agnostic (i.e. letters can replace numbers)
#     [x] Cell values, including the null value, can be any hashable object
#  [x] algorithm selection in solver creation

# future
# gui
# [x] sudoku generator


#import numpy as np
import random
import math
import sys
sys.path.append("/home/ef/code/py")

class DictSolver():
   'Container object for sudoku board and solver'
   def __init__(self, arrin, meta=None, blshape=None, algorithm='default', silent=False):
      '''
      Create sudoku board object from a block array. Input can be a list of lists or a string'
      arrin - List of lists or string with the board values
      meta - Any information to be stored in the solver
      blshape - (x,y) tuple with the shape of the "blocks"
      algorithm - Option to remove or change order of algoritms
      '''
      # Parsing the input "array"

      if (type(arrin) == str) or (type(arrin) == list and type(arrin[0]) != list):
         if blshape:
            if type(blshape) == tuple and len(blshape) == 2:
               self.blshape = blshape
               self.bsize = blshape[0] * blshape[1]
               if len(arrin) != self.bsize ** 2:
                  raise Exception(f'Length of input string ({len(arrin)}) does not match the given blshape([{blshape})')
            else:
               raise Exception('The block shape (blshape) must be a tuple of length 2')
         else:
            bsize = len(arrin) ** 0.5 # Assuming a square board
            if bsize == 9.0 or bsize == 16.0:
               self.bsize = int(bsize)
               self.blshape = (int(self.bsize ** 0.5),) * 2
            else:
               raise Exception(f'Input array length of {len(arrin)} not recognized')
         self.arr = [[arrin[j + i*self.bsize] for j in range(self.bsize)] for i in range(self.bsize)]
      
      elif type(arrin) == list:
         if blshape:
            if type(blshape) == tuple and len(blshape) == 2:
               self.blshape = blshape
               self.bsize = blshape[0] * blshape[1]
               if len(arrin) != self.bsize:
                  raise Exception(f'Length of input string ({len(arrin)}) does not match the given blshape([{blshape})')
            else:
               raise Exception('The block shape (blshape) must be a tuple of length 2')
         else:
            self.bsize = len(arrin)
            self.blshape = (int(self.bsize ** 0.5),) * 2
         if len(set(len(row) for row in arrin)) == 1:
            self.arr = [row.copy() for row in arrin]
         else:
            raise Exception(f'Length of rows are not even! {[len(row) for row in arrin]}')
      else:
         raise Exception(f'Input array type of {type(arrin)} is not valid')
      
      if not all(len(row) == self.bsize for row in self.arr):
         print('Warning, the input board is not square')
            
      # Set which algorithms should be used and in which order
      self.set_algorithms(algorithm)
      self.brute = True
      # Set inital values and useful sets
      self.meta = meta
      self.solved = False
      self.error = False
      self.silent = silent
      self.recipe = []
      self.all_values = set()
      flat = [c for row in self.arr for c in row]
      self.null = max(set(flat), key=flat.count) # The 'empty cell' value
      self.all_values = set(flat) - {self.null}
      if len(self.all_values) != self.bsize:
         if len(self.all_values) == self.bsize - 1: # Possible but unlikely for a value not to be in the starting board
            print('init warning: Mismatch between all_values and board size trying to add a value')
            self.all_values.add('XX')
         else:
            print('init warning: Mismatch between all_values and board size.')
      
      # Generate coordinates for the different shapes in the board
      self.coord_rows = tuple(tuple((i,j) for j in range(self.bsize)) for i in range(self.bsize))
      self.coord_cols = tuple(tuple((i,j) for i in range(self.bsize)) for j in range(self.bsize))
      self.coord_blocks = self.generate_coord_blocks()


   def __repr__(self):
      return self.sprint()

   
   def set_algorithms(self, algorithm='def'):
      'Set the algorithms to be used'
      alg_dict = {'brute': (),
                  'ins': (self.only_val, self.only_cell),
                  'lite': (self.only_val, self.only_cell, self.val_iso),
                  'full': (self.only_val, self.only_cell, self.val_iso, self.cell_union, self.val_union),
                  'opt': (self.only_val, self.only_cell, self.val_iso, self.val_union)}
      self.algorithms = alg_dict.get(algorithm, (self.only_val, self.only_cell))


   def copyarr(self):
      'Return an independent copy of the board array'
      return [row.copy() for row in self.arr]


   def blockrowcol(self, i, j) -> tuple: 
      'Return the coordinates for all cells sharing the block, row and col of cell (i,j)'
      for s in self.coord_blocks:
         if (i,j) in s:
            return self.coord_rows[i] + self.coord_cols[j] + s


   def generate_coord_blocks(self):
      'Returns a tuple of coordinates belonging to each block'
      blocks = []
      (ic, jc) = self.blshape
      for n in range(self.bsize):
         j_gen = range( (n%ic) * jc , (n%ic + 1) * jc)
         i_gen = range( math.floor(n/ic) * ic, (math.floor(n/ic) + 1) * ic)
         blocks.append(tuple((i, j) for i in i_gen for j in j_gen))
      return tuple(blocks)


   def sprint(self):
      'Return a prettified string of the sudoku board'
      bsize = self.bsize
      char_width = max(len(str(val)) for val in self.all_values)
      hline = ' ' + '-' * ((char_width +1) * self.bsize) + '--' * (self.blshape[0] - 1) + '- '
      printout = hline
      for i in range(bsize):
         printout += '\n| '
         for j in range(bsize):
            printout += f'{self.arr[i][j]:>{char_width}}' + ' '
            if not (j + 1) % self.blshape[1]:
               printout += '| '
         if not (i + 1) % self.blshape[0] :
            printout += '\n' + hline
      return printout


   def solve(self):
      'Solve the sudoku board'
      if not self.silent: 
         print(self)
      self.build_option_dictionary()
      self.ops = 0
      self.solver_recurser()
      if not self.silent:
         if self.solved:
            print('Board solved!')
         else:
            print('Board could not be solved')
         print(self)


   def solver_recurser(self):
      '''Evolve the solver to the next step'''
      if len(self.opt_dict) == 0 or self.solved == True:
         self.solved = True
         return
      for algorithm in self.algorithms:
         didsomething, ins = algorithm()
         if self.error:
            return
         if didsomething:
            self.ops += 1
            if len(ins):
               for (i, j), val in ins.items():
                  self.recipe.append('{:2d} - ({:2d},{:2d}) : {:>2} ({})'.format(self.ops, i, j, val, algorithm.__name__))
                  self.arr[i][j] = val
                  self.update_option_dictionary((i,j), val)
            self.solver_recurser()
            return
      if self.brute:
         # If no solution was found, we must make a guess:
         # The brute solver should be enough to solve a board by itself, however inefficiently.
         # With threshold = 1 it basically performs the role of the singe num algorithm (but it 
         # will never be True if the only_val algorithm is active)
         try:
            import copy
            save_arr = self.copyarr() # List of copies because lists are mutable
            save_recipe = self.recipe.copy() # Shallow copy is fine because strings are immutable
            threshold = 1 # It is most efficient to guess in a cell with the least alternatives
            while (len(self.opt_dict) > 0) & (threshold < self.bsize):
               for (i,j), cell_set in self.opt_dict.items():
                  if len(cell_set) == threshold:
                     copy_set = random.sample(cell_set, threshold)
                     #copy_set = cell_set.copy()
                     self.ops +=1
                     for val in copy_set:
                        self.recipe.append('{:2} - ({:2d},{:2d}) : {:>2} (guess)'.format(self.ops, i, j, val))
                        self.arr[i][j] = val
                        self.update_option_dictionary((i,j), val)
                        if not self.error:
                           self.solver_recurser()
                           if self.solved: return
                        # board.error is now implicit, otherwize the board would be solved
                        self.arr = [row.copy() for row in save_arr] # Undo and rebuild
                        #self.arr = save_arr # would occasionally produce errors on large boards
                        self.build_option_dictionary()
                        self.recipe = save_recipe
                        self.error = False
                     return # No solution, must be an earlier error
               threshold +=1
         except KeyError as k:
            print(self)
            print(save_arr)
            print(k, val, copy_set)
            raise Exception
      return


   def build_option_dictionary(self) -> dict:
      'initialize the dictionary of cell coordinates and legal numbers'
      self.opt_dict = {}
      for i in range(self.bsize):
         for j in range(self.bsize):
            if self.arr[i][j] == self.null:
               nums = set(self.arr[k][l] for (k,l) in self.blockrowcol(i,j))
               self.opt_dict[(i,j)] = self.all_values - nums


   def update_option_dictionary(self, coord, val) -> None:
      self.opt_dict.pop(coord)
      if len(self.opt_dict) == 0:
         self.solved = True
         return
      for cell in self.blockrowcol(*coord):
         if cell in self.opt_dict:
            self.opt_dict[cell].discard(val) # .discard removes the item if it is in the set
            if len(self.opt_dict[cell]) == 0: # Error detection
               self.error = True


   def only_val(self) -> tuple:
      '''Find a cell where only one number is legal
         Returns bool, {coord: value,...}'''
      # With the legal dictionary, this algorithm becomes pretty trivial
      ins = {}
      didsomething = False
      for cell, cell_set in self.opt_dict.items():
         if len(cell_set) == 1:
            ins[cell] = cell_set.copy().pop()
            didsomething = True
      return didsomething, ins


   def only_cell(self) -> tuple:
      '''Find a number for which only one cell is legal
         Returns bool, {coord: value,...}'''
      ins = {}
      didsomething = False
      scr = self.coord_blocks + self.coord_rows + self.coord_cols
      for series in scr:
         series = {c:self.opt_dict[c] for c in series if c in self.opt_dict}
         for cell, cell_set in series.items():
            not_set = set()
            other_cells = {k:v for k,v in series.items() if k != cell}
            for other_cell_set in other_cells.values():
               not_set.update(other_cell_set)
            the_set = cell_set - not_set
            if len(the_set) > 1: # Meaning an error has been made
               self.error = True 
               return False, ins
            if len(the_set) ==  1:
               ins[cell] = the_set.pop()
               didsomething = True
      return didsomething, ins


   def val_iso(self) -> tuple:
      '''Manipulate the opt_dict using the value isolaton algorithm.
         Returns True, [] if it did anything'''
      #   for one value in a series, if the set of legal cells are also within a different series,
      #   this value can't be in any other cells in that series
      didsomething = False
      for series in (self.coord_blocks + self.coord_rows + self.coord_cols):
         cells = [cell for cell in series if cell in self.opt_dict]
         val_set = set().union(*(self.opt_dict[cell] for cell in cells))
         for val in val_set:
            cell_set = set(cell for cell in cells if val in self.opt_dict[cell])
            if len(cell_set) <= 1:
               continue # This shouldn't happen if only_num is active 
            for other_series in (self.coord_blocks + self.coord_rows + self.coord_cols):
               other_cells = [cell for cell in other_series if cell in self.opt_dict]
               if other_cells == cells:
                  continue
               if all((cell in other_cells) for cell in cell_set):
                  for cell in other_cells:
                     if cell in cell_set:
                        continue
                     if val in self.opt_dict[cell]:
                        didsomething = True
                        self.opt_dict[cell].remove(val)
                        self.recipe.append(f'{self.ops +1:2} - ({cell[0]:2},{cell[1]:2}) : -{val} (val_iso ({cell_set})')
      return didsomething, []


   def cell_set(self) -> tuple: # OBSOLETE replaced by cell_union
      '''Modifies the options dictionary. Returns True if it did anything '''
#        if a set of values (within a series) share a set of legal cells (of the same len),
#        no other values can be legal in these cells
      from itertools import combinations
      didsomething = False
      sizes = range(2, 5)
      for size in sizes:
         for series in (self.coord_blocks + self.coord_rows + self.coord_cols):
            cells = [cell for cell in series if cell in self.opt_dict]
            if len(cells) < size:
               continue
            val_set = set().union(*(self.opt_dict[cell] for cell in cells)) 
            val_comb = combinations(val_set, size)
            for val_group in val_comb:
               c_sets = []
               for val in val_group:
                  c_sets.append(set(cell for cell in cells if val in self.opt_dict[cell]))
               if (c_sets.count(c_sets[0]) == size) & (len(c_sets[0]) == size):
                  if not all(len(self.opt_dict[c]) == size for c in c_sets[0]):
                     for cell in c_sets[0]:
                        self.opt_dict[cell] = set(val_group)
                     didsomething = True
                     self.recipe.append(f'opt_dict altered by cell_union({c_sets[0]}: {val_group})')
         if didsomething:
            return didsomething, []
      return didsomething, []
   

   def cell_union(self) -> tuple:
      '''Modifies the options dictionary. Returns True if it did anything '''
#        if a set of values (within a series) share a set of legal cells (of the same len),
#        no other values can be legal in these cells
      from itertools import combinations
      didsomething = False
      sizes = range(2, 5)
      for series in (self.coord_blocks + self.coord_rows + self.coord_cols):
         cells = [cell for cell in series if cell in self.opt_dict]
         val_set = set().union(*(self.opt_dict[cell] for cell in cells)) 
         val_comb = (group for r in sizes for group in combinations(val_set, r))
         for val_group in val_comb:
            group_union = set().union(*(set(cell for cell in cells if val in self.opt_dict[cell]) for val in val_group))
            if len(group_union) == len(val_group):
               cell_set_vals = set.union(*(self.opt_dict[cell] for cell in group_union)) # Need this to find out if the algorithm actually found something new
               if not cell_set_vals == set(val_group):
                  didsomething = True
                  for cell in group_union:
                     self.opt_dict[cell].intersection_update(set(val_group))
                  self.recipe.append(f'{self.ops +1:2} - options reduced by cell_union: {val_group} : {group_union}')
      return didsomething, []
   
   def val_set(self) -> tuple: # OBSOLETE replaced by val_union
      '''Modifies the options dictionary. Returns True if it did anything '''
      # if a set of cells (within a shape) share a set of values (of the same len), these 
      # values can't be legal for any other cells within that shape
      from itertools import combinations
      didsomething = False
      sizes = range(2, 5)
      for size in sizes:
         for series in (self.coord_blocks + self.coord_rows + self.coord_cols):
            cells = [cell for cell in series if cell in self.opt_dict]
            if len(cells) < size:
               continue
            cell_comb = combinations(cells, size)
            for cell_group in cell_comb: # if a set of cells
               v_sets = []
               for cell in cell_group:
                  v_sets.append(self.opt_dict[cell])
               if (v_sets.count(v_sets[0]) == size) & (len(v_sets[0]) == size): # share the same set of values
                  for other_cell in cells:
                     if other_cell not in cell_group:
                        for val in v_sets[0]: # these values
                           if val in self.opt_dict[other_cell]:
                              self.opt_dict[other_cell].remove(val)
                              didsomething = True
                              self.recipe.append(f'opt_dict altered at {other_cell} by val_set({v_sets[0]}: {cell_group})')
         if didsomething:
            return didsomething, []
      return didsomething, []
   

   def val_union(self) -> tuple:
      '''Modifies the options dictionary. Returns True if it did anything '''
      # if a set of cells (within a shape) share a set of values (of the same len), these 
      # values can't be legal for any other cells within that shape
      from itertools import combinations
      didsomething = False
      sizes = range(2, 5)
      for series in (self.coord_blocks + self.coord_rows + self.coord_cols):
         cells = [cell for cell in series if cell in self.opt_dict]
         cell_comb = (group for r in sizes for group in combinations(cells, r))
         for cell_group in cell_comb:
            group_union = set().union(*(self.opt_dict[cell] for cell in cell_group))
            if len(group_union) == len(cell_group):
               for other_cell in set(cells).difference(set(cell_group)):
                  for val in group_union:
                     if val in self.opt_dict[other_cell]:
                        self.opt_dict[other_cell].remove(val)
                        didsomething = True
                        self.recipe.append(f'{self.ops +1:2} - ({other_cell[0]:2},{other_cell[1]:2}) : options reduced by val_union: {cell_group} : {group_union}')
      return didsomething, []
            
   def available_ins(self) -> dict:
      self.build_option_dictionary()
      ins = {}
      for alg in (self.only_val, self.only_cell):
         ins[alg.__name__] = ''
         ins.update(alg()[1])
      return ins


   def available_ops(self):
      from efpkg.tools import TicToc
      self.build_option_dictionary()
      self.ops = 0
      ins = {}
      performance = {}
      t = TicToc()
      for alg in (self.only_val, self.only_cell):
         ins[alg.__name__] = ''
         t.tic()
         ins.update(alg()[1])
         performance.update({alg.__name__: t.tocout()})

      for alg in (self.val_iso, self.cell_union, self.val_union, self.cell_set, self.val_set):
         t.tic()
         alg()
         performance.update({alg.__name__: t.tocout()})
         ins.update({alg.__name__: line for line in self.recipe})
         self.recipe = []
         self.build_option_dictionary()
      return ins, performance
# End of class DictSolver
   

# Solver testing
if __name__ == "__main__":
   sample16 = [
         [10,5,0,0,0,0,0,0,3,8,6,0,0,15,0,0],
         [0,0,0,0,12,0,0,0,0,0,0,0,0,3,1,0],
         [0,2,0,0,3,11,8,0,0,0,0,0,0,6,7,4],
         [8,3,0,0,0,0,0,0,0,0,0,5,13,0,14,0],
         [0,7,0,0,5,8,2,13,0,9,0,0,0,4,0,1],
         [2,1,0,9,0,0,4,12,11,7,13,0,0,0,3,0],
         [16,4,0,14,0,3,0,7,0,12,0,0,0,13,10,2],
         [6,13,0,0,0,0,10,1,4,3,0,2,0,0,0,0],
         [1,0,0,0,10,4,14,11,12,16,2,8,3,7,0,0],
         [15,10,2,8,0,12,0,3,6,5,14,7,4,1,0,0],
         [0,0,7,4,0,0,0,2,0,0,3,0,0,0,12,0],
         [11,0,12,3,8,0,0,0,9,1,4,13,2,10,0,14],
         [5,0,0,0,0,0,3,4,13,6,11,12,1,0,9,0],
         [13,0,6,0,16,0,12,9,1,2,7,3,15,5,4,0],
         [0,0,0,0,0,5,13,8,16,10,15,9,0,0,0,0],
         [7,12,9,0,0,0,0,15,0,14,0,4,10,16,13,3]]

   sample2x3 = [
      [0,0,3,0,5,0],
      [2,4,5,0,3,0],
      [0,0,0,0,6,2],
      [1,2,0,0,0,0],
      [0,5,0,3,1,6],
      [0,6,0,4,0,0]
]
   samplex = [
      [5,0,0,8,0,0,1,2,0],
      [0,0,0,0,1,0,0,3,0],
      [1,4,0,0,5,6,9,7,8],
      [0,0,0,0,0,3,0,0,2],
      [0,7,5,0,2,0,6,1,0],
      [9,0,0,5,0,0,0,0,0],
      [4,5,1,6,3,0,0,8,7],
      [0,9,0,0,8,0,0,0,0],
      [0,6,8,0,0,2,0,0,5]]

   sample2x3_s = '003050245030000062120000050316060400'

   worlds_hardest = '800000000003600000070090200050007000000045700000100030001000068008500010090000400'
   s = DictSolver(worlds_hardest, blshape = None, algorithm='def')
   s.solve()
   s.recipe