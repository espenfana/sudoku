#import os
#os.chdir('d:\\code\\py\\ef_projects\\sudoku')
from sudoku_dict import DictSolver
import random
import string



class SudokuGenerator(DictSolver):
   'Generate sudoku boards'
   def __init__(self, seed=None, blshape=(3,3), null=0, values='numstr'):
      '''Initialize an empty board
         seed: For random function
         blshape: (x,y) The shape of the blocks of the board. The full board shape is determined by this
         null: What blank sqaures will be
         values: Options for values, [numstr, num, upper, lower]
      '''
      if not len(blshape) == 2:
         raise Exception('The supplied blshape is not len 2')
      self.bsize = blshape[0] * blshape[1]
      self.blshape = blshape
      self.arr = [[null for each in range(self.bsize)] for each in range(self.bsize)]
      if not seed:
         seed = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
      self.seed = seed
      self.null = null
      value_options = {'numstr' : set( (string.digits + string.ascii_uppercase)[1:self.bsize+1] ),
                       'num' : set(range(1,self.bsize+1)),
                       'upper' : set( (string.ascii_uppercase)[:self.bsize] ),
                       'lower' : set( (string.ascii_lowercase)[:self.bsize] )}
      self.all_values = value_options.get(values, values)
      if len(self.all_values) != self.bsize:
         raise Exception('The value set does not match the board size')
      if null in self.all_values:
         raise Exception('The value set can\'t contain the null value')


   def populate(self):
      'Populate the board with values'
      random.seed(self.seed)
      self.arr[0] = random.sample(self.all_values, self.bsize) # Get the board started by defining the first row
      super().__init__(self.arr, algorithm='brute', blshape=self.blshape) # Initilize the solver (parent class) to get access to those methods
      self.silent = True
      self.build_option_dictionary()
      self.randosolve() # 

   
   def decimate(self, algorithm='full', rot_symmetry=False):
      'Remove values from the board and checks if the board is still solvable'
      self.set_algorithms(algorithm)
      #if any([self.arr[i][j]])
      cs = [(i,j) for i in range(self.bsize) for j in range(self.bsize)]
      if rot_symmetry:
         cell_list = []
         st = set()
         for cell in cs:
            if cell not in st:
               cell_list.append((cell, (self.bsize-1-cell[0], self.bsize-1-cell[1])))
               st.add((self.bsize-1-cell[0], self.bsize-1-cell[1]))
      else:
         cell_list = [[cell] for cell in cs]
      random.shuffle(cell_list)
      self.cell_list = cell_list
      self.null_cells = 0
      self.pops = []
      for c in cell_list:
         pops = {(i,j): self.arr[i][j] for (i,j) in c}
         for (i,j) in pops:
            self.arr[i][j] = self.null
         if not self.gensolve(pops):
            for (i,j), val in pops.items():
               self.arr[i][j] = val
         else:
            self.null_cells += len(c)
            self.pops.append(pops)


   def randosolve(self):
      'Inserts values randomly in order to generate a valid sudoku board.'
      threshold = 1 # This is the set length we want to look for
      while (self.solved is False) & (threshold < self.bsize):
         for (i,j), cell_set in self.opt_dict.items():
            if len(cell_set) == threshold:
               rando_set = random.sample(cell_set, threshold)
               for val in rando_set:
                  self.arr[i][j] = val
                  self.update_option_dictionary((i,j), val)
                  if not self.error:
                     self.randosolve()
                     if self.solved: 
                        return
                  # Error is now implicit. Undo and retry
                  self.arr[i][j] = self.null
                  self.build_option_dictionary()
                  self.error = False
               return # No values were correct. The fault must be in earlier iterations
         threshold +=1 # No sets of this length were found, increase by 1
   

   def gensolve(self, pops):
      'Fully algorithmic solver. This guarantees unique solution.'
      arr_init = self.copyarr()
      self.solved = False
      self.recipe = []
      self.ops = 0
      self.build_option_dictionary()
      while (not self.solved) & (not self.error):
         # Loop through algorithms until board is solved or no action is found
         didsomething = False
         for algorithm in self.algorithms:
            didsomething, ins = algorithm()
            if len(ins):
               for (i, j), val in ins.items():
                  self.recipe.append(algorithm.__name__ + str(ins))
                  self.arr[i][j] = val
                  self.update_option_dictionary((i,j), val)
               break # returns to the while loop
         
         chksolv = (self.arr[i][j] == val for (i,j), val in pops.items())
         if all(chksolv): # We only need to check if the values we removed can be put back in place
            self.solved = True
         if not didsomething:
            break # exits the while loop (board not solvable)
            
      self.arr = arr_init
      return self.solved
      

if __name__== '__main__':
	from efpkg.tools import TicToc # timer tools
	t = TicToc()
	t.tic()
	g = SudokuGenerator(blshape=(4,3), values='num', seed='81xob3')
	g.populate()
	t.toc()
	print(g)

	t.tic()
	g.decimate(algorithm='full', rot_symmetry = False)
	print(g)
	print(g.null_cells, g.null_cells / (g.bsize ** 2))
	t.toc()

# s = DictSolver(g.copyarr(), algorithm='full', blshape=g.blshape)
# t.tic()
# s.solve()
# t.toc()
# s.recipe

# n16 = []
# for each in range(10):
#    g = SudokuGenerator(bsize= 16)
#    g.populate()
#    g.decimate()
#    n16.append(g.null_cells)
#    print(g)

   # Typical fill grades for (4,4) : [0.640625, 0.6171875, 0.6328125, 0.62890625, 0.64453125, 0.640625, 0.6328125, 0.63671875, 0.640625, 0.63671875]

# Interesting generated boards

# (4,4) seed '81xob3' 'num'

#  [[2, 8, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
#  [12, 0, 0, 15, 0, 0, 0, 0, 1, 0, 3, 9, 0, 8, 0, 0],
#  [0, 11, 5, 14, 0, 15, 0, 1, 0, 0, 0, 0, 0, 0, 7, 0],
#  [0, 0, 0, 10, 13, 4, 0, 0, 0, 5, 0, 12, 3, 0, 11, 0],
#  [0, 0, 0, 0, 14, 16, 2, 9, 12, 0, 0, 0, 0, 0, 0, 4],
#  [14, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 16, 11, 5, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 15, 3, 16, 0],
#  [0, 0, 7, 12, 6, 0, 0, 0, 0, 8, 0, 4, 0, 0, 0, 0],
#  [0, 16, 0, 0, 15, 0, 9, 6, 0, 0, 0, 0, 14, 0, 0, 0],
#  [0, 0, 0, 0, 2, 0, 0, 0, 0, 14, 0, 6, 0, 0, 0, 9],
#  [9, 13, 1, 0, 0, 0, 0, 4, 0, 0, 0, 2, 16, 0, 0, 0],
#  [0, 0, 0, 4, 0, 0, 12, 0, 0, 13, 0, 10, 0, 1, 15, 0],
#  [0, 0, 12, 8, 0, 1, 0, 0, 0, 9, 0, 0, 0, 0, 0, 5],
#  [15, 3, 0, 9, 0, 8, 7, 0, 0, 0, 0, 0, 0, 11, 4, 0],
#  [5, 0, 0, 0, 0, 0, 4, 11, 0, 0, 2, 3, 0, 0, 0, 10],
#  [0, 0, 6, 0, 0, 0, 13, 5, 0, 0, 0, 0, 0, 0, 2, 8]]
