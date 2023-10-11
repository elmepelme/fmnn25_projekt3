import numpy as np
import scipy


# Solves the laplace equation uxx + uyy = 0 on a rectangle with given
# width and height
""" 
Optional parameter to specifiy boundary condition setup

For example:
boundary_spec = {
    'North': 'Dirichlet',
    'South': 'Neumann',
    'East': 'Dirichlet',
    'West': 'Dirichlet'
}
Picture: 
                     Dirichlet
                -----------------
                |                |
                |                |
     Dirichlet  |                | Dirichlet
                |                |
                |                | 
                ------------------
                     Neumann 
"""

class LaplaceSolver:
    def __init__(self, width, height, dx, boundary_spec = {
        'North': 'Dirichlet',
        'South': 'Dirichlet',
        'East': 'Dirichlet',
        'West': 'Dirichlet'
    }):
        
        self.dx = dx  # Grid spacing (dx = dy)
        
        # Initialize grid (this includes boundary points)
        self.N = int(width / dx)
        self.M = int(height / dx)
        
        # Dimensions of the (internal) grid points we will solve for
        self.dim_X = self.N - 2
        self.dim_Y = self.M - 2
        
        """
        Example if all walls Dirichlet:
        
        N = 5
        M = 6
        dim_X = 3
        dim_Y = 4
        
        @ = "boundary", * = "point to solve for". 
        
        Then our grid looks like this:
        
        @ @ @ @ @
        @ * * * @
        @ * * * @
        @ * * * @
        @ * * * @
        @ @ @ @ @
        
        Each Neumann condition will change one wall from "@" to "*" 
        which means we will solve for these points
        """
        
        #: For each Neumann wall we will add those points to be solved in u
        for wall, condition in boundary_spec.items():
            if condition == 'Neumann':
                if wall == 'North' or wall == 'South':
                    self.dim_Y += 1
                elif wall == 'East' or wall == 'West':
                    self.dim_X += 1
        
        # Vector with approximated solutions
        self.u = np.zeros(self.dim_X * self.dim_Y)
        # u = [u11, u12, u21, u22, u31, u32];
        
        # Full grid will be here, mostly used to keep track of boundary values
        self.U = np.zeros((self.N, self.M))
        
        # The Neumann values
        self.dU = np.zeros((self.N, self.M))
        
        """
        The resulting matrix will be: U = 
            [u00, u01, u02, u03;
             u10, u11, u12, u13;
             u20, u21, u22, u23;
             u30, u31, u32, u33]
        """
        
        # It's so fun! Less if statements! 
        self.boundary_index = {
            'West'  : (slice(None, None, None), 0),  #  Equivalent to [:, 0]
            'East'  : (slice(None, None, None), -1), #  Equivalent to [:, -1]
            'North' : (0, slice(None, None, None)),  #  Equivalent to [0, :]
            'South' : (-1, slice(None, None, None))  #  Equivalent to [-1, :]
}
       
    # wall is 'North', 'East', 'South', or 'West'   
    # Values is a vector of size N or M 
    def set_Dirichlet_boundary(self, wall, values): 
        self.U[self.boundary_index[wall]] = values
        
    def get_Dirichlet_boundary(self, wall):
        return self.U[self.boundary_index[wall]]
    
    def set_Neumann_boundary(self, wall, values):
        self.dU[self.boundary_index[wall]] = values
        
    def get_Neumann_boundary(self, wall):
        return self.dU[self.boundary_index[wall]] = values        
    # Implementing only x derivative for now
    # This could be implemented a lot more readable, but this was such a 
    # blast
    
    """
    Calculates the Neumann boundary values on wall = 'East' or wall = 'West'
    Discretization from slide 11 of lecture on Laplace equation
    
    u_{i}'  = 1/dx * (u_{i+1} - u{i})
    
    West wall, then i =  0, we will need columns U[:,0] and U[:, 1]
    East wall, then i = -1, we will need columns U[:, -1] and U[:, -2]
    
    """
    def calculate_Neumann_boundary(self, wall):
        i1 = self.boundary_index[wall][1] # 0 left side, -1 right side
        plus_minus_one = + 1 * (i1 >= 0) - 1 * (i1 < 0) # It is so fun!
        i2 = i1 + plus_minus_one 
        u_boundary = self.U[self.boundary_index[wall]] # u_i
        u_internal = self.U[:, i2] # u_(i-1) or u_(i+1)
        
        u_neumann = (plus_minus_one) * (u_internal - u_boundary) * (1 / self.dx) # Tihi
        self.set_Neumann_boundary(wall, u_neumann) # Maybe not needed
        return u_neumann
    
    # Just returns the solution vectors, note U is the actual matrix grid
    # with both internal and external points that we want to plot
    def get_solutions(self):
        return self.u, self.U
    
    # Private method to calculate the matrices for solving the system
    def __create_matrices(self):
        A = np.zeros((self.dim_X * self.dim_Y, self.dim_X * self.dim_Y))
        b = np.zeros((self.dim_X * self.dim_Y, 1))
        return A,b

    # Extra parameters maybe? Solves the system
    def solve(self):
        A,b = self.__create_matrices()
        return np.linalg.solve(A,b)

if __name__ == "__main__":
    width, height = 1, 2  # Number of internal grid points in x and y dimensions
    dx = 0.2  # Grid spacing (dx = dy)
   
    # Initialize solver
    laplace_solver = LaplaceSolver(width, height, dx)
    u, U = laplace_solver.get_solutions()
    # Print the solution (you might want to visualize this using matplotlib)
    print("Solution grid:")
    print(U)
    print(laplace_solver.boundary_index['East'][1])
    val1 = np.ones(5)
    val2 = 3*np.ones(5)
    laplace_solver.set_Dirichlet_boundary('East', val1)
    laplace_solver.U[:,-2] = val2
    print(laplace_solver.U)
    laplace_solver.set_Dirichlet_boundary('West', val1)
    laplace_solver.U[:,1] = val2
    print(laplace_solver.U)
    print(laplace_solver.calculate_Neumann_boundary('West'))
