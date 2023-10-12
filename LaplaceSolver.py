import numpy as np
import scipy
from scipy import linalg
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

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
        self.boundary_spec = boundary_spec
        self.dx = dx  # Grid spacing (dx = dy)
        
        # Initialize grid (this includes boundary points)
        self.N = int(width / dx) + 1
        self.M = int(height / dx) + 1
        
        # Dimensions of the (internal) grid points we will solve for
        # Assuming all Dirichlet boundaries
        self.dim_X = self.N - 2
        self.dim_Y = self.M - 2
        
        """
        Example if all walls Dirichlet:
        
        N = 4
        M = 4
        dim_X = 2
        dim_Y = 2
        
        @ = "boundary", * = "point to solve for". 
        
        Then our grid looks like this:
        
        @ @ @ @ 
        @ * * @
        @ * * @
        @ @ @ @
        
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
        # u = [u11, u12, u21, u22] for example
        
        # Full grid will be here, mostly used to keep track of boundary values
        self.U = np.zeros((self.N, self.M))
        """
        The resulting matrix will be: U = 
            [u00, u01, u02, u03;
             u10, u11, u12, u13;
             u20, u21, u22, u23;
             u30, u31, u32, u33]
        """
        
        # The Neumann values (maybe not used)
        self.dU = np.zeros((self.N, self.M))
        
        # It's so fun! Less if statements! 
        self.boundary_index = {
            'West'  : (slice(None, None, None), 0),  #  Equivalent to [: ,  0]
            'East'  : (slice(None, None, None), -1), #  Equivalent to [: , -1]
            'North' : (0, slice(None, None, None)),  #  Equivalent to [0 ,  :]
            'South' : (-1, slice(None, None, None))  #  Equivalent to [-1,  :]
            }
        
        # Define the matrix that approximates the Laplacian
        size = self.dim_X*self.dim_Y
        self.A = np.zeros((size, size))
        self.b = np.zeros((size, 1))
        
        # Matrix that is reshaped since it's easier to think about 
        # the matrix and not the column.
        self.B = np.zeros((self.dim_X, self.dim_Y))
       
    # wall is 'North', 'East', 'South', or 'West'   
    # Values is a vector of size N or M 
    def set_Dirichlet_boundary(self, wall, values): 
        self.U[self.boundary_index[wall]] = values
    
    def get_Dirichlet_boundary(self, wall): 
        return self.U[self.boundary_index[wall]]
    
    def set_Neumann_boundary(self, wall, values):
        self.dU[self.boundary_index[wall]] = values
    
    def get_Neumann_boundary(self, wall):
        return self.dU[self.boundary_index[wall]]
        
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
        i2 = i1 + 1*(i1 >= 0) - 1*(i1 < 0) # It is so fun!
        u_boundary = self.U[self.boundary_index[wall]] # u_i
        u_internal = self.U[:, i2] # u_(i-1) or u_(i+1)
        
        u_neumann = (1*(i1 >= 0) - 1*(i1 < 0))*(u_internal - u_boundary)*(1/self.dx) # Tihi
        self.set_Neumann_boundary(wall, u_neumann)
        return u_neumann

    # Just returns the solution vectors, note U is the actual matrix grid
    # that we want to plot, contains boundary values, while u only contains
    # the inner points.
    def get_solutions(self):
        return self.u, self.U
    
    
    # Assume that this works, this generates the approximation of the Laplace
    # operator. 
    def generate_matrix_A(self):   
        size = self.dim_X*self.dim_Y
        A = np.zeros((size, size))
        # Fill the matrix
        for i in range(size):
            # Diagonal value
            A[i, i] = -4
            # Upper diagonal
            if i + 1 < size and (i + 1) % self.dim_Y != 0:  # Ensure it doesn't cross block boundary
                A[i, i+1] = 1

            # Lower diagonal
            if i - 1 >= 0 and i % self.dim_Y != 0:  # Ensure it doesn't cross block boundary
                A[i, i-1] = 1

            # Upper block diagonal (identity matrix)
            if i + self.dim_Y < size:
                A[i, i + self.dim_Y] = 1

            # Lower block diagonal (identity matrix)
            if i - self.dim_Y >= 0:
                A[i, i - self.dim_Y] = 1

        return (1  / self.dx**2) * A
    
    """
    Easier to work with matrices that we reshape.
    In the loop we are adding the contributions of each wall one at a time, 
    so in the corners we will have contributions from both longitudal and 
    latitudal directions. Those values need to be sent to the right hand side.
                
                             (1)
                              \
                       (1)--(-4)--(1)
                              \
                             (1)     
    Dirichlet                         
    If      U = [u00, u01, u02, u03;     
                 u10, u11, u12, u13;
                 u20, u21, u22, u23;
                 u30, u31, u32, u33]  
    
    and we have the vector we are solving for u = [u11, u12, u21, u22]
    Assuming U[:,0] and U[0,:] are Dirichlet walls, then according to the 
    stencil for u11, we need to send u01 and u10 to b on the row corresponding
    to the equation for u11 since these values are known. 
    
    TO BE ADDED: Neumann conditions
    
    This method does this for the matrix B that is then reshaped into a
    column for the equation system Au = b. 
    
    """
    def generate_matrix_b(self):
        B = np.zeros((self.dim_X, self.dim_Y))
        # Define the matrix that approximates the Laplacian
        for wall, condition in self.boundary_spec.items():
            if condition == 'Dirichlet':
                index = self.boundary_index[wall]
                # Note that the boundary values of the grid are stored on
                # boundary of the matrix U which contains the grid points!
                # We take [1:-1] because we want the "inner" boundary points
                # according to the above stencil.
                B[index] -= self.U[index][1:-1]
        
        b = B.reshape(-1,1) # Back to column
        return (1 / self.dx ** 2) * b
            
# solves Au = b
    def solve(self):
        A = self.generate_matrix_A()
        b = self.generate_matrix_b()
        u = scipy.linalg.solve(A,b)
        U_inner = u.reshape(self.dim_X, self.dim_Y)
        self.U[1:self.N-1, 1:self.M-1] = U_inner
        return u, U_inner, self.U
        
if __name__ == "__main__":
    width, height = 2, 1  # Number of internal grid points in x and y dimensions
    dx = 1/2  # Grid spacing (dx = dy)
    walls = ['North', 'East', 'West', 'South']
    laplace_solver = LaplaceSolver(width, height, dx)
    laplace_solver.set_Dirichlet_boundary('East', 100*np.ones(5))
    print(f'U = \n {laplace_solver.U}')
    b = laplace_solver.generate_matrix_b()
    print(b)
    print(b.reshape(-1,1))
    u, U_inner, U = laplace_solver.solve()
    
    """
    Dirichlet funkar nu väldigt bra, vi kan se hur varje internal
    punkt är precis medelvärdet av omringande punkter så som värmeledning
    ska funka
    
    """
