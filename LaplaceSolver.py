import numpy as np
import scipy
from scipy import linalg
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
"""
    Dirichlet funkar nu väldigt bra.
   
    Samma gäller Neumann nu också. Verkar funka bra efter några korta test.
   
    Rektangulära rum funkar.
   
    Vi kan se hur varje internal
    punkt är precis medelvärdet av omringande punkter så som värmeledning
    ska funka.
   
"""
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
        self.M = int(width / dx) + 1
        self.N = int(height / dx) + 1
       
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
                    self.dim_X += 1
                elif wall == 'East' or wall == 'West':
                    self.dim_Y += 1
       
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
       
        # The Neumann values in a similar grid.
        self.dU = np.zeros((self.N, self.M))
       
        # It's so fun! Less if statements!
        self.boundary_index = {
            'West'  : (slice(None, None, None), 0),  #  Equivalent to [: ,  0]
            'East'  : (slice(None, None, None), -1), #  Equivalent to [: , -1]
            'North' : (0, slice(None, None, None)),  #  Equivalent to [0 ,  :]
            'South' : (-1, slice(None, None, None))  #  Equivalent to [-1,  :]
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


   
    # Assume that this works, this generates the approximation of the Laplace
    # operator.
    def __generate_matrix_A(self):  
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
   
    # Updates the matrix A. Goes through each row that corresponds to
    # Neumann condition and changes it! According to slide 27 of lecture    
    # Only implemented for Neumann on East or West wall.
    def __Neumann_update_A(self, A):
        size = self.dim_X*self.dim_Y
        for wall, condition in self.boundary_spec.items():
           
            # See whiteboard notes lol
            # Find indices of these values in u (the rows that need to be changed in A)
            if condition == 'Neumann':
                if wall == 'West':
                    row_indices = [i * (self.M-1) for i in range(self.N-2)]
                else:
                    row_indices = [self.M - 2 + i * (self.M-1) for i in range(self.N-2)]
               
        # i the row that needs to be changed, j = k + 1 will be column    
        for i in row_indices:
            j = i
            A[i] = 0
            A[i, j] = -3 / (self.dx ** 2)
            if j - 1 >= 0:
                A[i, j - 1] = 1 / (self.dx ** 2) # won't be a problem I promise
            if j - 3 >= 0:
                A[i, j - 3] = 1 / (self.dx ** 2)
        return A

    """
    Easier to work with matrices that we reshape.
    In the loop we are adding the contributions of each wall one at a time,
    so in the corners we will have contributions from both longitudal and
    latitudal directions. Those values need to be sent to the right hand side.
   
    Dirichlet:
                             (1)
                              \
                       (1)--(-4)--(1)
                              \
                             (1)                    
    If      U = [u00, u01, u02, u03;    
                 u10, u11, u12, u13;
                 u20, u21, u22, u23;
                 u30, u31, u32, u33]  
   
    and we have the vector we are solving for u = [u11, u12, u21, u22]
    Assuming U[:,0] and U[0,:] are Dirichlet walls, then according to the
    stencil for u11, we need to send u01 and u10 to b on the row corresponding
    to the equation for u11 since these values are known.
   
    This method does this for the matrix B that is then reshaped into a
    column for the equation system Au = b.
   
    """
    def __generate_matrix_b(self):
        B = np.zeros((self.dim_X, self.dim_Y))
        # Note that the boundary values of the grid are stored on
        # boundary of the matrix U (or dU) which contains the grid points!
        # We take [1:-1] because we want the "inner" boundary points
        # according to the above stencil.
        for wall, condition in self.boundary_spec.items():
            index = self.boundary_index[wall]
            if condition == 'Dirichlet':
                # Ugly solution :(
                # Only when we are updating the west or east walls will we
                # go in here to update the boundary matrix B,
                # need to play with indices because Neumann boundaries
                # are clunky ...
                if 'Neumann' in self.boundary_spec.values() and (index[0] == 0 or index[0] == -1):
                    Neumann_wall = self.__get_key(self.boundary_spec, 'Neumann')
                    Neumann_index = self.boundary_index[Neumann_wall][1]
                    if Neumann_index == -1: # East wall
                        B[index][0:-1] -= self.U[index][1:-1] / (self.dx ** 2)
                    elif Neumann_index == 0: # West wall
                        B[index][1:] -= self.U[index][1:-1] / (self.dx ** 2)
                else:
                    B[index] -= self.U[index][1:-1] / (self.dx ** 2)
            else:
                B[index] -= self.dU[index][1:-1] / (self.dx)
        b = B.reshape(-1,1) # Back to column
        return b
   
    # Bad solution ...
    def __get_key(self, my_dict, value):
        for key, val in my_dict.items():
            if val == value:
                return key
        return None
    def give_matrices(self):
       
        A = self.__generate_matrix_A()
        b = self.__generate_matrix_b()
        if 'Neumann' in self.boundary_spec.values():
            A = self.__Neumann_update_A(A)
       
        return A,b
# solves Au = b
    def solve(self):
        # A = self.__generate_matrix_A()
        # b = self.__generate_matrix_b()
        A,b = self.give_matrices() # YES!!
        u = scipy.linalg.solve(A,b)
        U_inner = u.reshape(self.dim_X, self.dim_Y)
        if 'Neumann' in self.boundary_spec.values():
            A = self.__Neumann_update_A(A=A) # Update
            Neumann_wall = self.__get_key(self.boundary_spec, 'Neumann')
            Neumann_index = self.boundary_index[Neumann_wall][1]
            if Neumann_index == 0: # West wall
                self.U[1:self.N-1, 0:self.M-1] = U_inner
            else:
                self.U[1:self.N-1, 1:self.M] = U_inner
        else:
            self.U[1:self.N-1, 1:self.M-1] = U_inner
        return u, U_inner, self.U
       

    # Just returns the solution vectors, note U is the actual matrix grid
    # that we want to plot, contains boundary values, while u only contains
    # the inner points.
    def get_solutions(self):
        _, _, solutions = self.solve()
        return solutions
   
if __name__ == "__main__":
    dx = 1 / 20
    Omega_1 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Neumann', 'South': 'Dirichlet', 'West': 'Dirichlet'})    
    Omega_2 = LaplaceSolver(1,2, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Dirichlet'})
    Omega_3 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'} )
    Omega_3 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'} )
    Omega_4 = LaplaceSolver(0.5, 0.5, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'})
    hr = int(Omega_2.N/2)
    qr = int(Omega_2.N/4)
    U_4 = Omega_4.U
    bound_1 = np.ones(Omega_2.U.shape[0]-hr)
    U_2 = Omega_2.U
    Omega_2.set_Dirichlet_boundary('West', np.append(U_2[0:hr,0], bound_1))
    b1 = Omega_1.get_Dirichlet_boundary('East')
    bound_N_1 = Omega_2.calculate_Neumann_boundary('West')
    bound_N_2 = Omega_2.calculate_Neumann_boundary('East')
   
    # Specify specific bound vals.
    Gamma_heater = 40*np.ones(Omega_1.N)
    Gamma_window = 5*np.ones(Omega_1.N)
    Gamma_O2 = 15*np.ones(Omega_1.N - 1)

    Omega_1.set_Dirichlet_boundary('West', Gamma_heater)
    Omega_1.set_Dirichlet_boundary('North', 15*np.ones(Omega_1.N))
    Omega_1.set_Dirichlet_boundary('South', 15*np.ones(Omega_1.N))

    Omega_2.set_Dirichlet_boundary('East', 15*np.ones(Omega_2.N))
    Omega_2.set_Dirichlet_boundary('West', 15*np.ones(Omega_2.N))
    Omega_2.set_Dirichlet_boundary('North', Gamma_heater)
    Omega_2.set_Dirichlet_boundary('South', Gamma_window)

    Omega_3.set_Dirichlet_boundary('East', Gamma_heater)
    Omega_3.set_Dirichlet_boundary('North', 15*np.ones(Omega_1.N))
    Omega_3.set_Dirichlet_boundary('South', 15*np.ones(Omega_1.N))
    # Omega_1.set_Neumann_boundary('East', 15*np.ones(Omega_1.N))
