import numpy as np
import scipy
from scipy import linalg
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

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
            
            u11, u12, u13 u21, u22, u23
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
        i2 = i1 + 1*(i1 >= 0) - 1*(i1 < 0)
        u_boundary = self.U[self.boundary_index[wall]] # u_i
        u_internal = self.U[:, i2] # u_(i-1) or u_(i+1)
       
        u_neumann = (1*(i1 >= 0) - 1*(i1 < 0))*(u_internal - u_boundary)*(1/self.dx)
        
        
        # self.set_Neumann_boundary(wall, u_neumann)
        # print(u_boundary)
        return u_neumann


   
    # Assume that this works, this generates the approximation of the Laplace
    # operator.
    def __generate_matrix_A(self):  
        size = self.dim_X*self.dim_Y
        A = np.zeros((size, size))
        """
        Ddiag  = -4 * np.eye(size)
        Dupper = np.diag([1] * (size - 1), 1)
        Dlower = np.diag([1] * (size - 1), -1)
        D = Ddiag + Dupper + Dlower
        Ds = [D] * (size)
        A  = scipy.linalg.block_diag(*Ds)
        I = np.ones((size) * (size - 1))
        Iupper = np.diag(I, size)
        Ilower = np.diag(I, -size)
        A += Iupper + Ilower
        """
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
        
        Neumann_wall = self.__get_key(self.boundary_spec, 'Neumann')        
        i1 = self.boundary_index[wall][1] # 0 left side, -1 right side
        i2 = i1 + 1*(i1 >= 0) - 1*(i1 < 0) # i2 is + or - 1 depending on
        # where the Neumann condition is located
        print(f'i2 = {i2}')
        
        ## Notera många ändringar har skett här. När detta ändrades
        # ser lösningen ännu sämre ut nu, men boundary conditionen 
        # där vi har 0 är iaf symmetrisk för tillfället. Jag kollar
        # om felet ligger i update b
        for i in row_indices: # row indices är rader där vi har Neumann punkter
            j = i
            k = self.dim_X
            A[i] = 0
            A[i, j] = -3 / (self.dx ** 2)
            print(f'A size is {A.size}')
            if j + i2 >= 0 and j + i2 < A.size:
                A[i, j + i2] = 1 / (self.dx ** 2)
            if j - k >= 0:
                A[i, j - k] = 1 / (self.dx ** 2)
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
        B = np.zeros((self.dim_X, self.dim_Y)) # B corresponds to U, so 
        # B[i,j] is right hand side for the equation for the equation 
        # for the unknown U[i,j]
        B_2 = np.zeros((self.dim_X*self.dim_Y, self.dim_X*self.dim_Y))
        # Note that the boundary values of the grid are stored on
        # boundary of the matrix U (or dU) which contains the grid points!
        # We take [1:-1] because we want the "inner" boundary points
        # according to the above stencil.
        for wall, condition in self.boundary_spec.items():
            index = self.boundary_index[wall]
            if condition == 'Dirichlet':
                if 'Neumann' in self.boundary_spec.values() and (wall == 'North' or wall == 'South'):
                    Neumann_wall = self.__get_key(self.boundary_spec, 'Neumann')
                    if Neumann_wall == 'East':
                        B[index][0:-1] -= self.U[index][1:-1] / (self.dx ** 2)
                    elif Neumann_wall == 'West':
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
        A, b = self.give_matrices()
        u = scipy.linalg.solve(A,b)
        U_inner = u.reshape(self.dim_X, self.dim_Y)
        if 'Neumann' in self.boundary_spec.values():
            Neumann_wall = self.__get_key(self.boundary_spec, 'Neumann')
            if Neumann_wall == 'West': # West wall
                self.U[1:-1, 0:-1] = U_inner
            else:
                self.U[1:-1, 1:] = U_inner
        else:
            self.U[1:self.N-1, 1:self.M-1] = U_inner
        return u, U_inner, self.U
       

    # Just returns the solution vectors, note U is the actual matrix grid
    # that we want to plot, contains boundary values, while u only contains
    # the inner points.
    def get_solutions(self):
        _, _, solutions = self.solve()
        return solutions
  
def plot_and_save_heatmap(matrix, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title('Heatmap of ' + filename)
    plt.savefig(filename + '.png')
    plt.close()
    
if __name__ == "__main__":
    dx = 1 / 4
    
    """
    Omega_1 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Neumann', 'South': 'Dirichlet', 'West': 'Dirichlet'})    
    Omega_2 = LaplaceSolver(1,2, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Dirichlet'})
    """
    Omega_3 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'} )
    #Omega_3 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Dirichlet'} )
    #Omega_4 = LaplaceSolver(0.5, 0.5, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'})

   
    # Specify specific bound vals.
    Gamma_heater = 40*np.ones(Omega_3.N)
    Gamma_window = 5*np.ones(Omega_3.N)
    # Gamma_O2 = 15*np.ones(Omega_1.N - 1)

    """
    Omega_1.set_Dirichlet_boundary('West', Gamma_heater)
    Omega_1.set_Dirichlet_boundary('North', 15*np.ones(Omega_1.N))
    Omega_1.set_Dirichlet_boundary('South', 15*np.ones(Omega_1.N))

    Omega_2.set_Dirichlet_boundary('East', 15*np.ones(Omega_2.N))
    Omega_2.set_Dirichlet_boundary('West', 15*np.ones(Omega_2.N))
    Omega_2.set_Dirichlet_boundary('North', Gamma_heater)
    Omega_2.set_Dirichlet_boundary('South', Gamma_window)
    """
    Omega_3.set_Dirichlet_boundary('East', Gamma_heater)
    Omega_3.set_Dirichlet_boundary('North', 15*np.ones(Omega_3.N))
    Omega_3.set_Dirichlet_boundary('South', 15*np.ones(Omega_3.N))
    #Omega_3.set_Neumann_boundary('West', 100*np.ones(Omega_3.N))
    #Omega_3.set_Dirichlet_boundary('West', 15*np.ones(Omega_3.N))
    U = Omega_3.get_solutions()
    plt.figure(figsize=(10, 8))
    plt.imshow(U, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Value')
    #plt.title('Heatmap of ' + filename)
    #plot_and_save_heatmap(U, 'Test på Omega3 bara')
