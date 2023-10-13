import numpy as np
from mpi4py import MPI
from LaplaceSolver import LaplaceSolver
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

def plot_and_save_heatmap(matrix, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title('Heatmap of ' + filename)
    plt.savefig(filename + '.png')
    plt.close()


"Builds rooms, sets the boundary vals for each boundary,"
"and assigns the different ranks to solve on the rooms"
dx = (1/20)

n = 10

Omega_1 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Neumann', 'South': 'Dirichlet', 'West': 'Dirichlet'})    
Omega_2 = LaplaceSolver(1,2, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Dirichlet'})
Omega_3 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'} )
# Omega_4 = LaplaceSolver(0.5, 0.5, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'})
   
print('here')
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
Omega_3.set_Dirichlet_boundary('North', 15*np.ones(Omega_3.N))
Omega_3.set_Dirichlet_boundary('South', 15*np.ones(Omega_3.N))

U_1 = Omega_1.U
U_2 = Omega_2.U
U_3 = Omega_3.U

U_1List = []
U_2List = []
U_3List = []

omega = 0.8

## Måste köra mpirun --oversubscribe -np 3 python3 mpicom.py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



for i in range(n):
    print(f'i = {i}')
    if rank == 2:
        if i == 0: # Omega 2
            U_2 = Omega_2.get_solutions()
            #print(f'First iteration, U1 = {U_1}')
        else:
            bound_1 = comm.recv(source = 1, tag = 1)
            bound_2 = comm.recv(source = 3, tag = 4)
            hr = int(Omega_2.N/2)
            dc_west = np.append(U_2[0:hr,0], bound_1)
            dc_east = np.append(bound_2, U_2[0:hr,-1])
            Omega_2.set_Dirichlet_boundary('West', dc_west)
            Omega_2.set_Dirichlet_boundary('East', dc_east)
            
            # Relaxation
            U_2_past = U_2
            U_2 = Omega_2.get_solutions()
            U_2 = omega*U_2 + (1-omega)*U_2_past # Relaxation
            U_2List.append(U_2)
            
        bound_N_1 = Omega_2.calculate_Neumann_boundary('West')
        bound_N_1 = bound_N_1[Omega_1.N-2:-1] # Need only half values
        bound_N_2 = Omega_2.calculate_Neumann_boundary('East')
        bound_N_2 = bound_N_2[0:Omega_1.N]
        comm.send(bound_N_1, dest = 1, tag = 2)
        comm.send(bound_N_2, dest = 3, tag = 3)
        if i == 9:
            comm.send(U_2, dest = 0, tag = 12)
            #print(f'U1 = \n {U_1}')
    if rank == 1: #Omega 1
        bound_N_1 = comm.recv(source = 2, tag = 2)
        Omega_1.set_Neumann_boundary('East', bound_N_1)
        U_1_past = U_1
        U_1 = Omega_1.get_solutions()
        U_1 = omega*U_1 + (1-omega)*U_1_past # Relaxation
        bound_1 = Omega_1.get_Dirichlet_boundary('East')
        comm.send(bound_1, dest = 2, tag = 1)
        if i == 9:
            comm.send(U_1, dest = 0, tag = 11)
        
    if rank == 3: # Omega 3
        bound_N_2 = comm.recv(source = 2, tag = 3)
        Omega_3.set_Neumann_boundary('West', bound_N_2)
        U_3_past = U_3
        U_3 = Omega_3.get_solutions()
        U_3 = omega*U_3 + (1-omega)*U_3_past # Relaxation
        U_3List.append(U_3)
        bound_2 = Omega_3.get_Dirichlet_boundary('West')
        comm.send(bound_2, dest = 2, tag = 4)
        if i == 9:
            comm.send(U_3, dest = 0, tag = 10)
    
    if rank == 0:
        U_1_final = comm.recv(source = 1, tag = 11)
        U_2_final = comm.recv(source = 2, tag = 12)
        U_3_final = comm.recv(source = 3, tag = 10)
        
        """
        print(f'U_1final = \n {U_1_final}')
        print(f'U_2final = \n {U_2_final}')
        print(f'U_3final = \n {U_3_final}')
        """
        plot_and_save_heatmap(U_1_final, 'U_1_final')
        plot_and_save_heatmap(U_2_final, 'U_2_final')
        plot_and_save_heatmap(U_3_final, 'U_3_final')
        
        N_square = Omega_1.N
        N_rectangle = Omega_2.N
        M_rectangle = Omega_2.M
        
        # North West, South East, ...
        NW_N = N_square - 1
        
        NW = np.zeros((NW_N, NW_N))
        
        # Remove last column to be able to block correctly
        SW = U_1_final[0:, 0:-1]
        
        
        NW.fill(None)
    
        West = np.block( [ [NW],[SW]  ]  )
        
        West_Middle = np.block( [West, U_2_final[0:,0:-1] ])
        print(U_2_final.shape[0])
        
        NE = U_3_final[0:, 0:]
        SE = np.zeros((N_square, N_square))
        SE = SE[0:-1, 0:]
        SE.fill(None)

        East = np.block([[NE], [SE]])   
        Apartment = np.block([West_Middle, East])
        
        plot_and_save_heatmap(Apartment, 'Apartment Projekt 3')
        
        


       
