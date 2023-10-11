import numpy as np
from mpi4py import MPI
from LaplaceSolver import LaplaceSolver

"Builds rooms, sets the boundary vals for each boundary,"
"and assigns the different ranks to solve on the rooms"




if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    print(size)
    dx = 0.05
    Omega_1 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Neumann', 'South': 'Dirichlet', 'West': 'Dirichlet'})
    Omega_2 = LaplaceSolver(1,2, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Dirichlet'})
    Omega_3 = LaplaceSolver(1,1, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'} )
    # Omega_4 = LaplaceSolver(0.5, 0.5, dx, {'North': 'Dirichlet', 'East': 'Dirichlet', 'South': 'Dirichlet', 'West': 'Neumann'})

    # Specify specific bound vals.

    iter = 10
    for i in range(iter):
        if rank == 0: # room2
            print("Rank is 0")
            u, U = Omega_2.get_solutions()
            hr = int(U.shape[0]/2) # half row index
            if(i==0): # 1st iter
                Omega_2.solve()
            else:
                
                bound_1 = comm.recv(source = 1)
                bound_3 = comm.recv(source = 2)
                # bound4 = comm.recv(source = 3)
                
                Omega_2.set_Dirichlet_boundary('West', np.append(U[0:hr-1,0], bound_1))
                Omega_2.set_Dirichlet_boundary('East', np.append(bound_3, U[hr+1:-1,-1]))
                
                Omega_2.solve() 
            # Gets whole wall of Omega_2 and we only want to send the Gamma_1 boundary
            bounds_r1 = Omega_2.calculate_Neumann_boundary('West')[hr:-1]
            bounds_r3 = Omega_2.calculate_Neumann_boundary('East')[0:hr]
            comm.send(bounds_r1, dest = 1) # dest = 0?
            comm.send(bounds_r3, dest = 2)
    
        if rank == 1: #room1
            print("Rank is 1")
            bounds_r1 = comm.recv(source = 0)
       
            Omega_1.set_Neumann_boundary('East', bounds_r1)
            Omega_1.solve()
            bound_1 = Omega_1.get_Dirichlet_boundary('East')
            comm.send(bound_1, dest = 0)
    
        if rank == 2: #room3
            print("Rank is 2")
            bounds_r3 = comm.recv(source = 0)
       
            Omega_3.set_Neumann_boundary('West', bounds_r3)
            Omega_3.solve()
            bound_3 = Omega_3.get_Dirichlet_boundary('West')
            comm.send(bound_3, dest = 0)

        if(i == iter-1):
            if rank == 0:
                _, U_2 = Omega_2.get_solutions()

                comm.send(U_2, dest=3, tag=2)
            if rank == 1:
                _, U_1 = Omega_1.get_solutions()
                comm.send(U_1, dest=3, tag=1)
            if rank == 2:
                _, U_3 = Omega_3.get_solutions()
                comm.send(U_3, dest=3, tag=3)
        if rank == 3:
            print("Rank is 3, We're done")
            U_1 = comm.recv(source = 1, tag=1)
            U_3 = comm.recv(source = 2, tag=3)
            U_2 = comm.recv(source=0, tag=2)

            import matplotlib.pyplot as plt

        def plot_solution(U, title):
            plt.imshow(U, cmap='jet', extent=[0, U.shape[1], 0, U.shape[0]], origin='lower')
            plt.colorbar()
            plt.title(title)
            plt.show()
        
        plot_solution(U_1, "Room 1 Solution")
        plot_solution(U_2, "Room 2 Solution")
        plot_solution(U_3, "Room 3 Solution")
            



