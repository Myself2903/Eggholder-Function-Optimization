import numpy as np
from time import time
from simanneal import Annealer
import matplotlib.pyplot as plt
from ipywidgets import interactive, fixed


def eggholder_function(x):
    return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))

#Eggholder function and solution plot
def plotter(E,A, solution):
    try:
        x, y, z = solution
        fig=plt.figure(figsize=[12,8])
        ax=plt.axes(projection='3d')
        ax.plot_surface(X1,X2,eggholder_function((X1,X2)),color='red',alpha=0.7)
        ax.plot_wireframe(X1,X2,eggholder_function((X1,X2)),ccount=2,rcount=2, color='orange',alpha=0.8)   
        ax.scatter(x, y, z, color='blue', s=100)
        ax.view_init(elev=E,azim=A)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1,x2)')
        plt.show()
    except Exception as e:
        print(e)


class EggholderProblem(Annealer):
    
    def __init__(self, state, interval=[-512,512], neighbour_delta=128, Tmax=25000, Tmin=2.5, steps=20000):
        self.interval = interval
        self.energies = []
        self.neighbour_delta = neighbour_delta
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.steps = steps
        super(EggholderProblem, self).__init__(state)

    #new solution creation
    def move(self):
        delta = np.random.uniform(-self.neighbour_delta,self.neighbour_delta,2)
        self.state[0] = self.state[0]+delta[0] if self.interval[0] <= self.state[0] + delta[0] <= self.interval[1] else self.state[0]
        self.state[1] = self.state[1]+delta[1] if self.interval[0] <= self.state[1] + delta[1] <= self.interval[1] else self.state[1]
        
    #fitness function
    def energy(self):
        energy = eggholder_function(self.state)
        self.energies.append(self.best_energy)
        return energy



if __name__ == '__main__':
    interval = [-512,512]
    experiment_iter = 30
    energies = []
    execution_times = []
    convergences = []
    best_solution = None
    best_energy = float("Inf")

    for i in range(experiment_iter):
        print('-'*100)
        print(f'starting iter {i}:')
        initial_state = [np.random.uniform(*interval), np.random.uniform(*interval)]
        simAnneal = EggholderProblem(initial_state, interval)
    
        init_time = time()
        state, energy = simAnneal.anneal()
        end_time = time()
        execution_time = end_time - init_time
        convergence = simAnneal.energies.index(energy)
        
        energies.append(energy)
        execution_times.append(execution_time)
        convergences.append(convergence)

        if energy < best_energy:
            best_energy = energy
            best_solution = state

        # Imprimir la solución encontrada
        print("Best State: ", state)
        print("Best Energy: ", energy)
        print(f"Execution time: {execution_time:.4f}")
        print(f'Iter convergence: {convergence}')

    print()
    print('-'*100)
    print(f'Best Solution: {best_solution}')
    print(f'Best Energy: {best_energy}')
    print(f'Average Energy: {np.mean(energies)}')
    print(f'Variance: {np.var(energies)}')
    print(f'Average execution time: {np.mean(execution_times):.4f}')
    print(f'Average Iter Convergence: {np.mean(convergences)}')

    plt.plot([i for i in range(len(simAnneal.energies))], simAnneal.energies)
    plt.ylabel("Energy")
    plt.xlabel("Iteration")
    plt.title("Energy Convergence")

    x1=np.linspace(-512,512,100)
    x2=np.linspace(-512,512,100)
    X1,X2=np.meshgrid(x1,x2)
    iplot=interactive(plotter,E=(-90,90,5),A=(-90,90,5), solution=fixed((*best_solution, best_energy)))
    iplot