# import pyswarm as ps
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import numpy as np
import matplotlib.pyplot as plt
from time import time
from ipywidgets import interactive, fixed

def eggholder_function_PSO(x):
    x1, x2 = x[:, 0], x[:, 1]
    return eggholder_function([x1,x2])

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

if __name__ == '__main__':
    x_max = 512 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.992}

    experiment_iter = 20
    execution_time_history = []
    fitness_history = []
    convergences = []
    best_fitness = float("Inf")
    best_solution = None

    for i in range(experiment_iter):
        print('-'*100)
        print(f'starting iter {i}:')
        init_time = time()
        
        optimizer = GlobalBestPSO(n_particles = 10, dimensions=2, options=options, bounds=bounds)
        fitness, solution = optimizer.optimize(eggholder_function_PSO, iters=10000)
        
        end_time = time()
        execution_time = end_time - init_time
        convergence = list(map(lambda x: round(x,2),optimizer.cost_history))
        convergence = convergence.index(round(fitness,2))
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = solution

        fitness_history.append(fitness)
        execution_time_history.append(execution_time)
        convergences.append(convergence)
        
        print("\nBest Solution:", solution)
        print("Best Fitness:", fitness)
        print(f"Execution Time: {execution_time:.4f}")
        print(f"Iter convergence: {convergence}")

    print()
    print('-'*100)
    print(f'Best Solution: {best_solution}')
    print(f'Best Fitness: {best_fitness}')
    print(f'Average Fitness: {np.mean(fitness_history):.4f}')
    print(f'Fitness Variance: {np.var(fitness_history):.4f}')
    print(f'Average Execution Time: {np.mean(execution_time_history):.4f}')
    print(f'Average Iter Convergence: {np.mean(convergences)}')

    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()

    x1=np.linspace(-512,512,100)
    x2=np.linspace(-512,512,100)
    X1,X2=np.meshgrid(x1,x2)
    iplot=interactive(plotter,E=(-90,90,5),A=(-90,90,5), solution=fixed((*best_solution, best_fitness)))
    iplot