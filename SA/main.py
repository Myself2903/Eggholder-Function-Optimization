from SA import SimAnneal as SA
import time
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, fixed

def f(x, y):
    return (-(y + 47.0) * np.sin(np.sqrt(np.abs(x/2.0 + (y + 47.0)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47.0)))))

def plotter(E,A, solution):
    x, y, z = solution
    fig=plt.figure(figsize=[12,8])
    ax=plt.axes(projection='3d')
    ax.plot_surface(X1,X2,f(X1,X2),color='red',alpha=0.7)
    ax.plot_wireframe(X1,X2,f(X1,X2),ccount=2,rcount=2, color='orange',alpha=0.8)   
    ax.scatter(x, y, z, color='blue', s=100)
    ax.view_init(elev=E,azim=A)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    plt.show()

if __name__ == '__main__':
    ex_times = 20
    best_solutions = []
    best_fitnesses = []
    n_iters = []
    end_times = []
    iter_time = []

    for i in range(ex_times):
        print('Ejecución No. ', i)
        start_time = time.time()
        sa = SA()
        best_solution, best_fitness , n_iter = sa.anneal()
        stop_time = time.time()
        end_time = stop_time - start_time
        print(f" running_time: {end_time:.8f}")
        end_times.append(end_time)
        best_solutions.append(best_solution)
        best_fitnesses.append(best_fitness)
        n_iters.append(n_iter)
        iter_time.append(end_time)
        print(f' Solution found in iter: {n_iter}')
    
    var = np.var(best_fitnesses)
    best_fitness = min(best_fitnesses)
    best_solution = (*best_solutions[best_fitnesses.index(best_fitness)], best_fitness)
    
    print(f'\nAverage z: {np.mean(best_fitnesses):.4f}')
    print("Variance: ", round(var,2))
    print(f"Best solution found in  {ex_times} iter : {best_fitness:.4f}")
    print(f"Best solution found: {tuple(round(i,4) for i in best_solution)}")
    print(f'average execution time: {np.mean(iter_time):.8f}')

    sa.plot_learning()

    x1=np.linspace(-512,512,100)
    x2=np.linspace(-512,512,100)
    X1,X2=np.meshgrid(x1,x2)
    iplot=interactive(plotter,E=(-90,90,5),A=(-90,90,5), solution=fixed(best_solution))
    iplot

    