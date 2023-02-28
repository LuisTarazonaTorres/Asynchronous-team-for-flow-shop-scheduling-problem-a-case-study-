import random
import math
import numpy as np
from multiprocessing import Process, Manager

# Define the solution representation
class Solution:
    def __init__(self, job_order):
        self.job_order = job_order # Order in which jobs are processed
        self.makespan = self.evaluate() # Makespan of the solution
        
    def evaluate(self):
        # Implement a function that calculates the makespan of a solution  

        completion_times = np.zeros((num_jobs, num_machines))
        job_order = self.job_order[:]
        for i in range(num_jobs):
            for j in range(num_machines):
                if i == 0 and j == 0:
                    completion_times[i,j] = processing_time[job_order[i],j]
                elif i == 0:
                    completion_times[i,j] = completion_times[i,j-1] + processing_time[job_order[i],j]
                elif j == 0:
                    completion_times[i,j] = completion_times[i-1,j] + processing_time[job_order[i],j]
                else:
                    completion_times[i,j] = max(completion_times[i,j-1], completion_times[i-1,j]) + processing_time[job_order[i],j]
        return completion_times[-1,-1]
    
# Define the search operators
def perturbate_shared_best_solution():
    # Function that perturbates the shared best solution
    job_order = list(range(num_jobs))
    random.shuffle(job_order)
    return Solution(job_order)

def swap_jobs(solution):
    # Function that swaps two jobs in a solution
    job_order = solution.job_order[:]
    i, j = random.sample(range(num_jobs), 2)
    job_order[i], job_order[j] = job_order[j], job_order[i]
    return Solution(job_order)

def insert_job(solution):
    # Function that inserts a job at a random position in a solution
    job_order = solution.job_order[:]
    i, j = random.sample(range(num_jobs), 2)
    job_order.insert(i, job_order.pop(j))
    return Solution(job_order)

def two_opt(solution):
    # Function that swaps two jobs in a solution using 2-OPT
    job_order = solution.job_order[:]
    i, j = random.sample(range(num_jobs), 2)
    if i > j:
        i, j = j, i
    job_order[i:j+1] = reversed(job_order[i:j+1])
    return Solution(job_order)

# Single Tabu Algorithm with Shared Memory
def local_tabu_search(shared_best_solution):
    current_solution = perturbate_shared_best_solution()
    best_current_solution = current_solution
    best_neighbor_cost = np.inf
    for it in range(MAX_LOCALITER):
        cambio = False
        for i in range(MAX_NEIGHBORS):
            neighbor_solution_swap = swap_jobs(current_solution)
            neighbor_cost_swap = neighbor_solution_swap.makespan
            if  neighbor_cost_swap < current_solution.makespan:
                current_solution = neighbor_solution_swap
                cambio = True
            else:
                if neighbor_cost_swap < best_neighbor_cost:
                    best_neighbor_cost = neighbor_cost_swap
                    best_neighbor = neighbor_solution_swap
                neighbor_solution_insert = insert_job(current_solution)
                neighbor_cost_insert = neighbor_solution_insert.makespan
                if neighbor_cost_insert < current_solution.makespan:
                    current_solution = neighbor_solution_insert
                    cambio = True
                else:
                    if neighbor_cost_insert < best_neighbor_cost:
                        best_neighbor_cost = neighbor_cost_insert
                        best_neighbor = neighbor_solution_insert
                    neighbor_solution_two_opt = two_opt(current_solution)
                    neighbor_cost_two_opt = neighbor_solution_two_opt.makespan
                    if neighbor_cost_two_opt < current_solution.makespan:
                        current_solution = neighbor_solution_two_opt
                        cambio = True
                    else:
                        if neighbor_cost_two_opt < best_neighbor_cost:
                            best_neighbor_cost = neighbor_cost_two_opt
                            best_neighbor = neighbor_solution_two_opt
            if current_solution.makespan < best_current_solution.makespan:
                best_current_solution = current_solution
        if cambio == False:
            current_solution = best_neighbor
            
    return best_current_solution

# Asynchronous Parallel Process of Tabu Algorithms
def parallel_local_search(shared_best_solution):
    local_best_solution = local_tabu_search(shared_best_solution)
    if local_best_solution.makespan < shared_best_solution.makespan:
        shared_best_solution.job_order = local_best_solution.job_order[:]
        shared_best_solution.makespan = local_best_solution.makespan

# NEH Constructive Algorithm
def constructive_heuristic():
    # Initialize the job order and start times
    job_order = np.zeros(num_jobs, dtype=int)
    start_times = np.zeros((num_jobs, num_machines), dtype=int)

    # Calculate the sum of processing times for each job and sort the jobs by increasing sum
    job_sums = np.sum(processing_time, axis=1)
    sorted_jobs = np.argsort(job_sums)

    # Assign the first job to the first machine and update the start times
    job_order[0] = sorted_jobs[0]
    start_times[0, 0] = processing_time[job_order[0], 0]

    # Assign the remaining jobs to the machine that has the earliest available time
    for i in range(1, num_jobs):
        job = sorted_jobs[i]
        min_time = np.min(start_times[i-1])
        min_machine = np.argmin(start_times[i-1])
        job_order[i] = job
        start_times[i, min_machine] = min_time + processing_time[job, min_machine]

    return job_order

# Asynchronous team of metaheuristics
def asynchronous_team_of_metaheuristics():
    global best_known_solution
    global constructivo
    manager = Manager()
    shared_best_solution = manager.Namespace()
    shared_best_solution.job_order = np.random.permutation(num_jobs)
    shared_best_solution.job_order = constructive_heuristic()
    constructivo = shared_best_solution.job_order
    best_known_solution = Solution(shared_best_solution.job_order).makespan
    print(best_known_solution)
    #random.shuffle(shared_best_solution.job_order)
    best_known_solution = Solution(shared_best_solution.job_order).makespan
    shared_best_solution.makespan = best_known_solution
    for iter_global in range(1,MAX_ITER):

        # Improve the local solution using parallel local search
        processes = []
        for i in range(MAX_THREADS): # Number of parallel instances
            p = Process(target=parallel_local_search, args=(shared_best_solution,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        #processes = []

        # Update the best solution found
        if shared_best_solution.makespan < best_known_solution:
            best_known_solution = shared_best_solution.makespan
    return Solution(shared_best_solution.job_order)

# Main function
# Define the flowshop problem instance file
id_instance = 81
if id_instance < 10:
    id = "00" + str(id_instance)
else:
    id = "0" + str(id_instance)
path_name_file = "C:\\Users\\in666\\Universidad de los Andes\\Luis Enrique Tarazona Torres - artículo\\input_files\\tai"+id+".txt"

# Open the instance file and read, num_jobs, num_machines, and processing_times
with open(path_name_file, 'r') as f:
    # Read the number of jobs and machines
    num_jobs, num_machines = map(int, f.readline().split())

    # Read the processing times for each job and machine
    processing_time = np.zeros((num_jobs, num_machines), dtype=int)
    for i in range(num_jobs):
        processing_time[i] = list(map(int, f.readline().split()))

# Set the parameters of the Asynchronous team of metaheuristics MAX_ITER = total number of global iterations, MAX_LOCALITER = total number of local search iterations
# MAX_NEIGHBORS = total number of random neighbors explored in a neighborhood and MAX_THREADS = total number of asynchronous metaheuristics
MAX_ITER = 15
MAX_LOCALITER = 100
MAX_NEIGHBORS = num_jobs
MAX_THREADS = 20


best_known_solution = math.inf # Best known solution for the problem instance

if __name__ == '__main__':
    # Run the asynchronous team of metaheuristics
    best_solution = asynchronous_team_of_metaheuristics()

    # Print the best solution found
    print("Best solution:", best_solution.job_order)
    print("Makespan:", best_solution.makespan)