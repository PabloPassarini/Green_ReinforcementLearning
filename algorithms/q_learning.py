import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tsplib95
from codecarbon import track_emissions


def get_instance(instance_name):
    base_dir = Path(__file__).resolve().parent.parent
    instance_path = base_dir / 'instance' / instance_name
    return tsplib95.load(instance_path)

def reward_function(r_type, distance):
    if r_type == '1/d':
        return 1.0 / distance if distance != 0 else 0.0
    elif r_type == '-d':
        return -distance
    elif r_type == '-(d^2)':
        return -(distance ** 2)
    return None
     
def epsilon_decay(e_type, episode, total_episodes):
    if e_type == 'linear':
        return 1 - (episode / total_episodes)
    elif e_type == 'concave':
        return 0.999 ** episode
    elif e_type == 'convex':
        return - (episode / total_episodes) ** 6 + 1
    return None

@track_emissions
def train_Qlearning(r_type, e_type, matrix_d, n_points, episodes, alpha, gamma, epsilon):
    Q_table = np.zeros((n_points, n_points))    
    
    best_path = list()
    best_distance = float('inf')
    record_distance = list()

    for ep in range(episodes):
        curr_p = random.randint(0, n_points - 1) #Select a random start point
        unvisited = list(range(n_points)) #Create a list of unvisited points
        unvisited.remove(curr_p)    #Remove the start point from unvisited
        path = [curr_p] #Initialize path with start point
        curr_distance = 0

        while unvisited:
            """e-greedy policy"""
            n_a = random.uniform(0, 1)
            if n_a < epsilon: 
                next_p = random.choice(unvisited)
            else:
                q_unvisited = {point: Q_table[curr_p, point] for point in unvisited}
                next_p = max(q_unvisited, key=q_unvisited.get)
        
            d = float(matrix_d[curr_p][next_p])
            reward = reward_function(r_type, d)

            """ Bellman Equation """
            last_q = Q_table[curr_p, next_p]
            future_q = np.max(Q_table[next_p, :])
            new_q = last_q + alpha*(reward + gamma*future_q - last_q)
            Q_table[curr_p, next_p] = new_q


            curr_distance += d
            path.append(next_p)
            curr_p = next_p
            unvisited.remove(next_p)
        
        last_p = path[-1]
        curr_distance += float(matrix_d[last_p][path[0]]) #Return to start point
        path.append(path[0]) #Complete the cycle

        record_distance.append(curr_distance)
        if curr_distance < best_distance:
            best_distance = curr_distance
            best_path = path.copy()

        epsilon = epsilon_decay(e_type, ep, episodes)
        

    print("\n--- Resultados ---")
    print("O melhor caminho encontrado durante o treinamento foi:")
    print(f"Caminho: {' -> '.join(map(str, best_path))}")
    print(f"Distância Total: {best_distance:.2f}")

instance_names = ['berlin52.tsp', 'br17.atsp', 'eil51.tsp', 'ftv33.atsp', 'ftv64.atsp', 'kroA100.tsp', 'st70.tsp', 'tsp225.tsp']
instance_folder = 'instance'

alpha = 0.01 #Learning rate
gamma = 0.15 #Discount factor
epsilon = 1.0 #Exploration rate

epsilon_decay_types = ['linear', 'concave', 'convex']
reward_types = ['1/d', '-d', '-(d^2)']

for instance_name in instance_names:
    instance_path = os.path.join(instance_folder, instance_name)
    problem = get_instance(instance_path)

    dist_matrix = np.array([
        [problem.get_weight(i, j) for j in problem.get_nodes()]
        for i in problem.get_nodes()
    ])
    n_points = problem.dimension

    for e_type in epsilon_decay_types:
        for r_type in reward_types:
            print(f"Training on instance '{instance_name}' with epsilon='{e_type}' and reward='{r_type}'")
            train_Qlearning(
                r_type=r_type,
                e_type=e_type,
                matrix_d=dist_matrix,
                n_points=n_points,
                episodes=10000,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon
            )
