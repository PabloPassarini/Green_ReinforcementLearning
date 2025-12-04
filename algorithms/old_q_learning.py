import random
from pathlib import Path

import numpy as np
import pandas as pd
import tsplib95
from codecarbon import track_emissions, EmissionsTracker, OfflineEmissionsTracker


def get_instance(filename):
    base_dir = Path(__file__).resolve().parent.parent
    instance_path = base_dir / 'instances' / filename
    return tsplib95.load(instance_path)

def reward_function(r_type, distance):
    if r_type == 'R1':
        return 1.0 / distance if distance != 0 else 0.0
    elif r_type == 'R2':
        return -distance
    elif r_type == 'R3':
        return -(distance ** 2)
    return None
     
def epsilon_decay(e_type, episode, total_episodes):
    if e_type == 'linear':
        return 1 - (episode / total_episodes)
    elif e_type == 'concave':
        return 0.999 ** episode
    elif e_type == 'convex':
        return - (episode / total_episodes) ** 6 + 1
    elif e_type == 'step':
        return max(1.0 - 0.1 * (episode // 1000), 0.0)
    return None

def train_sarsa(instance, r_type, e_type, matrix_d, n_points, episodes, alpha, gamma, epsilon):
    q_table = np.zeros((n_points, n_points))

    best_path = []
    best_distance = float('inf')
    distance_history = []

    results_dir = Path('results')
    base_name = f"{instance}_{r_type}_{e_type}_{gamma}"

    tracker = EmissionsTracker(
        project_name="q_learning_tsp",
        output_dir=results_dir,
        output_file=f"{base_name}_emissions.csv"
    )
    tracker.start()

    for ep in range(episodes):
        # Select a random starting point
        current_point = random.randint(0, n_points - 1)
        unvisited = list(range(n_points))
        unvisited.remove(current_point)
        path = [current_point]
        current_distance = 0

        while unvisited:
            # ε-greedy policy
            if random.uniform(0, 1) < epsilon:
                next_point = random.choice(unvisited)
            else:
                q_values = {point: q_table[current_point, point] for point in unvisited}
                next_point = max(q_values, key=q_values.get)

            distance = float(matrix_d[current_point][next_point])
            reward = reward_function(r_type, distance)

            if len(unvisited) > 1:
                unvisited_next = [p for p in unvisited if p != next_point]
                # Escolhe a próxima ação (ε-greedy novamente)
                if random.uniform(0, 1) < epsilon:
                    next_next_point = random.choice(unvisited_next)
                else:
                    q_values_next = {p: q_table[next_point, p] for p in unvisited_next}
                    next_next_point = max(q_values_next, key=q_values_next.get)

                future_q = q_table[next_point, next_next_point]
            else:
                future_q = 0

            # Atualização SARSA
            last_q = q_table[current_point, next_point]
            new_q = last_q + alpha * (reward + gamma * future_q - last_q)
            q_table[current_point, next_point] = new_q

            current_distance += distance
            path.append(next_point)
            current_point = next_point
            unvisited.remove(next_point)

        # Return to starting point to complete the cycle
        last_point = path[-1]
        current_distance += float(matrix_d[last_point][path[0]])
        path.append(path[0])

        distance_history.append(current_distance)
        if current_distance < best_distance:
            best_distance = current_distance
            best_path = path.copy()

        epsilon = epsilon_decay(e_type, ep, episodes)
    
    emissions_kg = tracker.stop()

    best_episode = distance_history.index(best_distance)

    results_df = pd.DataFrame({
        'Episode': list(range(episodes)),
        'Distance': distance_history,
    })

    results_filename = f"results/{base_name}_results.csv"
    results_df.to_csv(results_filename, index=False)

    # Salvar o tempo de execução
    summary_df = pd.DataFrame([{
        "Instance": instance_name,
        "RewardType": r_type,
        "EpsilonDecay": e_type,
        "Gamma": gamma,
        "BestEpisode": best_episode,
        "BestDistance": best_distance,
        "BestPath": " -> ".join(map(str, best_path)),
        "Emissions_kgCO2": emissions_kg
    }])

    summary_filename = f"results/{base_name}_summary.csv"
    summary_df.to_csv(summary_filename, index=False)

def train_q_learning(instance, r_type, e_type, matrix_d, n_points, episodes, alpha, gamma, epsilon):
    q_table = np.zeros((n_points, n_points))

    best_path = []
    best_distance = float('inf')
    distance_history = []

    results_dir = Path('results')
    base_name = f"{instance}_{r_type}_{e_type}_{gamma}"

    tracker = EmissionsTracker(
        project_name="q_learning_tsp",
        output_dir=results_dir,
        output_file=f"{base_name}_emissions.csv"
    )
    tracker.start()

    for ep in range(episodes):
        # Select a random starting point
        current_point = random.randint(0, n_points - 1)
        unvisited = list(range(n_points))
        unvisited.remove(current_point)
        path = [current_point]
        current_distance = 0

        while unvisited:
            # ε-greedy policy
            if random.uniform(0, 1) < epsilon:
                next_point = random.choice(unvisited)
            else:
                q_values = {point: q_table[current_point, point] for point in unvisited}
                next_point = max(q_values, key=q_values.get)

            distance = float(matrix_d[current_point][next_point])
            reward = reward_function(r_type, distance)

            # Bellman update
            last_q = q_table[current_point, next_point]
            future_q = np.max(q_table[next_point, :])
            new_q = last_q + alpha * (reward + gamma * future_q - last_q)
            q_table[current_point, next_point] = new_q

            current_distance += distance
            path.append(next_point)
            current_point = next_point
            unvisited.remove(next_point)

        # Return to starting point to complete the cycle
        last_point = path[-1]
        current_distance += float(matrix_d[last_point][path[0]])
        path.append(path[0])

        distance_history.append(current_distance)
        if current_distance < best_distance:
            best_distance = current_distance
            best_path = path.copy()

        epsilon = epsilon_decay(e_type, ep, episodes)
    
    emissions_kg = tracker.stop()

    best_episode = distance_history.index(best_distance)

    results_df = pd.DataFrame({
        'Episode': list(range(episodes)),
        'Distance': distance_history,
    })

    results_filename = f"results/{base_name}_results.csv"
    results_df.to_csv(results_filename, index=False)

    # Salvar o tempo de execução
    summary_df = pd.DataFrame([{
        "Instance": instance_name,
        "RewardType": r_type,
        "EpsilonDecay": e_type,
        "Gamma": gamma,
        "BestEpisode": best_episode,
        "BestDistance": best_distance,
        "BestPath": " -> ".join(map(str, best_path)),
        "Emissions_kgCO2": emissions_kg
    }])

    summary_filename = f"results/{base_name}_summary.csv"
    summary_df.to_csv(summary_filename, index=False)

def train_dqn(instance, r_type, e_type, matrix_d, n_points, episodes, alpha, gamma, epsilon):
    q1_table = np.zeros((n_points, n_points))
    q2_table = np.zeros((n_points, n_points))

    best_path = []
    best_distance = float('inf')
    distance_history = []

    results_dir = Path('results')
    base_name = f"{instance}_{r_type}_{e_type}_{gamma}"

    tracker = EmissionsTracker(
        project_name="dqn_tsp",
        output_dir=results_dir,
        output_file=f"{base_name}_emissions.csv"
    )
    tracker.start()

    for ep in range(episodes):
        # Select a random starting point
        current_point = random.randint(0, n_points - 1)
        unvisited = list(range(n_points))
        unvisited.remove(current_point)
        path = [current_point]
        current_distance = 0

        while unvisited:
            # ε-greedy com a média das duas tabelas
            if random.uniform(0, 1) < epsilon:
                next_point = random.choice(unvisited)
            else:
                q_values = {p: (q1_table[current_point, p] + q2_table[current_point, p]) / 2 for p in unvisited}
                next_point = max(q_values, key=q_values.get)

            distance = float(matrix_d[current_point][next_point])
            reward = reward_function(r_type, distance)

            if random.random() < 0.5:
                # Atualiza Q1 usando Q2
                best_action = np.argmax(q1_table[next_point, :])
                target = reward + gamma * q2_table[next_point, best_action]
                q1_table[current_point, next_point] += alpha * (target - q1_table[current_point, next_point])
            else:
                # Atualiza Q2 usando Q1
                best_action = np.argmax(q2_table[next_point, :])
                target = reward + gamma * q1_table[next_point, best_action]
                q2_table[current_point, next_point] += alpha * (target - q2_table[current_point, next_point])

            current_distance += distance
            path.append(next_point)
            current_point = next_point
            unvisited.remove(next_point)

        # Fechar o ciclo
        last_point = path[-1]
        current_distance += float(matrix_d[last_point][path[0]])
        path.append(path[0])

        distance_history.append(current_distance)
        if current_distance < best_distance:
            best_distance = current_distance
            best_path = path.copy()

        epsilon = epsilon_decay(e_type, ep, episodes)
    
    emissions_kg = tracker.stop()
    best_episode = distance_history.index(best_distance)

    results_df = pd.DataFrame({
        'Episode': list(range(episodes)),
        'Distance': distance_history,
    })

    results_filename = f"results/{base_name}_results.csv"
    results_df.to_csv(results_filename, index=False)

    summary_df = pd.DataFrame([{
        "Instance": instance,
        "RewardType": r_type,
        "EpsilonDecay": e_type,
        "Gamma": gamma,
        "BestEpisode": best_episode,
        "BestDistance": best_distance,
        "BestPath": " -> ".join(map(str, best_path)),
        "Emissions_kgCO2": emissions_kg
    }])

    summary_filename = f"results/dqn_{base_name}_summary.csv"
    summary_df.to_csv(summary_filename, index=False)



'''instance_names = [
    'berlin52.tsp', 'br17.atsp', 'eil51.tsp', 'ftv33.atsp',
    'ftv64.atsp', 'kroA100.tsp', 'st70.tsp', 'tsp225.tsp'
]'''
instance_names = ['br17.atsp']
instance_folder = 'instances'

learning_rate = 0.01  # Learning rate
epsilon = 1.0  # Exploration rate

epsilon_decay_types = ['linear', 'concave', 'convex', 'step']
reward_types = ['R1', 'R2', 'R3']
gamma_set = [0.01, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.99]  # Discount factors

for instance_name in instance_names:
    problem = get_instance(instance_name)

    dist_matrix = np.array([
        [problem.get_weight(i, j) for j in problem.get_nodes()]
        for i in problem.get_nodes()
    ])
    n_points = problem.dimension

    for gamma in gamma_set:
        for e_type in epsilon_decay_types:
            for r_type in reward_types:
                print(f"Training on instance '{instance_name}' with epsilon='{e_type}' and reward='{r_type}'")
                '''train_q_learning(
                    instance=instance_name,
                    r_type=r_type,
                    e_type=e_type,
                    matrix_d=dist_matrix,
                    n_points=n_points,
                    episodes=10000,
                    alpha=learning_rate,
                    gamma=gamma,
                    epsilon=epsilon
                )'''
                '''train_sarsa(
                    instance=instance_name,
                    r_type=r_type,
                    e_type=e_type,
                    matrix_d=dist_matrix,
                    n_points=n_points,
                    episodes=10000,
                    alpha=learning_rate,
                    gamma=gamma,
                    epsilon=epsilon
                )'''

                train_dqn(
                    instance=instance_name,
                    r_type=r_type,
                    e_type=e_type,
                    matrix_d=dist_matrix,
                    n_points=n_points,
                    episodes=10000,
                    alpha=learning_rate,
                    gamma=gamma,
                    epsilon=epsilon
                )