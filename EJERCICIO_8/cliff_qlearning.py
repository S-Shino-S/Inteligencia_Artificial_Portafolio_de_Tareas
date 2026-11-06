import random
import matplotlib.pyplot as plt

# Parámetros del experimento
ALPHA = 0.5
GAMMA = 1.0
EPSILON = 0.1
EPISODES = 500
MAX_STEPS = 200
SEED = 42

# Configurar semilla para reproducibilidad
random.seed(SEED)

# Definición del entorno
ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)
ACTIONS = ['N', 'E', 'S', 'W']
# Movimientos: (delta_row, delta_col)
MOVES = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}

def is_cliff(state):
    return state[0] == 3 and 1 <= state[1] <= 10

def step(state, action):
    r, c = state
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc
    
    # Chocar con los bordes (se queda en el mismo lugar)
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        nr, nc = r, c
        
    next_state = (nr, nc)
    
    # Evaluar transiciones
    if next_state == GOAL:
        return next_state, -1, True
    elif is_cliff(next_state):
        # Cae al acantilado: castigo fuerte, vuelve a START, pero el episodio NO termina
        return START, -100, False
    else:
        # Transición normal
        return next_state, -1, False

# Inicializar tabla Q en 0
# Q[estado][acción] -> float
Q = {}
for r in range(ROWS):
    for c in range(COLS):
        Q[(r, c)] = {a: 0.0 for a in ACTIONS}

def get_action(state, epsilon):
    """Política epsilon-greedy con orden fijo de desempate (N, E, S, W)"""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        # Encontrar el valor máximo
        max_q = max(Q[state].values())
        # Desempate con orden fijo: devuelve la primera acción de la lista que tenga el max_q
        for a in ACTIONS:
            if Q[state][a] == max_q:
                return a

# --- ENTRENAMIENTO ---
rewards_per_episode = []

for ep in range(EPISODES):
    state = START
    total_reward = 0
    
    for _ in range(MAX_STEPS):
        action = get_action(state, EPSILON)
        next_state, reward, terminal = step(state, action)
        
        # Backup de Q-learning (off-policy usa el max del siguiente estado)
        if terminal:
            max_next_q = 0.0
        else:
            max_next_q = max(Q[next_state].values())
            
        td_target = reward + GAMMA * max_next_q
        Q[state][action] = Q[state][action] + ALPHA * (td_target - Q[state][action])
        
        total_reward += reward
        state = next_state
        
        if terminal:
            break
            
    rewards_per_episode.append(total_reward)

# --- EVALUACIÓN GREEDY (ε = 0) ---
state = START
greedy_path = [START]
greedy_actions = []
greedy_return = 0

for _ in range(MAX_STEPS):
    action = get_action(state, epsilon=0.0) # Greedy
    next_state, reward, terminal = step(state, action)
    
    greedy_actions.append(action)
    greedy_path.append(next_state)
    greedy_return += reward
    state = next_state
    
    if terminal:
        break

# Imprimir resultados
print(f"Retorno de la política greedy: {greedy_return}")
print(f"Pasos tomados: {len(greedy_actions)}")

print("\nPolítica Greedy sobre la cuadrícula:")
# Símbolos para imprimir el mapa
arrows = {'N': '↑', 'E': '→', 'S': '↓', 'W': '←'}
for r in range(ROWS):
    row_str = ""
    for c in range(COLS):
        if (r, c) == GOAL:
            row_str += " G "
        elif is_cliff((r, c)):
            row_str += " C "
        elif (r, c) == START:
            best_a = get_action((r, c), 0.0)
            row_str += f" {arrows[best_a]} "
        else:
            best_a = get_action((r, c), 0.0)
            row_str += f" {arrows[best_a]} "
    print(row_str)

# --- GRÁFICA DE RECOMPENSAS ---
def moving_average(data, window_size=10):
    return [sum(data[i:i+window_size])/window_size for i in range(len(data) - window_size + 1)]

smoothed_rewards = moving_average(rewards_per_episode, 10)
plt.plot(rewards_per_episode, alpha=0.3, color='gray', label='Recompensa cruda')
plt.plot(range(9, len(rewards_per_episode)), smoothed_rewards, color='blue', label='Media móvil (n=10)')
plt.ylim([-300, 0])
plt.title("Q-learning en Cliff Walking (ε=0.1)")
plt.xlabel("Episodios")
plt.ylabel("Recompensa acumulada")
plt.legend()
plt.grid(True)
plt.savefig("Q-learning/ejercicios/Solucion-01/grafica.png")