import numpy as np
import random
import copy

# -----------------------------
# 1. 辽阳园区真实矩阵
# -----------------------------

# 距离矩阵（表 8）
D = np.array([
    [0, 210, 380, 279, 90, 173, 296],
    [210, 0, 160, 174, 110, 286, 280],
    [380, 160, 0, 180, 266, 388, 385],
    [279, 174, 180, 0, 130, 170, 80],
    [90, 110, 266, 130, 0, 125, 205],
    [173, 286, 388, 170, 125, 0, 64],
    [296, 280, 385, 80, 205, 64, 0]
])

# 流量矩阵（表 9）
C = np.array([
    [0, 50000, 8000, 5300, 32000, 200, 10000],
    [4500, 0, 1400, 1300, 1200, 800, 1400],
    [5900, 500, 0, 300, 850, 10, 35],
    [1500, 85, 12000, 0, 640, 8000, 120],
    [200, 350, 9000, 830, 0, 40, 8000],
    [0, 140, 5000, 0, 0, 0, 0],
    [0, 0, 3500, 0, 0, 0, 0]
])

N = C.shape[0]

# -----------------------------
# 2. QAP 目标函数tip:用于输出总运输成本
# -----------------------------
def qap_cost(perm, C, D):
    cost = 0
    for i in range(N):
        for j in range(N):
            cost += C[i, j] * D[perm[i], perm[j]]
    return cost

# -----------------------------
# 3. 启发式初始化tip：计算每个设施的流量总和距离和，把流量大的设施放在距离近的地点。设置一个起始较优解
# -----------------------------
def heuristic_init(C, D):
    facility_score = np.sum(C, axis=1)
    position_score = np.sum(D, axis=1)
    facility_order = np.argsort(-facility_score)
    position_order = np.argsort(position_score)
    perm = np.zeros(N, dtype=int)
    for i, f in enumerate(facility_order):
        perm[f] = position_order[i]
    return perm

# -----------------------------
# 4. 局部搜索（交换法）tip：遍历所有的位置，求成本，有更低成本时更新最优解，输出局部最优排列和对应的成本
# -----------------------------
def local_search(perm, C, D):
    best_perm = perm.copy()
    best_cost = qap_cost(best_perm, C, D)
    improved = True
    while improved:
        improved = False
        for i in range(N-1):
            for j in range(i+1, N):
                new_perm = best_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_cost = qap_cost(new_perm, C, D)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_perm = new_perm
                    improved = True
                    break
            if improved:
                break
    return best_perm, best_cost

# -----------------------------
# 5. HEDA 核心算法tip：
# -----------------------------
def HEDA_QAP(C, D, pop_size=20, generations=100, perturb_rate=0.1):
    # 初始化种群tip：将启发式初始化的解代入
    population = [heuristic_init(C, D)]
    while len(population) < pop_size:
        perm = np.random.permutation(N)
        population.append(perm)
    
    # 初始化概率矩阵tip:表示设施x放在位置y的可能性，
    prob_matrix = np.ones((N,N)) / N
    
    best_perm = None
    best_cost = float('inf')
    #迭代器
    for g in range(generations):
        # 计算种群成本，每代先评估所有个体的成本，排序后，最优个体在前。
        costs = [qap_cost(p, C, D) for p in population]
        sorted_idx = np.argsort(costs)
        population = [population[i] for i in sorted_idx]
        
        # 更新全局最优，记录历史最优解。
        if costs[sorted_idx[0]] < best_cost:
            best_cost = costs[sorted_idx[0]]
            best_perm = population[0].copy()
        
        # 更新概率矩阵，如果某个设施总是在最优解里放在某个位置 → 这个位置概率增加，形成“学习概率”，引导下一代个体
        prob_matrix *= 0
        top_k = max(1, pop_size//2)
        for p in population[:top_k]:
            for f, pos in enumerate(p):
                prob_matrix[f, pos] += 1
        prob_matrix = prob_matrix / top_k
        
        # 概率扰动，防止继续陷入局部最优解
        prob_matrix = (1-perturb_rate)*prob_matrix + perturb_rate*(1/N)
        
        # 根据概率采样新个体，根据概率矩阵采样设施位置，确保每个位置只放一个设施
        new_population = []
        for _ in range(pop_size):
            perm = np.full(N, -1)
            available_positions = list(range(N))
            for f in range(N):
                probs = prob_matrix[f, available_positions]
                probs = probs / np.sum(probs)
                chosen_idx = np.random.choice(len(available_positions), p=probs)
                perm[f] = available_positions[chosen_idx]
                available_positions.pop(chosen_idx)
            perm, _ = local_search(perm, C, D)
            new_population.append(perm)
        
        population = new_population
    
    return best_perm, best_cost

# -----------------------------
# -----------------------------
# 6. 运行 HEDA 验证（位置从1开始）
# -----------------------------
best_perm, best_cost = HEDA_QAP(C, D, pop_size=30, generations=100)

# 将 Python 0-based 编号改为 1-based
best_perm_1based = best_perm + 1

print("HEDA 最优排列（位置从1开始）:", best_perm_1based)
print("HEDA 最小总成本:", best_cost)
