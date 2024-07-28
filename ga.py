# -*- coding: utf-8 -*-
import numpy as np
import vars
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from collections import defaultdict
import random


# Função que converte indivíduo para matriz de adjacência
def chromosome_to_adjacency_matrix(chromosome):
    #chromosome_example = [1,0,0,0,0,0] # 4 nodes 6 links -> [0x1,0x2,0x3,1x2,1x3,2x3]
    #chromosome_example = [1,0,0,0,0,0,1,0,0,0,0,0] # 5 nodes 10 links -> [0x1,0x2,0x3,0x4,1x2,1x3,1x4,2x3,2x4,3x4]
    adjacency_matrix = np.zeros((vars.num_nodes, vars.num_nodes))
    index = 0
    for i in range(vars.num_nodes):
        for j in range(i + 1, vars.num_nodes):
            if chromosome[index] == 1:
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
            index += 1
    return adjacency_matrix


# Função que calcula o custo total do indivíduo
# Cada link possui um custo associado, e quando ativo tem o custo somado
# Cada link conecta dois nodes
def calculate_total_cost(chromosome):
    total_cost = 0
    for index, gene in enumerate(chromosome):
        if gene == 1:
            total_cost += vars.link_costs[vars.num_nodes][index]
    return total_cost


# Função que confere se a rede(indivíduo) está totalmente conectada
# Para isso, todos os nodes da rede devem possuir ao menos 1 link ativo conectando a outro node
def is_connected(adjacency_matrix):
    graph = csr_matrix(adjacency_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return n_components == 1


# Calcula o fitness da rede(indivíduo)
# Corresponde a soma do custo total, latência total, penalidade de largura de banda e penalidade de falta de conexão
# Quanto menor o valor melhor
def calculate_fitness(chromosome):
    # Custo da rede
    total_cost = calculate_total_cost(chromosome)
    # Confere a largura de banda requerida, aplicando penalidade quando necessário
    bandwidth_penalty = 0
    for index, gene in enumerate(chromosome):
        if gene == 1 and vars.link_bandwidths[vars.num_nodes][index] < vars.min_bandwidth:
            bandwidth_penalty += 200
    # Calcula latencia total
    total_latency = 0
    for index, gene in enumerate(chromosome):
        if gene == 1:
            total_latency += vars.link_latencies[vars.num_nodes][index]
    # Penalidade por falta de conectividade
    conn_penalty = 0
    adjacency_matrix = chromosome_to_adjacency_matrix(chromosome)
    if not is_connected(adjacency_matrix):
        conn_penalty += 5000
    # Somando o valor de aptidão
    return total_cost + total_latency + bandwidth_penalty + conn_penalty

# Converte matriz de adjacência para indivíduo
def adjacency_matrix_to_chromosome(adjacency_matrix):
    chromosome = []
    for i in range(vars.num_nodes):
        for j in range(i + 1, vars.num_nodes):
            chromosome.append(int(adjacency_matrix[i, j]))
    return chromosome


# Gera população inicial aleatóriamente
def generate_random_initial_population():
    population = []
    for _ in range(vars.population_size):
        chromosome = [random.randint(0, 1) for _ in range(vars.num_links)]
        population.append(chromosome)
    return population


# Gera matriz de uma minimum spanning tree
# Serve como um indivíduo viável de partida para o algoritmo genético
def mst_adjacency_matrix():
    # Cria o grafo completo
    complete_graph = np.full((vars.num_nodes, vars.num_nodes), np.inf)
    index = 0
    # Preenche o grafo com o custo dos links
    for i in range(vars.num_nodes):
        for j in range(i + 1, vars.num_nodes):
            complete_graph[i, j] = complete_graph[j, i] = vars.link_costs[vars.num_nodes][index]
            index += 1
    # Gera a minimum spanning tree
    mst = minimum_spanning_tree(complete_graph).toarray()
    # Converte pra binário
    mst[mst != 0] = 1  # Conversão binária
    return mst


# Gera população inicial utilizando método heurístico com Minimum spanning tree
def generate_heuristic_initial_population():
    population = []
    for _ in range(vars.population_size):
        mst_matrix = mst_adjacency_matrix()
        chromosome = adjacency_matrix_to_chromosome(mst_matrix)
        # Adiciona mutações aleatórias ao MST
        for _ in range(random.randint(1, 3)):
            index = random.randint(0, vars.num_links - 1)
            chromosome[index] = 1 - chromosome[index]  # Alterna o bit
        population.append(chromosome)
    return population


# Seleciona os melhores indivíduos da população, baseado na porcentagem definida
def select_top_percent_population(population, fitnesses, top_percent):
    num_top_individuals = int(len(population) * top_percent)
    fitness_population_pairs = list(zip(fitnesses, population))
    sorted_population = sorted(fitness_population_pairs, key=lambda x: x[0])
    top_individuals = [ind for fit, ind in sorted_population[:num_top_individuals]]
    top_fitnesses = [fit for fit, ind in sorted_population[:num_top_individuals]]
    return top_individuals, top_fitnesses


# Executa seleção por torneio, selecionando os individuos da população aleatóriamente
def tournament_selection(population, fitnesses, tournament_size):
    if tournament_size > len(population):
        tournament_size = len(population)
    
    tournament_indices = random.sample(range(len(population)), tournament_size)
    #tournament_individuals = [population[i] for i in tournament_indices]
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    
    best_index = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
    
    return population[best_index]


# Executa mutação do indivíduo, flipando o bit dentro da porcentagem definida em vars
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < vars.mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


# Cruza dois indivíduos para gerar dois novos indivíduos, dividindo em um ponto aleatório
def crossover(parent1, parent2):
    crossover_point = random.randint(1, vars.num_links - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Cruza dois indivíduos para gerar um novo indivíduo, dividindo em um ponto aleatório e garantindo conectividade
def crossover_connected(parent1, parent2):
    crossover_point = random.randint(1, vars.num_links - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    am = chromosome_to_adjacency_matrix(child)
    while (not is_connected(am)):
        crossover_point = random.randint(1, vars.num_links - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        am = chromosome_to_adjacency_matrix(child)
    return child


# Cruzamneto que combina os links de dois indivíduos preservando a adjacência(conectividade) da rede
def edge_recombination_crossover(parent1, parent2):
    adjacency_list = defaultdict(set)
    
    # Converte para am
    adj_matrix1 = chromosome_to_adjacency_matrix(parent1)
    adj_matrix2 = chromosome_to_adjacency_matrix(parent2)
    
    # Constrói a lista combinando os dois pais
    for i in range(vars.num_nodes):
        for j in range(i + 1, vars.num_nodes):
            if adj_matrix1[i, j] == 1:
                adjacency_list[i].add(j)
                adjacency_list[j].add(i)
            if adj_matrix2[i, j] == 1:
                adjacency_list[i].add(j)
                adjacency_list[j].add(i)
    
    # Gera o filho usando a lista
    current_node = random.choice(range(vars.num_nodes))
    visited = {current_node}
    child_edges = []
    while len(visited) < vars.num_nodes:
        neighbors = list(adjacency_list[current_node] - visited)
        if neighbors:
            next_node = random.choice(neighbors)
        else:
            next_node = random.choice(list(set(range(vars.num_nodes)) - visited))
        
        child_edges.append((current_node, next_node))
        visited.add(next_node)
        current_node = next_node
    
    # Converte o filho para cromosomo
    child = [0] * (vars.num_nodes * (vars.num_nodes - 1) // 2)
    edge_index = 0
    for i in range(vars.num_nodes):
        for j in range(i + 1, vars.num_nodes):
            if (i, j) in child_edges or (j, i) in child_edges:
                child[edge_index] = 1
            edge_index += 1
    
    return child

