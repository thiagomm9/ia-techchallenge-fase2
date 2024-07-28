import ga
import vars
import pg

top_fitnesses, top_individuals = [], []

### GERA POPULAÇÃO INICIAL
population = ga.generate_heuristic_initial_population()
#population = ga.generate_random_initial_population()


### ITERA NAS GERAÇÕES
running = True
generation = 0
while running and generation < vars.generations:
    for event in pg.pygame.event.get():
        if event.type == pg.pygame.QUIT:
            running = False
    
    fitnesses = [ga.calculate_fitness(individual) for individual in population]
    new_population = []
    
    # Separa top por cento da população
    top_individuals, top_fitnesses = ga.select_top_percent_population(population, fitnesses, 0.3)
    
    print(f"Generation {generation}: Melhor fitness: {top_fitnesses[0]}")
    
    # Mantém o melhor indivíduo
    new_population.append(top_individuals[0])
    
    # Gerando nova população
    while len(new_population) < vars.population_size:
        # SELEÇÃO
        parent1 = ga.tournament_selection(top_individuals, top_fitnesses, vars.tournament_size)
        parent2 = ga.tournament_selection(top_individuals, top_fitnesses, vars.tournament_size)
        
        # CROSSOVER
        # Crossover binário convencional, não progride pois não preserva a conectividade dos pais
        #child1, child2 = ga.crossover(parent1, parent2)
        
        # Crossover binário convencional, porém com garantia de conectividade, mas também não progride sifnificativamente
        #child1 = ga.crossover_connected(parent1, parent2)
        #child2 = ga.crossover_connected(parent1, parent2)
        
        # Crossover por Edge Combination
        child1 = ga.edge_recombination_crossover(parent1, parent2)
        child2 = ga.edge_recombination_crossover(parent1, parent2)
        
        # MUTAÇÃO
        # A mutação atrapalha o progresso, por geralmente desconectar a rede ou adicionar links desnecessários
        child1 = ga.mutate(child1)
        child2 = ga.mutate(child2)
        
        new_population.extend([child1, child2])
    
    population = new_population
    best_am = ga.chromosome_to_adjacency_matrix(top_individuals[0])
    pg.draw_network(best_am, pg.node_positions)
    generation += 1

pg.pygame.quit()    


print(f"Melhor fitness: {top_fitnesses[0]}")

