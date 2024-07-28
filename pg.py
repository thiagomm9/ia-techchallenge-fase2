# -*- coding: utf-8 -*-

import vars
import pygame
import random

pygame.init()

# Configura a janela
width, height = 800, 800
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Network GA Visualization")

# Configura as cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Função para desenhar a rede
def draw_network(adjacency_matrix, node_positions):
    window.fill(WHITE)
    num_nodes = len(adjacency_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] == 1:
                pygame.draw.line(window, BLACK, node_positions[i], node_positions[j], 2)
    for pos in node_positions:
        pygame.draw.circle(window, RED, pos, 5)
    pygame.display.flip()

# Função para geras posições aleatórias para os nodes
def generate_node_positions(num_nodes, width, height):
    positions = []
    margin = 50
    for _ in range(num_nodes):
        x = random.randint(margin, width - margin)
        y = random.randint(margin, height - margin)
        positions.append((x, y))
    return positions


# Gera as posições
node_positions = generate_node_positions(vars.num_nodes, width, height)
