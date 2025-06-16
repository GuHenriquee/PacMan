# seuPacManAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Um agente minimax que escolhe uma ação em cada ponto de decisão examinando
    suas alternativas através de uma função de avaliação de estado.
    """
    def getAction(self, gameState: GameState):
        """
        Retorna a ação minimax do gameState atual.
        """

        def minimax(agentIndex, depth, state):
            # Caso Base: O jogo terminou (vitória/derrota) ou a profundidade máxima foi atingida.
            # Nesses casos, retornamos o valor do estado atual usando a função de avaliação.
            # O `self.evaluationFunction` será `betterEvaluationFunction` se configurado corretamente.
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Calcula o próximo agente a jogar e a próxima profundidade.
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth
            # Se o próximo agente for o Pac-Man (índice 0), isso significa que
            # todos os fantasmas jogaram e um novo "ply" começa.
            if nextAgent == self.index:
                nextDepth += 1

            # Obtém todas as ações legais disponíveis para o agente atual no estado atual.
            legalActions = state.getLegalActions(agentIndex)

            # Se não houver ações legais (ex: encurralado), retorna a avaliação do estado atual.
            if not legalActions:
                return self.evaluationFunction(state)

            # Lógica para o Pac-Man (Agente Maximizer - índice 0)
            if agentIndex == self.index:
                max_score = -float('inf')  # Inicializa com um valor muito pequeno
                best_action = None         # Variável para armazenar a melhor ação
                
                # Obtém a direção atual do Pac-Man para incentivar movimento em linha reta
                pacman_current_direction = state.getPacmanState().getDirection()

                # Itera sobre todas as ações legais possíveis para o Pac-Man
                for action in legalActions:
                    # Gera o próximo estado do jogo após o Pac-Man realizar a ação
                    successorState = state.generateSuccessor(agentIndex, action)
                    # Chama minimax recursivamente para obter o score dessa ação
                    score = minimax(nextAgent, nextDepth, successorState)

                    # --- Penaliza a ação STOP se houver outras opções ---
                    # Isso desencoraja o Pac-Man a ficar parado desnecessariamente,
                    # a menos que seja a única opção sensata ou uma vitória/derrota seja iminente.
                    if action == Directions.STOP and len(legalActions) > 1:
                        score -= 100 # Penalidade aumentada para desencorajar mais fortemente parar
                    
                    # --- NOVO: Incentiva o movimento em linha reta ---
                    # Adiciona um bônus se a ação for continuar na mesma direção,
                    # desde que não seja a ação STOP.
                    if action == pacman_current_direction and action != Directions.STOP:
                        score += 7 # Bônus para incentivar o movimento em linha reta (valor ajustável)

                    if action == Directions.REVERSE[pacman_current_direction] and pacman_current_direction != Directions.STOP:
                        score -= 50 # Penalidade para reverter a direção (novo valor

                    # Se o score atual for maior que o max_score encontrado até agora, atualiza
                    if score > max_score:
                        max_score = score
                        best_action = action
                    # Se houver empate nos scores, prefere uma ação que não seja STOP ou que seja em linha reta
                    elif score == max_score:
                        # Prioriza não ser STOP
                        if best_action == Directions.STOP and action != Directions.STOP:
                            best_action = action
                        # Se ambos não são STOP, prioriza a linha reta
                        elif action == pacman_current_direction and best_action != pacman_current_direction:
                            best_action = action
                
                # No nível mais externo da recursão (profundidade 0, ou seja, a chamada inicial
                # de getAction), retornamos a melhor ação. Nas chamadas recursivas internas,
                # apenas retornamos o valor (score) para ser usado na minimização/maximização.
                if depth == 0:
                    return best_action
                else:
                    return max_score
            # Lógica para os Fantasmas (Agentes Minimizer - índices > 0)
            else:
                min_score = float('inf')  # Inicializa com um valor muito grande

                # Itera sobre todas as ações legais possíveis para o fantasma atual
                for action in legalActions:
                    # Gera o próximo estado do jogo após o fantasma realizar a ação
                    successorState = state.generateSuccessor(agentIndex, action)
                    # Chama minimax recursivamente para obter o score dessa ação
                    score = minimax(nextAgent, nextDepth, successorState)

                    # Se o score atual for menor que o min_score encontrado até agora, atualiza
                    if score < min_score:
                        min_score = score
                
                # Fantasmas sempre retornam o valor mínimo (pior cenário para o Pac-Man)
                return min_score

        # Chamada inicial para a função minimax. Começamos com o Pac-Man (self.index, que é 0)
        # na profundidade 0 e o estado de jogo atual.
        return minimax(self.index, 0, gameState)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Avalia o estado atual do jogo para fornecer uma pontuação para o algoritmo Minimax.
    Esta função considera a posição do Pac-Man, a comida, as cápsulas e o estado dos fantasmas,
    com um foco em incentivar a vitória e evitar perigos desnecessários.
    """
    pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore() # Pontuação base do jogo

    # --- INCENTIVO: COMIDA ---
    foodList = currentGameState.getFood().asList() # Lista de posições de comida
    numFood = currentGameState.getNumFood() # Quantidade de comida restante
    
    # Penaliza fortemente a quantidade total de comida restante.
    # O objetivo principal é comer toda a comida, então quanto menos comida, melhor.
    score -= numFood * 50 # Multiplicador aumentado para priorizar comer comida

    # Recompensa pela proximidade da comida mais próxima.
    # Isso guia o Pac-Man em direção aos pellets visíveis.
    if foodList:
        minFoodDistance = float('inf')
        for foodPos in foodList:
            dist = manhattanDistance(pos, foodPos)
            if dist < minFoodDistance:
                minFoodDistance = dist
        score += 1000 / (minFoodDistance + 1) # Aumenta a recompensa por proximidade da comida

    # --- INCENTIVO: CÁPSULAS ---
    capsules = currentGameState.getCapsules() # Lista de posições das cápsulas
    
    # Recompensa significativa pela proximidade da cápsula mais próxima.
    # Cápsulas são cruciais para inverter o jogo.
    if capsules:
        minCapsuleDistance = float('inf')
        for capsulePos in capsules:
            dist = manhattanDistance(pos, capsulePos)
            if dist < minCapsuleDistance:
                minCapsuleDistance = dist
        score += 200.0 / (minCapsuleDistance + 1) # Aumentado o peso da recompensa da cápsula

    # --- FANTASMAS: PERIGO E OPORTUNIDADE ---
    ghostStates = currentGameState.getGhostStates() # Estados de todos os fantasmas

    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pos, ghostPos)

        if ghostState.scaredTimer > 0: # Fantasma está assustado (oportunidade de comer)
            # Recompensa muito alta para comer o fantasma assustado se estiver adjacente.
            # Isso prioriza a captura de fantasmas.
            if ghostDistance <= 1:
                score += 10000 # Bônus muito alto para priorizar comer o fantasma assustado.
            # Caso contrário, recompensa por se aproximar de fantasmas assustados distantes.
            else:
                score += 0 / (ghostDistance + 0) # Incentivo maior para perseguir fantasmas assustados.
        else: # Fantasma não está assustado (perigo)
            # Penalidade muito forte se o fantasma estiver extremamente próximo (morte iminente).
            if ghostDistance <= 1:
                score -= 1000 # Penalidade ainda mais alta para evitar a morte.
            # Penalidade moderada para fantasmas em um raio de perigo próximo.
            elif ghostDistance < 5: # Aumentado o raio de perigo considerado
                score -= 300 / (ghostDistance + 1) # Penalidade mais acentuada para proximidade.
            # Pequena penalidade para fantasmas distantes (ainda são uma ameaça potencial a ser evitada).
            else:
                score -= 200.0 / (ghostDistance + 1) # Ajustado para uma penalidade menor, mas presente.
    
    # Considerações adicionais para guiar o Pac-Man:
    # Se houver pouca comida restante, o Pac-Man deve ser mais agressivo para terminar o jogo.


    return score

# Abreviação para betterEvaluationFunction.
better = betterEvaluationFunction
