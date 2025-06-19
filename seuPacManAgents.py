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

    Parte de implementação do minimax para o pacman, que toma decisões maximizando
    as chances de vitória com base na análise de alternativas, levando em contra múlti-
    plos agentes

    """
    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '3'):
        # inicia o agente com a função de avaliação e profundidade máxiam de busca
        super().__init__(evalFn=evalFn, depth=depth)
        self.depth = 3 # definir a profundidade de busca
        
    def getAction(self, gameState: GameState):
        """
        executa o algoritmo minimax a partir do estado atual do jogo, retornando a 
        melhor ação por meio do gameState atual
        """

        def minimax(agentIndex, depth, state):

            # Condições de parada: vitória, derrota ou profundidade máxima atingida
            # nesses casos, o valor do estado atual retorna usando a função de avaliação
            # o self.evaluationFunction vira betterEvaluationFunction se tudo estiver certo
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # calcula o próximo agente e a próxima profundidade
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth
            # se o próximo agente for 0 (pacman), significa que 
            # todos os fantasmas jogaram e um novo ply começa
            if nextAgent == self.index:
                nextDepth += 1

            # ações possíveis disponíveis para o agente atual no estado atual
            legalActions = state.getLegalActions(agentIndex)

            # caso não haja ações possíveis, retorna a avaliação do estado atual
            if not legalActions:
                return self.evaluationFunction(state)

            # lógica para o pacman (maximizador)
            if agentIndex == self.index:
                max_score = -float('inf') 
                best_action = None       
                
                # obtem a direção atual do pacman para incentivar movimento em linha reta
                pacman_current_direction = state.getPacmanState().getDirection()

                for action in legalActions:
                    # gera o próximo estado do jogo após o pacman realizar a ação
                    successorState = state.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgent, nextDepth, successorState)

                    # o pacman é penalizado se ficar parado, a não ser que seja a única opção para
                    # uma vitória ou derrota
                    if action == Directions.STOP and len(legalActions) > 1:
                        score -= 100 
                    
                    # incentiva a manter a direção atual
                    if action == pacman_current_direction and action != Directions.STOP:
                        score += 20 

                    # penaliza caso mude para a direção oposta
                    if action == Directions.REVERSE[pacman_current_direction] and pacman_current_direction != Directions.STOP:
                        score -= 50 

                    # escolhe a ação com o maior score
                    if score > max_score:
                        max_score = score
                        best_action = action
                    # critérios para evitar stop e dar prioridade para a direçao atual
                    elif score == max_score:
                        if best_action == Directions.STOP and action != Directions.STOP:
                            best_action = action
                        elif action == pacman_current_direction and best_action != pacman_current_direction:
                            best_action = action
                if depth == 0:
                    return best_action
                else:
                    return max_score
            # logica para os fantasmas
            else:
                min_score = float('inf')
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgent, nextDepth, successorState)

                    if score < min_score:
                        min_score = score
                
                return min_score

        # inicia a recursao minimax
        return minimax(self.index, 0, gameState)


def betterEvaluationFunction(currentGameState: GameState):
    """
    avaliação do estado atual do jogo, levando em consideração a posição do pacman, comida
    capsulas e o estado dos fantasmas, retorna uma pontuaçao numerica para orientar as decisoes do
    minimax
    """
    pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    # comida
    foodList = currentGameState.getFood().asList()
    numFood = currentGameState.getNumFood()
    score -= numFood * 50

    # recomepnsa pela proximidade da comida mais próxima, guiando o pacman
    # em direção aos pellets visíveis
    if foodList:
        minFoodDistance = float('inf')
        for foodPos in foodList:
            dist = manhattanDistance(pos, foodPos)
            if dist < minFoodDistance:
                minFoodDistance = dist
        score += 1000 / (minFoodDistance + 1) # aumenta a recompensa por proximidade da comida

    # capsulas
    capsules = currentGameState.getCapsules()
    if capsules:
        minCapsuleDistance = float('inf')

        for capsulePos in capsules:
            dist = manhattanDistance(pos, capsulePos)
            if dist < minCapsuleDistance:
                minCapsuleDistance = dist
        score += 200.0 / (minCapsuleDistance + 1) # aumenta o peso da recompensa da cápsula

    # estado em relacao aos fantasmas
    ghostStates = currentGameState.getGhostStates() # estados de todos os fantasmas

    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pos, ghostPos)

        if ghostState.scaredTimer > 0: # fantasma assustado = oportunidade para comer ele
            if ghostDistance <= 1:
                score += 10000 # bônus absurdamente alto para priorizar comer o fantasma assustado
            else:
                score += 0 / (ghostDistance + 0) # incentiva a perseguir o fantasma assustado
        else: # fantasma nao esta assustado

            # penalidade muito alta se o fantasma estiver extremamente próximo
            if ghostDistance <= 1:
            
                score -= 1000 # penalidade ainda mais alta para evitar morrer

            # penalidade leve para fantasmas em um raio de perigo próximo
            elif ghostDistance < 5: # aumenta consideravelmente o raio de perigo

                score -= 300 / (ghostDistance + 1) # penalidade por proximidade
            
            # penalidade pequena para fantasmas um pouco distantes
            else:
                score -= 200.0 / (ghostDistance + 1)
    
    # se houver pouca comida restante, o pacman deve ser mais agressivo para terminar o jogo


    return score
better = betterEvaluationFunction
