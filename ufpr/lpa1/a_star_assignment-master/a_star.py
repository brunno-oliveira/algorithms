import math
from dists import dists
from dists import straight_line_dists_from_bucharest as line


def getUpperStraightLine():
    straight_line = {}
    for key, value in line.items():
        straight_line.update({key.upper() : value})
    return  straight_line   


def getUpperDists():
    n_dist = {}
    for key, value in dists.items():
        n_key = {}
        n_values = {}
        for v in value:
            n_values.update({v[0].upper():v[1]})
        n_dist.update({key.upper() : n_values})
    return n_dist


def validateCity(node, new_city):
    ''' Valida se já passou por essa cidade,
        olhando para os nós que são dicionários
    '''
    for n in node:
        if(type(n) == dict):
            if(list(n.keys())[0] == new_city):
                return False
    return True
        

def getPathList(node):
    result = []
    for n in node:
        if(type(n) == dict):
            result.append(n)
    return result

# goal sempre sera 'bucharest'
# Criar um dicionario para o o caminho percorrido 
# E um com a ordem para gerenciar o step
def a_star(start, goal='BUCHAREST'):
    """
    Retorna uma lista com o caminho de start até
    goal segundo o algoritmo A*
    """
    start_city = start.upper()
    print(f'Starting in {start_city}')

    straight_line = getUpperStraightLine()
    distance = getUpperDists()

    node = []
    node.append({start_city:straight_line[start_city]})
    while(True == True):
        # Add Node
        new_node = []
        for city, dist in distance[list(node[-1].keys())[0]].items():
            new_node.append({city : (dist + straight_line[city])})
        node.append(new_node)

        # closest city
        closest_city = None
        closest_dist = None
        for city in node[-1]:
            for city_key, dist in city.items():
                if(closest_city is None):
                    if(validateCity(node[:-1], city_key)):
                        closest_city = city_key
                        closest_dist = dist
                elif(dist < closest_dist):
                    if(validateCity(node[:-1], city_key)):
                        closest_city = city_key
                        closest_dist = dist                    
        node.append({closest_city:closest_dist})

        if(goal == closest_city):
            print('FINISHED')
            return getPathList(node)
            
print(a_star(start='lugoj'))