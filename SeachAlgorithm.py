import numpy as np
import random
import math
import heapq
import path_planning as pathmap
import matplotlib.pyplot as plt
import xlsxwriter
  


# Priority Queue based on heapq
class PriorityQueue:
    def __init__(self):
        self.elements = []
    def isEmpty(self):
        return len(self.elements) == 0
    def add(self, item, priority):
        heapq.heappush(self.elements,(priority,item))
    def remove(self):
        return heapq.heappop(self.elements)[1]

def get_neighbors(current, start, cost, mapsize):
    x = []
    y = current
    if (0 <= y[0]+1 < mapsize):
        if (cost[y[0]+1, y[1]] == 0 or cost[y[0]+1, y[1]] == -3)  and not(start[0] == y[0]+1 and start[1] == y[1]):
            x.append([y[0] +1, y[1]])
    if (0 <= y[0]-1 < mapsize):
        if (cost[y[0]-1, y[1]] == 0 or cost[y[0]-1, y[1]] == -3)  and not(start[0] == y[0]-1 and start[1] == y[1]):
            x.append([y[0] -1, y[1]])
    if (0 <= y[1]+1 < mapsize):
        if (cost[y[0], y[1]+1] == 0 or cost[y[0], y[1]+1] == -3)  and not(start[0] == y[0] and start[1] == y[1] +1):
            x.append([y[0], y[1] + 1])
    if (0 <= y[1]-1 < mapsize):
        if (cost[y[0], y[1]-1] == 0 or cost[y[0], y[1]-1] == -3)  and not(start[0] == y[0] and start[1] == y[1] - 1):
            x.append([y[0], y[1]-1])
    return x

workbook = xlsxwriter.Workbook('hello.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'random')
worksheet.write(1, 0, 'cost')
worksheet.write(1, 1, 'nodes')
worksheet.write(0, 2, 'bfs')
worksheet.write(1, 2, 'cost')
worksheet.write(1, 3, 'nodes')
worksheet.write(0, 4, 'dfs')
worksheet.write(1, 4, 'cost')
worksheet.write(1, 5, 'nodes')
worksheet.write(0, 6, 'greedy manhattan')
worksheet.write(1, 6, 'cost')
worksheet.write(1, 7, 'nodes')
worksheet.write(0, 8, 'greedy euclidian')
worksheet.write(1, 8, 'cost')
worksheet.write(1, 9, 'nodes')
worksheet.write(0, 10, 'a* manhattan')
worksheet.write(1, 10, 'cost')
worksheet.write(1, 11, 'nodes')
worksheet.write(0, 12, 'a* euclidian')
worksheet.write(1, 12, 'cost')
worksheet.write(1, 13, 'nodes')
worksheet.write(0, 14, 'a8 modified')
worksheet.write(1, 14, 'cost')
worksheet.write(1, 15, 'nodes')
for trial in range(20):
    def random_search(map_, start, goal, mapsize):

 
        moving_cost = 1


        frontier = PriorityQueue()

        frontier.add(start, 0)

        map1 = map_

        came_from = []#{}
        d = dict()
        d[str(start)] = [start]

        cost = map_#cost = []#{}
        cost[start[0], start[1]] = 0

        nodes = 0

        while not frontier.isEmpty():
            current = frontier.remove()

            x = d[str([current[0], current[1]])]

            if current == goal or x == -3:
                break


        
            for next_ in get_neighbors(current, start, cost, mapsize):

                if cost[next_[0], next_[1]] == 0:
                
                    nodes = nodes+1

                    cost[next_[0], next_[1]] = cost[current[0], current[1]] +1
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                if cost[next_[0], next_[1]] == -3:
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                    frontier = PriorityQueue()
                    break
                frontier.add(next_, random.randint(0,mapsize**2))


        return d, cost, nodes


    def bfs_search(map_, start, goal, mapsize):


        moving_cost = 1


        frontier = PriorityQueue()

        frontier.add(start, 0)

        map1 = map_

        came_from = []#{}
        d = dict()
        d[str(start)] = [start]

        cost = map_#cost = []#{}
        cost[start[0], start[1]] = 0

        nodes = 0

        while not frontier.isEmpty():
            current = frontier.remove()

            x = d[str([current[0], current[1]])]

            if current == goal or x == -3:
                break


        
            for next_ in get_neighbors(current, start, cost, mapsize):

                if cost[next_[0], next_[1]] == 0:
                    nodes = nodes +1
                    cost[next_[0], next_[1]] = cost[current[0], current[1]] +1
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                if cost[next_[0], next_[1]] == -3:
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                    frontier = PriorityQueue()
                    break
                frontier.add(next_, cost[next_[0], next_[1]])


        return d, cost, nodes

    def dfs_search(map_, start, goal, mapsize):


        moving_cost = 1


        frontier = PriorityQueue()

        frontier.add(start, 0)

        map1 = map_

        came_from = []#{}
        d = dict()
        d[str(start)] = [start]

        cost = map_#cost = []#{}
        cost[start[0], start[1]] = 0

        nodes = 0

        while not frontier.isEmpty():
            current = frontier.remove()

            x = d[str([current[0], current[1]])]

            if current == goal or x == -3:
                break

            i = 0
            for next_ in get_neighbors(current, start, cost, mapsize):
                i =+ 1

                if cost[next_[0], next_[1]] == 0:
                    nodes = nodes+1
                    cost[next_[0], next_[1]] = cost[current[0], current[1]] + 1
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                if cost[next_[0], next_[1]] == -3:
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                    frontier = PriorityQueue()
                    break
                frontier.add(next_, mapsize**2 - i*cost[next_[0], next_[1]])


        return d, cost, nodes

    def g_m_search(map_, start, goal, mapsize):


        moving_cost = 1


        frontier = PriorityQueue()

        frontier.add(start, 0)

        map1 = map_

        came_from = []#{}
        d = dict()
        d[str(start)] = [start]

        cost = map_#cost = []#{}
        cost[start[0], start[1]] = 0

        nodes = 0

        while not frontier.isEmpty():
            current = frontier.remove()

            x = d[str([current[0], current[1]])]

            if current == goal or x == -3:
                break


        
            for next_ in get_neighbors(current, start, cost, mapsize):

                if cost[next_[0], next_[1]] == 0:
                    nodes = nodes+1
                    cost[next_[0], next_[1]] = abs(next_[0] - goal[0]) + abs(next_[1] - goal[1])
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                if cost[next_[0], next_[1]] == -3:
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                    frontier = PriorityQueue()
                    break
                frontier.add(next_, cost[next_[0], next_[1]])


        return d, cost, nodes

    def g_e_search(map_, start, goal, mapsize):


        moving_cost = 1


        frontier = PriorityQueue()

        frontier.add(start, 0)

        map1 = map_

        came_from = []#{}
        d = dict()
        d[str(start)] = [start]

        cost = map_#cost = []#{}
        cost[start[0], start[1]] = 0


        nodes = 0

        while not frontier.isEmpty():
            current = frontier.remove()

            x = d[str([current[0], current[1]])]

            if current == goal or x == -3:
                break


        
            for next_ in get_neighbors(current, start, cost, mapsize):

                if cost[next_[0], next_[1]] == 0:
                    nodes = nodes + 1
                    cost[next_[0], next_[1]] = ((next_[0] - goal[0])**2) + ((next_[1] - goal[1])**2)
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                if cost[next_[0], next_[1]] == -3:
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                    frontier = PriorityQueue()
                    break
                frontier.add(next_, cost[next_[0], next_[1]])


        return d, cost, nodes
    def a_m_search(map_, start, goal, mapsize):


        moving_cost = 1


        frontier = PriorityQueue()

        frontier.add(start, 0)

        map1 = map_

        came_from = []#{}
        d = dict()
        d[str(start)] = [start]

        cost = map_#cost = []#{}
        cost[start[0], start[1]] = 0

        nodes = 0

        while not frontier.isEmpty():
            current = frontier.remove()

            x = d[str([current[0], current[1]])]

            if current == goal or x == -3:
                break


        
            for next_ in get_neighbors(current, start, cost, mapsize):

                if cost[next_[0], next_[1]] == 0:
                    nodes = nodes+1
                    cost[next_[0], next_[1]] = min(cost[current[0], current[1]] +1, abs(next_[0] - start[0]) + abs(next_[1] - start[1]))
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                if cost[next_[0], next_[1]] == -3:
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                    frontier = PriorityQueue()
                    break
                frontier.add(next_, cost[next_[0], next_[1]] + abs(next_[0] - goal[0]) + abs(next_[1] - goal[1]))


        return d, cost, nodes


    def a_e_search(map_, start, goal, mapsize):


        moving_cost = 1


        frontier = PriorityQueue()

        frontier.add(start, 0)

        map1 = map_

        came_from = []#{}
        d = dict()
        d[str(start)] = [start]

        cost = map_#cost = []#{}
        cost[start[0], start[1]] = 0

        nodes = 0

        while not frontier.isEmpty():
            current = frontier.remove()

            x = d[str([current[0], current[1]])]

            if current == goal or x == -3:
                break

        
            for next_ in get_neighbors(current, start, cost, mapsize):

                if cost[next_[0], next_[1]] == 0:
                    nodes = nodes+1
                    cost[next_[0], next_[1]] = min(cost[current[0], current[1]] +1, abs(next_[0] - start[0]) + abs(next_[1] - start[1]))
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                if cost[next_[0], next_[1]] == -3:
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                    frontier = PriorityQueue()
                    break
                frontier.add(next_, cost[next_[0], next_[1]] + (next_[0] - goal[0])**2 + (next_[1] - goal[1])**2)


        return d, cost, nodes

    def a_mod_search(map_, start, goal, ob, mapsize):


        moving_cost = 1


        frontier = PriorityQueue()

        frontier.add(start, 0)

        map1 = map_

        came_from = []#{}
        d = dict()
        d[str(start)] = [start]

        cost = map_#cost = []#{}
        cost[start[0], start[1]] = 0
        print(start[0], start[1])
        nodes = 0
        while not frontier.isEmpty():
            current = frontier.remove()

            x = d[str([current[0], current[1]])]

            if current == goal or x == -3:
                break


        
            for next_ in get_neighbors(current, start, cost, mapsize):

                if cost[next_[0], next_[1]] == 0:
                    nodes = nodes+1
                    cost[next_[0], next_[1]] = (next_[0] - goal[0])**2 + (next_[1] - goal[1])**2 #+ mapsize**2
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                elif cost[next_[0], next_[1]] == -3:
                    y = x + [[next_[0], next_[1]]]
                    d[str([next_[0], next_[1]])] = y
                    frontier = PriorityQueue()
                    break
                if (current[1] < next_[1] < ob[2]) and (ob[1] < next_[0] < ob[0]):
                    None#frontier.add(next_, cost[next_[0], next_[1]]  )

                else:
                    frontier.add(next_, cost[next_[0], next_[1]])


        return d, cost, nodes
##############################3
    mapsize = 60
    a, start, goal, ob = pathmap.generateMap2d_obstacle([mapsize, mapsize])

    aa = a.copy()
    print("b", start, goal)
    print(ob)

    #plt.imshow(a)
    #plt.show()

############################################
#random
    aa = a.copy()
    came_from, cost, nodes = random_search(aa, start, goal, mapsize)
    cost[start[0],start[1]] = -2
    bb = []
    for i in came_from[str(goal)]:
        bb.append([i[1], i[0]])

    #pathmap.plotMap(cost, np.array(bb), 'random')
    path = len(bb)
    nodes = nodes +1
    print('random')
    print('nodes:', nodes)
    print('path:', len(bb))
    worksheet.write(trial +2, 0, path)
    worksheet.write(trial +2, 1, nodes)
############################################
#bfs
    aa = a.copy()
    came_from, cost, nodes = bfs_search(aa, start, goal, mapsize)
    cost[start[0],start[1]] = -2
    bb = []
    for i in came_from[str(goal)]:
        bb.append([i[1], i[0]])

    #pathmap.plotMap(cost, np.array(bb), 'bfs')
    path = len(bb)
    nodes = nodes +1
    print('bfs')
    print('nodes:', nodes)
    print('path:', len(bb))
    worksheet.write(trial +2, 2, path)
    worksheet.write(trial +2, 3, nodes)
############################################
#dfs
    aa = a.copy()
    came_from, cost, nodes = dfs_search(aa, start, goal, mapsize)
    cost[start[0],start[1]] = -2
    bb = []
    for i in came_from[str(goal)]:
        bb.append([i[1], i[0]])

    #pathmap.plotMap(cost, np.array(bb),'dfs')
    path = len(bb)
    nodes = nodes +1
    print('dfs')
    print('nodes:', nodes)
    print('path:', len(bb))
    worksheet.write(trial +2, 4, path)
    worksheet.write(trial +2, 5, nodes)
#############################################
#greedy-manhattan
    aa = a.copy()
    came_from_1, cost_1, nodes = g_m_search(aa, start, goal, mapsize)
    cost_1[start[0],start[1]] = -2
    bb = []
    for i in came_from_1[str(goal)]:
        bb.append([i[1], i[0]])

    #pathmap.plotMap(cost_1, np.array(bb), 'greedy-manhattan')
    path = len(bb)
    nodes = nodes +1
    print('greedy_manhattan')
    print('nodes:', nodes)
    print('path:', len(bb))
    worksheet.write(trial +2, 6, path)
    worksheet.write(trial +2, 7, nodes)
#############################################
#greedy-euclid
    aa = a.copy()
    came_from_2, cost_2, nodes = g_e_search(aa, start, goal, mapsize)
    cost_2[start[0],start[1]] = -2
    bb = []
    for i in came_from_2[str(goal)]:
        bb.append([i[1], i[0]])

    #pathmap.plotMap(cost_2, np.array(bb),'greedy-euclid')
    path = len(bb)
    nodes = nodes +1
    print('greedy_euclid')
    print('nodes:', nodes)
    print('path:', len(bb))
    worksheet.write(trial +2, 8, path)
    worksheet.write(trial +2, 9, nodes)
#############################################
#a*-manhattan
    aa = a.copy()
    came_from_3, cost_3, nodes = a_m_search(aa, start, goal, mapsize)
    cost_3[start[0],start[1]] = -2

    bb = []
    for i in came_from_3[str(goal)]:
        bb.append([i[1], i[0]])

    #pathmap.plotMap(cost_3, np.array(bb), 'a*-manhattan')
    path = len(bb)
    nodes = nodes +1
    print('a*-manhattan')
    print('nodes:', nodes)
    print('path:', len(bb))
    worksheet.write(trial +2, 10, path)
    worksheet.write(trial +2, 11, nodes)
#############################################
#a*-euclid
    aa = a.copy()
    came_from_3, cost_3, nodes = a_e_search(aa, start, goal, mapsize)
    cost_3[start[0],start[1]] = -2

    bb = []
    for i in came_from_3[str(goal)]:
        bb.append([i[1], i[0]])

    #pathmap.plotMap(cost_3, np.array(bb), 'a*-euclid')
    path = len(bb)
    nodes = nodes +1
    print('a*-euclid')
    print('nodes:', nodes)
    print('path:', len(bb))
    worksheet.write(trial +2, 12, path)
    worksheet.write(trial +2, 13, nodes)
#############################################
#a*-modified
    aa = a.copy()
    came_from_3, cost_3, nodes = a_mod_search(aa, start, goal, ob, mapsize)
    cost_3[start[0],start[1]] = -2
    #print(cost)
    bb = []
    print(came_from_3)
    for i in came_from_3[str(goal)]:
        bb.append([i[1], i[0]])

    #pathmap.plotMap(cost_3, np.array(bb), 'a*-modified')
    path = len(bb)
    nodes = nodes +1
    print('a*-modified')
    print('nodes:', nodes)
    print('path:', len(bb))
    worksheet.write(trial +2, 14, path)
    worksheet.write(trial +2, 15, nodes)
workbook.close()
