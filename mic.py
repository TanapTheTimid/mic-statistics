import numpy as np
import pandas as pd
import math

def sort_by_column(a, col):
    return a[a[:, col].argsort()]

def getClumpsPartition(D, Q):
    curr_clump = Q[(D[0][0], D[0][1])]
    cluster = [0]
    for idx, point in enumerate(D):
        point = (point[0], point[1])
        clump = Q[point]
        if clump != curr_clump:
            curr_clump = clump
            cluster = cluster + [idx]
    cluster = cluster + [len(D)]
    return cluster

def h1p(P):
    total_num_points = P[len(P) - 1] - P[0]
    sum = 0
    for i in range(1, len(P)):
        prob_cluster = (P[i] - P[i-1]) / total_num_points
        incr = 0
        if prob_cluster > 0:
            incr = prob_cluster * math.log(prob_cluster)
        sum = sum + incr
    sum = sum * -1
    return sum

def h1q(Q, y):
    total_num_points = len(Q)
    sum = 0
    for current_cluster in range(1, y+1):
        num_points_in_cluster = 0
        for (point, clstr) in Q.items():
            if clstr == current_cluster:
                num_points_in_cluster += 1
        prob_cluster = num_points_in_cluster / total_num_points 
        incr = 0
        if prob_cluster > 0:
            incr = prob_cluster * math.log(prob_cluster)
        sum = sum + incr
    sum = sum * -1
    return sum

def h2pq(P, Q, y, D):
    total_num_points = P[len(P) - 1] - P[0]
    sum = 0
    for i in range(1, len(P)):
        lower_bound_excl = P[i-1]
        upper_bound_incl = P[i]
        for current_cluster in range(1, y+1):
            num_points_in_cluster = 0
            for ((pointx, pointy), clstr) in Q.items():
                if (clstr == current_cluster 
                    and (lower_bound_excl == 0 
                        or pointx > D[lower_bound_excl - 1][0]) 
                    and (pointx <= D[upper_bound_incl - 1][0])):
                    num_points_in_cluster += 1
            prob_cluster = num_points_in_cluster / total_num_points 
            incr = 0
            if prob_cluster > 0:
                incr = prob_cluster * math.log(prob_cluster)
            sum = sum + incr
    sum = sum * -1
    return sum

def optimizeXAxis(D, Q, x, y):
    c = getClumpsPartition(D, Q)
    k = len(c) - 1

    HQ = h1q(Q, y)

    P = dict()
    I = dict()

    for t in range(2, k + 1):
        max_s = 1
        max_s_val = h1p([c[0], c[1], c[t]]) - h2pq([c[0], c[1], c[t]], Q, y, D)
        for s in range(2, t):
            s_val = h1p([c[0], c[s], c[t]]) - h2pq([c[0], c[s], c[t]], Q, y, D)
            if s_val > max_s_val:
                max_s_val = s_val 
                max_s = s 
        P[(t,2)] = [c[0], c[max_s], c[t]]
        I[(t,2)] = HQ + h1p(P[(t,2)]) - h2pq(P[(t,2)], Q, y, D)

    for l in range(3, x + 1):
        for t in range(l, k + 1):
            max_s = l - 1 
            max_s_val = (c[max_s] / c[t]) * (I[(max_s, l-1)] - HQ) - ((c[t] - c[max_s]) / c[t]) * h2pq([c[max_s], c[t]], Q, y, D)
            for s in range(l, t):
                s_val = (c[s] / c[t]) * (I[(s, l-1)] - HQ) - ((c[t] - c[s]) / c[t]) * h2pq([c[s], c[t]], Q, y, D)
                if s_val > max_s_val:
                    max_s_val = s_val
                    max_s = s
            P[(t,l)] = P[(max_s, l - 1)] + [c[t]]
            I[(t,l)] = HQ + h1p(P[(t,l)]) - h2pq(P[(t,l)], Q, y, D)
    
    for l in range(k + 1, x + 1):
        P[(k,l)] = P[(k,k)]
        I[(k,l)] = I[(k,k)]

    fin = []
    for i in range(2, x+1):
        fin = fin + [I[(k,i)]]

    return fin

def equipartitionYAxis(D, y):
    n = len(D)
    D = sort_by_column(D, 1)
    i = 1
    currRow = 1
    desiredRowSize = n / y
    Q = dict()

    for row in D:
        Q[(row[0], row[1])] = -1

    while i <= n:
        bi = D[i - 1][1]
        S = []
        for row in D:
            if row[1] == bi:
                S = S + [(row[0], row[1])]
        sharp = 0
        for row in D:
            if Q[(row[0], row[1])] == currRow:
                sharp = sharp + 1
        if sharp != 0 and abs(sharp + len(S) - desiredRowSize) >= abs(sharp - desiredRowSize):
            currRow = currRow + 1
            desiredRowSize = (n - i + 1) / (y - currRow + 1)
        for (sx, sy) in S:
            Q[(sx, sy)] = currRow
        i = i + len(S)
    return Q 

def approxMaxMI(D, x, y):
    Q = equipartitionYAxis(D, y)
    D = sort_by_column(D, 0)
    return optimizeXAxis(D, Q, x, y)

def approxCharacteristicMatrix(D, B):
    D_complement = D.copy()
    D_complement[:, [1, 0]] = D[:, [0, 1]]

    I = dict()
    I_complement = dict()
    M = dict()

    for y in range(2, math.floor(B/2) + 1):
        x = math.floor(B/y)
        res1 = approxMaxMI(D, x, y)
        for idx, elem in enumerate(res1):
            I[(2 + idx, y)] = elem 
        res2 = approxMaxMI(D_complement, x, y)
        for idx, elem in enumerate(res2):
            I_complement[(2 + idx, y)] = elem 
    
    for x in range(2, B + 1):
        for y in range(2, B + 1):
            if x * y <= B:
                I[(x,y)] = max(I[(x,y)], I_complement[(y,x)])
                M[(x,y)] = I[(x,y)] / min(math.log(x), math.log(y))

    return M

D = np.array(([1,7], [0,0], [2,-2], [8,0], [1,3], [2,4], [12,19], [14,14], [16, 18], [20,17], [24,14]))

print(approxCharacteristicMatrix(D, 10))