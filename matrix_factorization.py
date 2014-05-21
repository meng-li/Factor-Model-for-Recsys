# -*- coding:utf-8 -*-
'''
the basic version of matrix facotrization based on explicit feedback
reference:
    1. "matrix factorization techniques for recommender"
    2. a simple implementation: http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/#the-mathematics-of-matrix-factorization
'''
import numpy as np

ORI_MAT = [
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4]
          ]
print 'original matrix:'
print '\n'.join([str(val) for val in ORI_MAT])

row_cnt, col_cnt = len(ORI_MAT), len(ORI_MAT[0])
dim = 2
user_vec = np.random.rand(row_cnt, dim + 1)
item_vec = np.random.rand(col_cnt, dim + 1)

steps = 5000
gamma = 0.0002
theta = 0.05
miu, cnt = 0, 0
for row_idx in range(row_cnt):
    for col_idx in range(col_cnt):
        if ORI_MAT[row_idx][col_idx] == 0:
            continue
        miu += ORI_MAT[row_idx][col_idx]
        cnt += 1
miu /= float(cnt)
for cur_step in range(steps):
    # batch update
    for row_idx in range(row_cnt):
        for col_idx in range(col_cnt):
            if ORI_MAT[row_idx][col_idx] == 0:
                continue
            pu = user_vec[row_idx, 1:]
            bu = user_vec[row_idx, 0]
            qi = item_vec[col_idx, 1:]
            bi = item_vec[col_idx, 0]
            eij = ORI_MAT[row_idx][col_idx] - miu - bu - bi - np.dot(pu, qi)
            pu = pu + gamma * (eij * qi - theta * pu)
            qi = qi + gamma * (eij * pu - theta * qi)
            bu = bu + gamma * (eij - theta * bu)
            bi = bi + gamma * (eij - theta * bi)
            user_vec[row_idx, 0] = bu
            user_vec[row_idx, 1:] = pu
            item_vec[col_idx, 0] = bi
            item_vec[col_idx, 1:] = qi

print 'new matrix:'
for row_idx in range(row_cnt):
    l = []
    for col_idx in range(col_cnt):
        pu = user_vec[row_idx, 1:]
        qi = item_vec[col_idx, 1:]
        bu = user_vec[row_idx, 0]
        bi = user_vec[col_idx, 0]
        l.append(miu + bi + bu +np.dot(pu, qi))
    print '[' + ', '.join('%.3f' % val for val in l) + ']'



