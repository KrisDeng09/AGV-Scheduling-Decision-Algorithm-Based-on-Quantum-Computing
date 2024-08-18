import numpy as np
import kaiwu as kw
import matplotlib.pyplot as plt
import math
import itertools
import random
import time

agv_capacity = 50

target_num = 10
time_start = time.time()
start_point = np.random.randint(0,100,size = [1,2])
target_point = np.random.randint(0,100,size = [target_num,2])
location = np.append(start_point,target_point,axis = 0)
loc = set()
for i in location:
    loc.add(tuple(i))
location = np.array(list(loc))
load = np.random.randint(1,20,size = [location.shape[0],1])
load[0] = 0
sum_load = float(sum(load))
vehicle_max = sum_load/agv_capacity
data = np.concatenate((location,load),axis=1)

dist_matrix = []
length = len(data)
for i in range(length):
    for j in range(length):
        dist = np.sqrt((data[i][0]-data[j][0])**2+(data[i][1]-data[j][1])**2)
        dist_matrix.append(dist)
dist_matrix = np.matrix(dist_matrix)
dist_matrix = np.around(dist_matrix, decimals = 2)
dist_matrix = dist_matrix.reshape(length,length)
max_dist = np.amax(dist_matrix)
if max_dist > 10:
    divide_index = 10 / max_dist
w = dist_matrix * divide_index

vehicle_num = int(vehicle_max)
if vehicle_max > vehicle_num:
    vehicle_num = vehicle_num+1
z = np.zeros((vehicle_num,len(data)))
for i in range(vehicle_num):
    z[i]=load.T
#print(w)

z_x = z.shape[0]
z_y = z.shape[1]

y = kw.qubo.ndarray((z_x, z_y), "y", kw.qubo.binary)

n = w.shape[0]

x = kw.qubo.ndarray((n, n), "x", kw.qubo.binary)

edges = [(u, v) for u in range(n) for v in range(n) if w[u, v] != 0]
no_edges = [(u, v) for u in range(n) for v in range(n) if w[u, v] == 0]

v_cons = kw.qubo.sum((1-kw.qubo.sum(y[k, i] for k in range(z_x)))**2 for i in range(1,z_y))
i_cons = kw.qubo.sum((1-kw.qubo.sum(x[i, j] for j in range(1,n) if w[i, j] != 0))**2 for i in range(1,n))
o_cons = kw.qubo.sum((1-kw.qubo.sum(x[i, j] for i in range(1,n) if w[i, j] != 0))**2 for j in range(1,n))
s_cons = kw.qubo.sum(1-kw.qubo.sum(y[k, j] * x[0, j] for j in range(1,n)) for k in range(z_x))
e_cons = kw.qubo.sum(1-kw.qubo.sum(y[k, i] * x[i, 0] for i in range(1,n)) for k in range(z_x))
n_cons = (kw.qubo.sum((x[i, j]for i, j in no_edges)))**2
l_cons = kw.qubo.sum(kw.qubo.sum(kw.qubo.sum(y[k, h]*x[i, h] for i in range(n) if w[i, h] != 0)-kw.qubo.sum(y[k, h]*x[h, j] for j in range(n) if w[h, j] != 0) for h in range(n))for k in range(z_x))
weight_cons = kw.qubo.sum(agv_capacity - kw.qubo.sum(z[k, j]*y[k, i]*x[i, j]for i,j in edges) for k in range(z_x))
#for k in range(z_x):
    #vehicle_weight = agv_capacity - kw.qubo.sum(z[k, j]*y[k, i]*x[i, j]for i,j in edges)
    #vehicle_weight = 1 - kw.qubo.sum(y[k, i]*x[i, j]for i,j in edges)
#kw.qubo.details(weight_cons)
    #pre_ve = kw.qubo.sum(vehicle_weight)
    #if vehicle_weight < 0:
        #s = 1
    #else:
        #s = 0
    #q = s * M + pow(vehicle_weight,2)

ham_cycle = v_cons+i_cons+o_cons+s_cons+e_cons+n_cons+l_cons

path_cost = kw.qubo.sum(kw.qubo.sum(w[i, j] * y[k,i] * x[i, j] for i, j in edges) for k in range(z_x))

obj = 20 * ham_cycle + path_cost + weight_cons

obj = kw.qubo.make(obj)

obj_ising = kw.qubo.cim_ising_model(obj)

matrix = obj_ising.get_ising()["ising"]

pump_num = 1.4
noise_num = 0.03
nor_num = 0.5
laps_num = int(target_num/vehicle_num*800)
dt_num = 0.1
matrix_n = kw.cim.normalizer(matrix, nor_num)

output,ham_output = kw.cim.simulator(
                matrix,
                pump = pump_num,
                noise = noise_num,
                laps = laps_num,
                dt = dt_num,
                normalization = nor_num,
                iterations = int(10*target_num/vehicle_num))

opt = kw.sampler.optimal_sampler(matrix, output, bias=0, negtail_ff=False)

cim_best = opt[0][:50]

tsbest_s, tsbest_h = kw.utils.tabu_optimizer(cim_best, matrix, max_samp=50, gap=2)

ts_best = tsbest_s[0]

ts_best = ts_best * ts_best[-1]

vars = obj_ising.get_variables()

sol_dict = kw.qubo.get_sol_dict(ts_best, vars)


v_val = kw.qubo.get_val(v_cons, sol_dict)
i_val = kw.qubo.get_val(i_cons, sol_dict)
o_val = kw.qubo.get_val(o_cons, sol_dict)
s_val = kw.qubo.get_val(s_cons, sol_dict)
e_val = kw.qubo.get_val(e_cons, sol_dict)
n_val = kw.qubo.get_val(n_cons, sol_dict)
l_val = kw.qubo.get_val(l_cons, sol_dict)
ham_val = kw.qubo.get_val(ham_cycle, sol_dict)
#print('position cons: {}'.format(seq_val))
#print('node_cons cons: {}'.format(node_val))
print('ham_cycle: {}'.format(ham_val))

path_val = kw.qubo.get_val(path_cost, sol_dict)
path_val = path_val / divide_index
total_path = path_val
print('path_cost: {}'.format(path_val))
h = kw.sampler.hamiltonian(matrix_n, ham_output)
plt.plot(h, linewidth=1)

if ham_val == 0:
      print('valid path')
      x_val = kw.qubo.get_array_val(y, sol_dict)
      nonzero_index = np.array(np.nonzero(y_val)).T
      for i in range(z_x):
          print(nonzero_index[i,1].flatten()+"\n")
      
else:
      print('invalid path')

plt.show()
