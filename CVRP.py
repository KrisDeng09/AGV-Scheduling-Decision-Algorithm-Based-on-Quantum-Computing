import kaiwu as kw
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import time
import random
import itertools

# 目标站点最优起始划分点(序号)
optimal_start_point = 1

# AGV参数设定
agv_type = 0 # 型号0
agv_num = 10 # AGV可调度数量

agv_para = np.dtype({
    'names':['model','load','diatance'],
    'formats':['U10','f','i']
})
agv = np.array([('normal',100,1000),('advance',120,1800)],dtype = agv_para) # 型号0:('normal',100,1000)；型号1:('advance',120,1800)
print('Model is',agv[agv_type]['model'],', max capacity is',agv[agv_type]['load'],', max diatance is',agv[agv_type]['diatance'])
agv_capacity = agv[agv_type]['load']

time_start = time.time()
# 目标站点文件读取生成
data = np.loadtxt('C:/Users/ASUS/Desktop/CVRP/data.txt')
data = data[:,1:3]
load = np.loadtxt('C:/Users/ASUS/Desktop/CVRP/load.txt')
load = load[:,1:2]
point_num = len(data) # 目标站点个数


# 目标站点随机函数生成
#target_num = 50
#data = np.random.randint(0,100,size = [target_num,2])
#load = np.random.randint(1,10,size = [target_num,1])
#load[0] = 0
#data = np.concatenate((data,load),axis=1)


# 角度计算
angle = np.zeros(point_num)
for i in range (point_num):
    angle[i] = math.atan2((data[i,1]-data[0,1]),(data[i,0]-data[0,0]))
    angle[i] = int(angle[i] * 180 / math.pi)
angle = np.reshape(angle,(point_num,1))


# 合并数据(x,y,angle.load)
location = data
location = np.append(location,angle,axis = 1)
location = np.append(location,load,axis = 1)


# 数据划分
start_point = location[0]
start_point = start_point[0:2]
#print("Start Point Location:",start_point)
target_point = location[1:]
target_point = target_point[np.argsort(target_point[:,2])]
#print("Target Point Location:",target_point)


# 参数初始化
load_index = 0
agv_need = 1
target_num = point_num - 1
target_index = optimal_start_point # 自定义初始划分点，待通过优化算法完善
start_target_point = target_index


# 计算所需车辆计算
for index in range(target_num):
    if target_index > target_num:
        target_index = target_index - target_num
    load_index = load_index + target_point[target_index - 1][3]
    if load_index > agv_capacity:
        agv_need = agv_need + 1
        load_index = target_point[target_index - 1][3]
    target_index = target_index + 1


# Cluster初始化
cluster = np.zeros(4*agv_need*point_num).reshape(agv_need,point_num,4)


# 参数初始化
point_index = 0
load_index = 0
cluster_index = 0
bit = target_index
target_index = start_target_point
count = 0
total_path = 0


# 目标站点(Cluster形式）
for index in range(target_num):
    if target_index > target_num:
        target_index = target_index - target_num
    load_index = load_index + target_point[target_index - 1][3]
    if load_index > agv_capacity:
        cluster[cluster_index][0] = np.asarray([location[0]])
        for i in range(1,point_index + 1):
            if bit > target_num:
                bit = bit - target_num
            cluster[cluster_index][i] = target_point[bit - 1]
            bit = bit + 1
            count = count + 1
        cluster_index = cluster_index + 1
        load_index = target_point[target_index - 1][3]
        point_index = 1
        if cluster_index > agv_need - 1:
            cluster_index = agv_need - 1
    else:
        point_index = point_index + 1
    target_index = target_index + 1
if count != target_num:
    cluster[cluster_index][0] = np.asarray([location[0]])
    for i in range(1,target_num-count+1):
        if bit > target_num:
            bit = bit - target_num
        cluster[cluster_index][i] = target_point[bit - 1]
        bit = bit + 1
#print("Cluster form:",cluster)


# TSP算法
for i in range(agv_need):

    
    # 提取每辆agv坐标数据
    extract_data = []
    index_3 = 0
    for j in range(target_num):
        if cluster[i,j,0] == 0 and cluster[i,j,1] == 0:
            break
        else:
            index_3 = index_3 + 1
            
    extract_data.append(cluster[i,0:index_3])
    extract_data = np.reshape(extract_data,[index_3,4])
    #print(extract_data)


    # 原始坐标点画图
    plt.subplot(2, 2, 1)
    plt.scatter(extract_data[:,0],extract_data[:,1])
    extract_data[:,0] = extract_data[:,0]
    extract_data[:,1] = extract_data[:,1]


    # 距离矩阵创建
    dist_matrix = []
    length = len(extract_data)
    for i in range(length):
        for j in range(length):
            dist = np.sqrt((extract_data[i][0]-extract_data[j][0])**2+(extract_data[i][1]-extract_data[j][1])**2)
            dist_matrix.append(dist)
    dist_matrix = np.matrix(dist_matrix)
    dist_matrix = np.around(dist_matrix, decimals = 2)
    dist_matrix = dist_matrix.reshape(length,length)
    #print(dist_matrix)
    max_dist = np.amax(dist_matrix)
    if max_dist > 10:
        divide_index = 10 / max_dist
    dist_matrix = dist_matrix * divide_index

    
    # 导入距离矩阵
    w = dist_matrix

    # 获取节点数
    n = w.shape[0]

    # 创建qubo变量矩阵。
    x = kw.qubo.ndarray((n, n), "x", kw.qubo.binary)

    # 获取边和非边的点对集合
    edges = [(u, v) for u in range(n) for v in range(n) if w[u, v] != 0]
    no_edges = [(u, v) for u in range(n) for v in range(n) if w[u, v] == 0]

    # 点约束，每个点只属于一个位置
    sequence_cons = kw.qubo.sum((1-kw.qubo.sum(x[v, j] for j in range(n)))**2 for v in range(n))

    # 位置约束，每个位置只能有一个点
    node_cons = kw.qubo.sum((1-kw.qubo.sum(x[v, j] for v in range(n)))**2 for j in range(n))

    # 边约束，无边的点对不能出现在路径里
    connect_cons = kw.qubo.sum(kw.qubo.sum(x[u,j] * x[v, j+1] for j in range(n-1)) + x[u,n-1] * x[v, 0] for u, v in no_edges)

    # 哈密尔顿环约束，以上三个约束之和
    ham_cycle = sequence_cons + node_cons + connect_cons

    # TSP环路长度
    path_cost = kw.qubo.sum(w[u, v] * (kw.qubo.sum(x[u,j] * x[v, j+1] for j in range(n-1)) + x[u,n-1] * x[v, 0]) for u, v in edges)

    # 最终的目标函数，其中哈密尔顿约束的惩罚系数为20
    obj = 20 * ham_cycle + path_cost

    # 解析QUBO
    obj = kw.qubo.make(obj)

    # 转化为Ising模型
    obj_ising = kw.qubo.cim_ising_model(obj)

    # 提取Ising矩阵
    matrix = obj_ising.get_ising()["ising"]

    pump_num = 1.5
    noise_num = 0.03
    nor_num = 0.5
    laps_num = int(target_num/agv_need*40)
    dt_num = 0.1
    matrix_n = kw.cim.normalizer(matrix, nor_num)

    # 使用CIM模拟器进行计算
    output,ham_output = kw.cim.simulator(
                    matrix,
                    pump = pump_num,
                    noise = noise_num,
                    laps = laps_num,
                    dt = dt_num,
                    normalization = nor_num,
                    iterations = int(target_num/agv_need))
    
    # 对结果进行排序
    opt = kw.sampler.optimal_sampler(matrix, output, bias=0, negtail_ff=False)

    # 选取其中最好的50个解
    cim_best = opt[0][:50]

    # 对这50个解分别使用禁忌搜索在其邻域进行搜索
    tsbest_s, tsbest_h = kw.utils.tabu_optimizer(cim_best, matrix, max_samp=50, gap=2)

    # 选取最好的解
    ts_best = tsbest_s[0]

    # 如线性项变量为-1，进行反转
    ts_best = ts_best * ts_best[-1]

    #print(ts_best)

    # 得到变量名的列表
    vars = obj_ising.get_variables()

    # 代入spin向量，求得结果字典
    sol_dict = kw.qubo.get_sol_dict(ts_best, vars)

    # 代入硬约束验证是否为合法路径
    seq_val = kw.qubo.get_val(sequence_cons, sol_dict)
    node_val = kw.qubo.get_val(node_cons, sol_dict)
    ham_val = kw.qubo.get_val(ham_cycle, sol_dict)
    print('position cons: {}'.format(seq_val))
    print('node_cons cons: {}'.format(node_val))
    print('ham_cycle: {}'.format(ham_val))

    # 代入path_cost得到环路长度
    path_val = kw.qubo.get_val(path_cost, sol_dict)
    path_val = path_val / divide_index
    total_path = total_path + path_val
    print('path_cost: {}'.format(path_val))

    if ham_val == 0:
        print('valid path')
    
        # 获得x的数值矩阵
        x_val = kw.qubo.get_array_val(x, sol_dict)
    
        # 找到其中的非0项的脚标
        nonzero_index = np.array(np.nonzero(x_val)).T
    
        # 对非零项的顺序脚标进行排序
        nonzero_index = nonzero_index[nonzero_index[:, 1].argsort()]

        # 打印路径顺序
        print(nonzero_index[:,0].flatten())
    else:
        print('invalid path')
        

    # 反馈结果画图   
    xvalue = extract_data[nonzero_index[:,0],0]
    xvalue = np.append(xvalue,xvalue[0])
    yvalue = extract_data[nonzero_index[:,0],1]
    yvalue = np.append(yvalue,yvalue[0])
    plt.subplot(2, 2, 2)
    plt.plot(xvalue,yvalue)
    plt.subplot(2, 2, 3)
    h = kw.sampler.hamiltonian(matrix_n, ham_output)
    plt.plot(h, linewidth=1)
    plt.subplot(2, 2, 4)
    plt.plot(ham_output, linewidth=1)


# 打印结果
time_end = time.time()
print("computing time："+str(time_end - time_start)+"秒")
print('number of vehicle: {}'.format(agv_need))  
print('total_path: {}'.format(total_path))
plt.show()


obj_ising = kw.qubo.cim_ising_model(obj)

matrix = obj_ising.get_ising()["ising"]

pump_num = 1.5
noise_num = 0.03
nor_num = 0.5
laps_num = int(target_num/agv_need*200)
dt_num = 0.1
matrix_n = kw.cim.normalizer(matrix, nor_num)

output,ham_output = kw.cim.simulator(
                matrix,
                pump = pump_num,
                noise = noise_num,
                laps = laps_num,
                dt = dt_num,
                normalization = nor_num,
                iterations = int(target_num/agv_need))

opt = kw.sampler.optimal_sampler(matrix, output, bias=0, negtail_ff=False)
cim_best = opt[0][:50]
tsbest_s, tsbest_h = kw.utils.tabu_optimizer(cim_best, matrix, max_samp=50, gap=2)
ts_best = tsbest_s[0]
ts_best = ts_best * ts_best[-1]
vars = obj_ising.get_variables()
sol_dict = kw.qubo.get_sol_dict(ts_best, vars)
