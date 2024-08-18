from sklearn.cluster import KMeans
import numpy as np
import kaiwu as kw
import matplotlib.pyplot as plt
import math
import itertools
import random
import time

n_clust=13

time_start = time.time()
# AGV parameter setting
agv_type = 0 # Model 0
#agv_num = 30 # Number of dispatchable AGVs

agv_para = np.dtype({
    'names':['model','load','diatance'],
    'formats':['U10','f','i']
})
agv = np.array([('normal',500,1000),('advance',200,600)],dtype = agv_para) # Model 0:('normal',100,1000)；Model 1:('advance',120,1800)
print('Model is',agv[agv_type]['model'],', max capacity is',agv[agv_type]['load'],', max diatance is',agv[agv_type]['diatance'])
agv_capacity = agv[agv_type]['load']


# Target site file reading generation
data = np.loadtxt('C:/Users/ASUS/Desktop/CVRP/data.txt')
data = data[:,1:3]
load = np.loadtxt('C:/Users/ASUS/Desktop/CVRP/load.txt')
load = load[:,1:2]
point_num = len(data) # Number of target sites

#plt.subplot(2, 2, 1)
#plt.scatter(data[:,0],data[:,1])
#plt.xlabel('X (m)')
#plt.ylabel('Y (m)')

# Angle calculation
angle = np.zeros(point_num)
for i in range (point_num):
    angle[i] = math.atan2((data[i,1]-data[0,1]),(data[i,0]-data[0,0]))
    angle[i] = (angle[i] * 180 / math.pi)
angle = np.reshape(angle,(point_num,1))

# Consolidated data (x,y,angle.load)
location = data
location_a = np.append(location,angle,axis = 1)
location_L = np.append(location,load,axis = 1)
#print(location_L)

n_cluster = KMeans(n_clusters=n_clust, random_state=0, n_init="auto").fit(location_a)
n_predict = n_cluster.predict(location_a)
#plt.scatter(location[:,0],location[:,1],c=n_predict)
centroid=n_cluster.cluster_centers_
#print(centroid)

total_path = 0
for i in range(n_clust):
    res = location_L[(n_cluster.labels_ == i)]
    centroid[i][0] = int(centroid[i][0])
    centroid[i][1] = int(centroid[i][1])
    centroid[i][2] = 0
    res = np.vstack((centroid[i],res))
    plt.subplot(2, 2, 2)
    plt.scatter(res[:,0],res[:,1])
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    # Angle calculation
    point_num = len(res)
    angle = np.zeros(point_num)
    for i in range (point_num):
        angle[i] = math.atan2((res[i,1]-res[0,1]),(res[i,0]-res[0,0]))
        angle[i] = (angle[i] * 180 / math.pi)
    angle = np.reshape(angle,(point_num,1))
    res_loc = res[:,0:2]
    res_load = res[:,2:3]
    res_loc = np.append(res_loc,angle,axis = 1)
    res_all = np.append(res_loc,res_load,axis = 1)
    #print(res)
    
    # Data segmentation
    start_point = res_all[0]
    start_point = start_point[0:2]
    print("Start Point Location:",start_point)
    target_point = res_all[1:]
    target_point = target_point[np.argsort(target_point[:,2])]
    arrange_point = res
    #print("Target Point Location:",target_point)

    # Parameter initialisation
    load_index = 0
    agv_need = 1
    target_num = point_num - 1
    d_matrix = []
    d_length = len(arrange_point)
    for i in range(d_length):
        for j in range(d_length):
            d_dist = np.sqrt((arrange_point[i][0]-arrange_point[j][0])**2+(arrange_point[i][1]-arrange_point[j][1])**2)
            d_matrix.append(d_dist)
    d_matrix = np.matrix(d_matrix)
    d_matrix = np.around(d_matrix, decimals = 2)
    d_matrix = d_matrix.reshape(d_length,d_length)
    save_cost = []
    for i in range(d_length-1):
        if i < d_length-2:
            save_dist = d_matrix[0][i+1]+d_matrix[0][i+2]-d_matrix[i+1][i+2]
        else:
            save_dist = d_matrix[0][i+1]+d_matrix[0][1]-d_matrix[i+1][1]
        save_cost.append(save_dist)
    target_index = np.argmin(save_cost)+2
    start_target_point = target_index

    # Calculation of vehicle requirements
    for index in range(target_num):
        if target_index > target_num:
            target_index = target_index - target_num
        load_index = load_index + target_point[target_index - 1][3]
        if load_index > agv_capacity:
            agv_need = agv_need + 1
            load_index = target_point[target_index - 1][3]
        target_index = target_index + 1

    # Cluster initialisation
    cluster = np.zeros(4*agv_need*point_num).reshape(agv_need,point_num,4)

    # Parameter initialisation
    point_index = 0
    load_index = 0
    cluster_index = 0
    bit = target_index
    target_index = start_target_point
    count = 0

    # Target site (in the form of a Cluster)
    for index in range(target_num):
        if target_index > target_num:
            target_index = target_index - target_num
        load_index = load_index + target_point[target_index - 1][3]
        if load_index > agv_capacity:
            cluster[cluster_index][0] = np.asarray([res_all[0]])
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
        cluster[cluster_index][0] = np.asarray([res_all[0]])
        for i in range(1,target_num-count+1):
            if bit > target_num:
                bit = bit - target_num
            cluster[cluster_index][i] = target_point[bit - 1]
            bit = bit + 1
    #print("Cluster form:",cluster)


    # TSP algorithm
    for i in range(agv_need):

        
        # Extraction of coordinate data for each agv
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


        # Drawing of original coordinate points
        plt.subplot(2, 2, 2)
        plt.scatter(extract_data[:,0],extract_data[:,1])
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        extract_data[:,0] = extract_data[:,0]
        extract_data[:,1] = extract_data[:,1]


        # Distance matrix creation
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

        pump_num = 1.4
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
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.subplot(2, 2, 3)
        h = kw.sampler.hamiltonian(matrix_n, ham_output)
        plt.plot(h, linewidth=1)
        plt.xlabel('Laps')
        plt.ylabel('Energy')
        plt.subplot(2, 2, 4)
        plt.plot(ham_output, linewidth=1)
        plt.xlabel('Laps')
        plt.ylabel('Qubit Phase')

# print result       
time_end = time.time()
print("computing time："+str(time_end - time_start)+"秒")
print('number of vehicle: {}'.format(agv_need))  
print('total_path: {}'.format(total_path))  
plt.show()


