from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import euclidean_distances,classification_report
from sklearn.preprocessing import MinMaxScaler

data = load_iris()
x = data['data']
y = data['target']
minmax = MinMaxScaler()
x = minmax.fit_transform(x)

R = 2
n_classes = 3
epsilon = 0.9
epsilon_dec_factor = 0.001

class prototype(object):
    def __init__(self,class_id,p_vector,epsilon):
        self.class_id = class_id
        self.p_vector = p_vector
        self.epsilon = epsilon

    def update(self,u_vector,increment = True):
        if increment:
            self.p_vector = self.p_vector + self.epsilon*(u_vector - self.p_vector)
        else:
            self.p_vector = self.p_vector - self.epsilon*(u_vector - self.p_vector)

def find_closest(in_vector,proto_vectors):
    closest = None
    closest_distance = 99999
    for p_v in proto_vectors:
        distance = euclidean_distances(in_vector.reshape(1,4),p_v.p_vector.reshape(1,4))
        if distance < closest_distance:
            closest_distance = distance
            closest = p_v
    return closest
def find_class_id(test_vector,p_vectors):
    return find_closest(test_vector,p_vectors).class_id

p_vectors = []
for i in range(n_classes):
    y_subset = np.where(y == 1)
    x_subset = x[y_subset]
    samples = np.random.randint(0,len(x_subset),R)
    for sample in samples:
        s = x_subset[sample]
        p = prototype(i,s,epsilon)
        p_vectors.append(p)
print("class id \t Initial protype vector\n")
for p_v in p_vectors:
    print(p_v.class_id,'\t',p_v.p_vector)
while epsilon >= 0.01:
    # 随机采样一个训练实例
    rnd_i = np.random.randint(0,149)
    rnd_s = x[rnd_i]
    target_y = y[rnd_i]

    # 为下一次循环减小ε
    epsilon = epsilon - epsilon_dec_factor
    # 查找与给定点最相近的原型向量
    closest_pvector = find_closest(rnd_s,p_vectors)

    # 更新最相近的原型向量
    if target_y == closest_pvector.class_id:
        closest_pvector.update(rnd_s)
    else:
        closest_pvector.update(rnd_s,False)
    closest_pvector.epsilon = epsilon


predicted_y = [find_class_id(instance,p_vectors) for instance in x ]

print(classification_report(y,predicted_y,target_names=['Iris-Setosa','Iris-Versicolour', 'Iris-Virginica']))
