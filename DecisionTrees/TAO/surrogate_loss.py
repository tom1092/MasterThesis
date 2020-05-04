from DecisionTree import TreeNode, ClassificationTree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize
import random

X = np.random.uniform(size = (500, 10))
labels = np.random.randint(0, high=2, size= (500, ))
labels = [-1 if lab==0 else 1 for lab in labels]
clf = DecisionTreeClassifier(random_state=0, max_depth=1, min_samples_leaf=4)
clf.fit(X, labels)


def rennie_loss(threshold, node, X, labels):
    l = 0
    #l = sum((0.5-(X[i, node.feature]+threshold)*labels[i] if (X[i, node.feature]+threshold)*labels[i]<=0 else (0.5*(1-(X[i, node.feature]+threshold)*labels[i])**2 if (X[i, node.feature]+threshold)*labels[i]>0 and (X[i, node.feature]+threshold)*labels[i]<1 else 0)) for i in range(len(labels)))

    for i in range(len(X)):
        if (X[i, node.feature]+threshold)*labels[i]<=0:
            l+=0.5-(X[i, node.feature]+threshold)*labels[i]
        elif 0 < (X[i, node.feature]+threshold)*labels[i] and (X[i, node.feature]+threshold)*labels[i]<1:
            l+=0.5*(1-(X[i, node.feature]+threshold)*labels[i])**2

    return l/len(X)

def quadratically_loss(threshold, node, X, labels):
    l = 0
    gamma = 2
    for i in range(len(X)):
        if (X[i, node.feature]+threshold)*labels[i]>=1-gamma:
            l+=1/(2*gamma)*max(0, (1-(X[i, node.feature]+threshold)*labels[i])**2)
        else:
            l+=1-gamma/2-(X[i, node.feature]+threshold)*labels[i]
    return l/len(X)


def grad(node, X, labels, threshold):
    g = 0
    for i in range(len(X)):
        if (X[i, node.feature]+threshold)*labels[i]<=0:
            g+=-labels[i]
        elif 0 < (X[i, node.feature]+threshold)*labels[i] and (X[i, node.feature]+threshold)*labels[i]<1:
            g+=-labels[i]*(1-labels[i]*(X[i, node.feature]+threshold))
    return g/len(X)


def armijo(Delta_k, gamma, delta, node, X, labels, threshold):
    alfa = Delta_k
    dk = -grad(node, X, labels, threshold)

    while (rennie_loss(threshold+alfa*dk, node, X, labels)>rennie_loss(threshold, node, X, labels)+gamma*alfa*dk*grad(node, X, labels, threshold)):
        alfa = delta*alfa
    return alfa


def gradient_descend(alfa, node, X, labels):
    threshold = node.threshold
    Delta_k = alfa
    loss_before = rennie_loss(threshold, node, X, labels)
    #alfa = armijo(Delta_k, 0.6, 0.99, node, X, labels, threshold)
    threshold = threshold-alfa*grad(node, X, labels, threshold)
    loss_after = rennie_loss(threshold, node, X, labels)
    while(np.abs(loss_before-loss_after) > 1e-01):
        print("loss: ", rennie_loss(threshold, node, X, labels))
        loss_before = loss_after
        #alfa = armijo(Delta_k, 0.9, 0.99, node, X, labels, threshold)
        threshold = threshold-alfa*grad(node, X, labels, threshold)
        loss_after = rennie_loss(threshold, node, X, labels)

    return threshold


def zero_one_loss(threshold, node, X, labels):
    predictions = np.array([-1 if X[i, node.feature]+threshold<=0 else 1 for i in range(len(X))])
    mis_loss = np.count_nonzero(predictions-labels)/len(labels)
    return mis_loss


def mde(node, X, labels, pop_size=10, dimension=1, gen=5, cr=0.9, f=0.1):
    pop = np.ndarray(shape=(pop_size, dimension), dtype='float32')
    for i in range(dimension):
        pop[:, i] = np.random.uniform(size = pop_size)
    best = pop[0]
    print ("Minimizing surrogate with MDE + L-BFGS-B...")
    j = 0
    for generation in range(gen):

        for i in range(pop_size):
            indexes = list(range(0, pop_size))
            indexes.remove(i)
            k_indexes = random.sample(indexes, 3)
            trial = pop[i, 0] if np.random.random_sample() < cr else pop[k_indexes[0], 0] + f * (pop[k_indexes[1], 0]-pop[k_indexes[2], 0])
            #print("minimizzo con lbfgs...")
            trial = minimize(rennie_loss, trial, args = (node, X, labels), method='L-BFGS-B').x
            if rennie_loss(trial, node, X, labels) < rennie_loss(pop[i], node, X, labels):
                pop[i] = trial
            if rennie_loss(pop[i], node, X, labels) < rennie_loss(best, node, X, labels):
                best = pop[i]
    return best


#Qui faccio uno split parallelo agli assi
def best_parallel_split(node, X, labels):
    error_best = np.inf
    best_tresh = 0
    best_feature = 0
    for j in range(len(X[0])):
        #Prendo tutte le j-esime componenti del dataset e le ordino
        #in modo crescente
        vals = {}
        for point_idx in range(len(X)):
            vals[point_idx] = X[point_idx, j]

        values = sorted(vals.items(), key=lambda x: x[1])
        sorted_indexes = [tuple[0] for tuple in values]
        node.feature = j
        thresh = 0.5*X[sorted_indexes[0], j]
        node.threshold = -thresh
        actual_loss = zero_one_loss(node.threshold, node, X, labels)
        #Ciclo su ogni valore di quella componente e provo tutti gli split
        #possibili
        i = 0
        while i < len(sorted_indexes):
            if i < len(sorted_indexes)-1:
                thresh = 0.5*(X[sorted_indexes[i], j]+X[sorted_indexes[i+1], j])
            else:
                thresh = 1.5*X[sorted_indexes[i], j]
            node.threshold = -thresh
            actual_loss = zero_one_loss(node.threshold, node, X, labels)
            if actual_loss < error_best:
                error_best = actual_loss
                best_tresh = -thresh
                best_feature = j
            i+=1

    return best_tresh, best_feature


best = np.inf
best_f = 0
best_t = 0
node = TreeNode(0, 0, None, None, None, None, 0, 0, 0, None)
for j in range(len(X[0])):
    node.feature = j
    node.threshold = 1



    #opt_t = gradient_descend(1, node, X, labels)
    opt_t = minimize(quadratically_loss, 0, args = (node, X, labels), method='BFGS', tol = 1e-04).x
    print("ottimizzo feature: ", j)
    err = quadratically_loss(opt_t, node, X, labels)
    if err < best:
        best = err
        best_f = j
        best_t = opt_t


node.feature = best_f
node.threshold = best_t

#print(best)

print("best feature loss: ", best_f)
print("best threshold loss: ", best_t)
print("accuracy loss split", 1-zero_one_loss(node.threshold, node, X, labels))
par_t, par_f = best_parallel_split(node, X, labels)
print("best feature par: ", par_f)
print("best threshold par: ", par_t)
node.feature = par_f
node.threshold = par_t
print("accuracy par split", 1-zero_one_loss(node.threshold, node, X, labels))
print("clf feature: ", clf.tree_.feature[0])
print("clf thresh: ", clf.tree_.threshold[0])
print("clf score: ", clf.score(X, labels))
