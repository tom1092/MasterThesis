import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import random
from DecisionTree import TreeNode, ClassificationTree
from sklearn.svm import LinearSVC

class TAO:
    def __init__(self, tree, oblique = False):
        self.classification_tree = tree
        self.X = []
        self.y = []

    def evolve(self, X, y, n_iter = 5, min_size_prune = 1):
        self.X = X
        self.y = y
        for i in range(n_iter):
            for depth in reversed(range(self.classification_tree.depth + 1)):
                #print("Ottimizzo depth", depth, "....")
                T = self.classification_tree
                nodes = ClassificationTree.get_nodes_at_depth(depth, T)
                #print ([node.id for node in nodes])

                for node in nodes:
                    self.optimize_nodes(node)
                #pool = Pool(4)
                #pool.map(self.optimize_nodes, nodes)
                #pool.close()
                #pool.join()


                #for node in nodes:
                    #self.optimize_nodes(node)

                #Rimetto apposto i punti associati ad ogni nodo
                self.classification_tree.build_idxs_of_subtree(X, range(len(X)), T.tree[0], oblique = T.oblique)
        #Effettua il pruning finale per togliere dead branches e pure subtrees
        #self.prune(min_size = min_size_prune)

    def optimize_nodes(self, node):
            #print ("node id = ", node.id, "depth = ", node.depth)
            #print("ottimizzo nodo: ", node.id)
            T = self.classification_tree
            X = self.X
            y = self.y
            if node.data_idxs:
                if node.is_leaf :
                    #Se il nodo è una foglia ottimizzo semplicemente settando la predizione
                    #alla classe più comune tra le instanze che arrivano a quella foglia
                    #print(max(node.data_idxs))
                    best_class = np.bincount(y[node.data_idxs]).argmax()
                    node.value = best_class
                    care_points_idxs = node.data_idxs
                    #print("Node value: ", node.value)
                    #print("Best class: ", best_class)
                else:
                    #In questo caso devo risolvere un problema di minimizzazione
                    #I punti da tenere in considerazione sono solo quelli tali che la
                    #predizione dei due figli del nodo corrente è diversa, e in almeno
                    #uno il punto viene classificato correttamente
                    care_points_idxs = []
                    correct_classification_tuples = {}
                    for data_idx in node.data_idxs:
                        left_label_predicted = T.predict_label(X[data_idx].reshape((1, -1)), node.left_node, T.oblique)
                        right_label_predicted = T.predict_label(X[data_idx].reshape((1, -1)), node.right_node, T.oblique)
                        correct_label = y[data_idx]
                        if (left_label_predicted != right_label_predicted and right_label_predicted==correct_label):
                            correct_classification_tuples[data_idx] = 1
                            care_points_idxs.append(data_idx)
                        elif (left_label_predicted != right_label_predicted and left_label_predicted==correct_label):
                            care_points_idxs.append(data_idx)
                            correct_classification_tuples[data_idx] = 0
                    self.TAO_best_split(node.id, X, y, care_points_idxs, correct_classification_tuples, T)
                    #print("care_points: ", care_points_idxs)

                #print ("indici nodo prima", node.id, "--->", node.data_idxs)
                #prec = node.data_idxs
                #print (set(prec) == set(node.data_idxs))


    def TAO_best_split(self, node_id, X, y, care_points_idxs, correct_classification_tuples, T):

        #Itero sulle componenti
        #print ("Ottimizzo nodo:", node_id)
        if len(care_points_idxs) > 0:
            #print(len(care_points_idxs))
            if T.oblique:
                best_t, best_weights = self.TAO_best_SVM_split(node_id, X, y, care_points_idxs, correct_classification_tuples, T)
                T.tree[node_id].intercept = best_t
                T.tree[node_id].weights = best_weights
            else:
                best_t, best_feature = self.TAO_best_parallel_split(node_id, X, y, care_points_idxs, correct_classification_tuples, T)
                #print("finale: ", error_best)
                T.tree[node_id].threshold = best_t
                T.tree[node_id].feature = best_feature


    #Qui uso SVM per ottenere il miglior split obliquo
    def TAO_best_SVM_split(self, node_id, X, y, care_points_idxs, correct_classification_tuples, T):


        y = [correct_classification_tuples[i] for i in care_points_idxs]
        if not all(i == y[0] for i in y):
            clf = LinearSVC(tol=1e-6, C=10 , max_iter=10000, loss='squared_hinge', penalty = 'l1', dual=False)
            clf.fit(X[care_points_idxs], y)
            #print (clf.score(X[care_points_idxs], y))
            #weights = clf.coef_.reshape((len(X[0])),)
            #intercept = clf.intercept_
            #plt.scatter(X[care_points_idxs, 0], X[care_points_idxs, 1] , s=0.9, c=[y[i]*255 for i in range(len(y))])
            #x = np.linspace(0, 10, 100)
            #y = -(np.dot(weights[0], x) + intercept)/weights[1]
            #print(weights)
            #print(intercept)

            #plt.plot(x, y, c="black")
            #plt.show()

            #print(np.sign(np.dot(X[care_points_idxs], weights) + intercept))
            #print(y)
            #predictions = np.sign(np.dot(X[care_points_idxs], weights) + intercept)

            return clf.intercept_, clf.coef_.reshape((len(X[0])),)
        return T.tree[node_id].intercept, T.tree[node_id].weights


    #Qui faccio uno split parallelo agli assi
    def TAO_best_parallel_split(self, node_id, X, y, care_points_idxs, correct_classification_tuples, T):
        error_best = np.inf
        best_t = T.tree[node_id].threshold
        best_feature = T.tree[node_id].feature
        for j in range(len(X[0])):
            #Prendo tutte le j-esime componenti del dataset e le ordino
            #in modo crescente
            vals = {}
            for point_idx in care_points_idxs:
                vals[point_idx] = X[point_idx, j]

            values = sorted(vals.items(), key=lambda x: x[1])
            sorted_indexes = [tuple[0] for tuple in values]
            #plt.scatter(X[sorted_indexes, j], range(len(values)), s=0.4, c=list(correct_classification_tuples.values()))
            #plt.show()
            T.tree[node_id].feature = j
            thresh = 0.5*X[sorted_indexes[0], j]
            actual_loss = self.zero_one_loss(node_id, X, y, sorted_indexes, correct_classification_tuples, thresh)
            #if j==2:
                #base = actual_loss
            #actual_loss = self.binary_loss(node_id, X, y, sorted_indexes[i], correct_classification_tuples[sorted_indexes[i]], actual_loss, thresh)
            #print ("loss: ", actual_loss, "n punti: ", len(care_points_idxs))
            #print("vecchia: ", self.vecchia_loss(node_id, X, y, care_points_idxs, correct_classification_tuples, thresh))
            #Ciclo su ogni valore di quella componente e provo tutti gli split
            #possibili
            i = 0
            while i < len(sorted_indexes):
                if i < len(sorted_indexes)-1:
                    thresh = 0.5*(X[sorted_indexes[i], j]+X[sorted_indexes[i+1], j])
                else:
                    thresh = 1.5*X[sorted_indexes[i], j]
                #sum, k = self.binary_loss(X, j, sorted_indexes, i, correct_classification_tuples)
                #actual_loss += sum


                #Qualche print per debug
                '''
                if j==2 and node_id==3:
                    ones = [correct_classification_tuples[k] for k in sorted_indexes]
                    print("base: ", base, "  n punti:", len(sorted_indexes), "    loss:", actual_loss, "     thresh:", thresh, "     x:", X[sorted_indexes[i], j], "    prediction: ", correct_classification_tuples[sorted_indexes[i]])
                    print(X[sorted_indexes, j])
                    print()
                '''
                actual_loss = self.zero_one_loss(node_id, X, y, care_points_idxs, correct_classification_tuples, thresh)
                if actual_loss < error_best:
                    error_best = actual_loss
                    best_t = thresh
                    best_feature = j
                i+=1

            '''
            #print(error_best)
            #errors = [self.binary_loss(node_id, X, y, care_points_idxs, correct_classification_tuples, threshes[i]) for i in range(len(values)-1)]
            #min_index = np.argmin(errors)

        #Ancora print per debug
        if node_id==3:
            print("finale: ", error_best)
            print ("ones:", ones, " len:  ", len(ones), "   care_p_len:", len(care_points_idxs))
            print("best_feature:", best_feature, "   best_thresh:", best_t)

        '''
        return best_t, best_feature


    #Alla fine tolgo dead branches e pure subtrees
    def prune(self, min_size=1):
        #Prima controllo se ci sono sottoalberi puri.
        #Visito l'albero e verifico se esistono nodi branch ai quali arrivano punti associati a solo una label
        #Poi vedo se il nodo attuale è morto
        T = self.classification_tree
        stack = [T.tree[0]]
        while (stack):
            actual = stack.pop()
            if len(actual.data_idxs) > 0:
                #Se il nodo è puro
                if not actual.is_leaf and all(i == self.y[actual.data_idxs[0]] for i in self.y[actual.data_idxs]):
                    #Devo far diventare una foglia questo nodo con valore pari alla label
                    actual.is_leaf = True
                    actual.left_node = None
                    actual.right_node = None
                    actual.left_node_id = -1
                    actual.right_node_id = -1
                    actual.value = self.y[actual.data_idxs[0]]

                #Se il nodo ha un figlio morto devo sostituire il padre con l'altro figlio
                elif not actual.is_leaf and len(actual.left_node.data_idxs) < min_size:
                    stack.append(actual.right_node)
                    ClassificationTree.replace_node(actual, actual.right_node, T)
                elif not actual.is_leaf and len(actual.right_node.data_idxs) < min_size:
                    stack.append(actual.left_node)
                    ClassificationTree.replace_node(actual, actual.left_node, T)
                elif not actual.is_leaf:
                    stack.append(actual.right_node)
                    stack.append(actual.left_node)
            ClassificationTree.restore_tree(T)






    #Definico la loss base del nodo.
    #Significa che parto considerando come thresh la metà del minimo valore su quella componente.
    #Questo mi consente di evitare un O(n) e quindi
    #di ciclare su ogni punto del dataset per calcolare di nuovo la loss cambiando il
    #thresh. Ad ogni cambio, essendo i dati ordinati per componente la loss cambia
    #al più di una unità +/- 1. OCCORRE FARE ATTENZIONE A PUNTI UGUALI! VEDI BINARY LOSS
    def zero_one_loss(self, node_id, X, y, care_points_idxs, correct_classification_tuples, thresh):
        loss = 0
        T = self.classification_tree.tree
        node = T[node_id]
        #Per ogni punto del nodo verifico dove questo lo inoltrerebbe
        #se il punto viene inoltrato su un figlio che porta a misclassificazione
        #è errore
        targets = [correct_classification_tuples[x] for x in care_points_idxs]
        data = X[care_points_idxs]
        predictions = np.array([0 if sample[node.feature] <= thresh else 1 for sample in data[:,]])
        n_misclassified = np.count_nonzero(targets-predictions)
        return n_misclassified
        #return loss


    #Definisco la loss in eq (4) per il problema al singolo nodo
    def binary_loss(self, X, feature, sorted_indexes, i, targets):

        #Dato un punto del nodo verifico dove questo lo inoltrerebbe
        #se il punto viene inoltrato su un figlio che porta a misclassificazione
        #è errore
        #print("loss")
        sum = 0
        k = 1
        if targets[sorted_indexes[i]] == 0:
            sum -= 1
        else:
            sum +=1

        #Vedo se ci sono punti uguali che potrebbero dar fastidio con la loss.
        #Se mi fermassi a vedere solo il primo potrei ottenere che lo split migliore
        #sia uno tale che dopo ci sono altri punti uguali che però sono della classe sbagliata
        #se dopo provassi a splittare su quei punti la loss ovviamente si alzerebbe ma
        #nel frattempo, essendo uguali, mi sarei salvato come miglior split proprio il primo.
        #Occorre decidere se conviene eseguire lo split guardando insieme tutti i punti uguali
        #costruendo una loss per maggioranza
        #devo quindi pesare la loss su quanti punti sono classificati bene tra quelli uguali
        #dunque devo guardare i consecutivi
        while k+i < len(sorted_indexes) - 1 and X[sorted_indexes[k+i], feature] == X[sorted_indexes[i], feature]:
            if targets[sorted_indexes[k+i]] == 0:
                sum -= 1
            else:
                sum +=1
            k+=1
        return sum, k




class LocalSearch:
    def __init__(self, tree):
        self.classification_tree = tree
        self.max_id = len(self.classification_tree.tree)-1
        self.X = []
        self.y = []


    def evolve(self, X, y, alfa=0, max_iteration = 1):
        ClassificationTree.restore_tree(self.classification_tree)
        complexity = self.classification_tree.n_leaves - 1
        T = self.classification_tree.tree
        self.X = X
        self.y = y
        i = 0


        while (i<max_iteration):
            optimized = []
            error_prev = ClassificationTree.misclassification_loss(T[0], X, y, T[0].data_idxs, self.classification_tree.oblique) + alfa*complexity

            values = list(self.classification_tree.tree.keys())
            random.shuffle(values)
            #print(values)
            #print("values: ", values)
            while (len(values) > 0):
                #print(complexity)
                node_id = values.pop()
                optimized.append(node_id)
                #print("optimizing node:", node_id)
                self.optimize_node_parallel(T[node_id], X, y, alfa, complexity)
                #print("nodo ottimizzato:  ", node_id)
                ids = ClassificationTree.restore_tree(self.classification_tree)
                complexity = self.classification_tree.n_leaves - 1
                #print("complexity: ", complexity)
                #print("ids: ", ids)
                values = list(set(ids)-set(optimized))
                print("values dopo restore:  ", values)
                self.classification_tree.build_idxs_of_subtree(X, range(len(X)), T[0], self.classification_tree.oblique)
                error_curr = ClassificationTree.misclassification_loss(T[0], X, y, T[0].data_idxs, self.classification_tree.oblique) + alfa*complexity
            #print(self.max_id)
            #print("i-->", i, "node: ", node_id)
            #for node_id in to_delete:
                #self.delete_node(node_id)

            i+=1
            print("Ottimizzato nodi algoritmo ls: ", i, " volte")
            if np.abs(error_curr - error_prev) < 1e-01:
                break


    def optimize_node_parallel(self, node, X, y, alfa, complexity):


        rho = 0
        if node.is_leaf:
            rho = 1
        error_best = ClassificationTree.misclassification_loss(node, X, y, node.data_idxs, self.classification_tree.oblique) + alfa*complexity

        #print("prima parallel")
        #Provo lo split
        if self.classification_tree.oblique:
            node_para, error_para = self.best_svm_split(node, X, y, alfa, complexity)
        else:
            node_para, error_para = self.best_parallel_split(node, X, y, alfa, complexity)
        #print(node.id, "  fatto parallel plit")

        if error_para < error_best:
            #print("error para migliore")
            error_best = error_para
            ClassificationTree.replace_node(node, node_para, self.classification_tree)

        #Metto questo if perchè nel caso di svm se il nodo è una foglia pura allora
        #non riesco a fare SVM perchè giustamente si aspetta label differenti.
        #Questo non perde di generalità poichè se il nodo fosse una foglia pura allora
        #creando due nuovi figli e ottimizzando otterremmo un figlio foglia puro e l'altro
        #senza punti. Che avrebbe la stessa loss iniziale del padre. Quindi alla fine la foglia iniziale
        #non verrebbe branchata.
        if node_para.left_node != None and node_para.right_node != None:
            #Errore sul figlio sinistro, se è migliore allora sostituisco
            error_lower = ClassificationTree.misclassification_loss(node_para.left_node, X, y, node.data_idxs, self.classification_tree.oblique) + alfa*(complexity+rho)
            if error_lower < error_best:
                #print("error lower migliore")
                ClassificationTree.replace_node(node, node_para.left_node, self.classification_tree)
                error_best = error_lower

            #Errore sul figlio sinistro, se è migliore allora sostituisco
            error_upper = ClassificationTree.misclassification_loss(node_para.right_node, X, y, node.data_idxs, self.classification_tree.oblique) + alfa*(complexity+rho)
            if error_upper < error_best:
                #print("error upper migliore")
                ClassificationTree.replace_node(node, node_para.right_node, self.classification_tree)
                error_best = error_upper
        return node


    def best_parallel_split(self, node, X, y, alfa, complexity):

        T = self.classification_tree

        #Creo un nuovo sottoalbero fittizio con root nel nodo
        new_node = TreeNode.copy_node(node)
        error_best = np.inf
        #if new_node.is_leaf:
            #complexity += 1
        was_leaf = False
        improve = False
        rho = 0
        if new_node.is_leaf:
            was_leaf = True
            rho = 1

        if new_node.data_idxs:
            for j in range(len(X[0])):

                #Prendo tutte le j-esime componenti del dataset e le ordino
                #in modo crescente
                vals = {}
                for point_idx in new_node.data_idxs:
                    vals[point_idx] = X[point_idx, j]

                values = sorted(vals.items(), key=lambda x: x[1])
                sorted_indexes = [tuple[0] for tuple in values]

                #plt.scatter(X[sorted_indexes, j], range(len(values)), s=0.4, c=list(correct_classification_tuples.values()))
                #plt.show()
                new_node.feature = j
                #if j==2:
                    #base = actual_loss
                #actual_loss = self.binary_loss(node_id, X, y, sorted_indexes[i], correct_classification_tuples[sorted_indexes[i]], actual_loss, thresh)
                #print ("loss: ", actual_loss, "n punti: ", len(care_points_idxs))
                #print("vecchia: ", self.vecchia_loss(node_id, X, y, care_points_idxs, correct_classification_tuples, thresh))
                #Ciclo su ogni valore di quella componente e provo tutti gli split
                #possibili
                '''
                new_node.threshold = 0.5*X[sorted_indexes[0], j]
                '''
                i = -1
                actual_loss = ClassificationTree.misclassification_loss(new_node, X, y, sorted_indexes, self.classification_tree.oblique) + alfa*(complexity+rho)
                while i < len(sorted_indexes):

                    pre_thresh = new_node.threshold
                    #print("Ottimizzo best parallel: ", i*100/len(sorted_indexes))

                    if i<0:
                        thresh = 0.5*X[sorted_indexes[0], j]

                    if i < len(sorted_indexes)-1:
                        thresh = 0.5*(X[sorted_indexes[i], j]+X[sorted_indexes[i+1], j])
                    else:
                        thresh = 1.5*X[sorted_indexes[i], j]

                    new_node.threshold = thresh

                    '''
                    #Se il nodo da ottimizzare era una foglia allora dobbiamo creare le foglie figlie
                    #queste vengono ottimizzate subito per maggioranza
                    if was_leaf:
                        self.create_new_children(new_node, j, thresh)
                        inc, k = self.binary_loss(X, new_node, sorted_indexes, i, y, pre_thresh)
                        actual_loss += inc
                    else:
                        inc, k = self.binary_loss(X, new_node, sorted_indexes, i, y, pre_thresh)
                        actual_loss += inc

                    '''
                    #Se il nodo da ottimizzare era una foglia allora dobbiamo creare le foglie figlie
                    #queste vengono ottimizzate subito per maggioranza
                    if was_leaf:
                        ClassificationTree.create_new_children(new_node, X, y, self.max_id, j, thresh, oblique = T.oblique)

                    actual_loss = ClassificationTree.misclassification_loss(new_node, X, y, sorted_indexes, self.classification_tree.oblique) + alfa*(complexity+rho)


                    if actual_loss < error_best:
                        improve = True
                        error_best = actual_loss
                        best_t = thresh
                        best_feature = j
                    i+=1

            #print ("error best: ", error_best)
            new_node.threshold = best_t
            new_node.feature = best_feature
            if was_leaf and improve:
                self.max_id+=2
        return new_node, error_best


    def best_svm_split(self, node, X, y, alfa, complexity):
        T = self.classification_tree
        #Creo un nuovo sottoalbero fittizio con root nel nodo
        new_node = TreeNode.copy_node(node)
        #if new_node.is_leaf:
            #complexity += 1
        rho = 0
        data = X[new_node.data_idxs]
        label = y[new_node.data_idxs]
        if not all(i == label[0] for i in label) and len(data) > 0:
            #Questo fa SVM multiclasse 1 vs rest
            clf = LinearSVC(tol=1e-6, C=10 , max_iter=10000, loss='squared_hinge', penalty = 'l1', dual=False)
            clf.fit(data, label)
            #for n_class in range(n_classes):
                #n_misclassified = np.count_nonzero(label-np.sign(np.dot(data, clf.coef_[n_class].T)+clf.intercept_[n_class]))
            #Devo capire come ottenere i coefficienti migliori tra il numero di iperpiani addestrati
            weights = clf.coef_.reshape((len(X[0])),)
            intercept = clf.intercept_
            if new_node.is_leaf:
                ClassificationTree.create_new_children(node, X, y, self.max_id, None, None, oblique = T.oblique, weights=weights, intercept=intercept)
                rho = 1
            else:
                new_node.weights = weights
                new_node.intercept = intercept
            return new_node, ClassificationTree.misclassification_loss(new_node, X, y, new_node.data_idxs, T.oblique) + alfa*(complexity+rho)
        return new_node, np.inf
