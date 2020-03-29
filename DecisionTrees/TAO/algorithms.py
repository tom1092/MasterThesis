import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import random
from ClassificationTree import TreeNode

class TAO:
    def __init__(self, tree):
        self.classification_tree = tree
        self.X = []
        self.y = []

    def evolve(self, X, y):
        self.X = X
        self.y = y
        for i in range(1):
            for depth in reversed(range(self.classification_tree.tree_depth + 1)):
                #print("Ottimizzo depth", depth, "....")
                #print("ottimizzo depth: ", depth)
                nodes = self.classification_tree.get_nodes_at_depth(depth)
                #print ([node.id for node in nodes])
                T = self.classification_tree
                a = Parallel(n_jobs=1)(delayed(self.optimize_nodes)(node)
                                                        for node in nodes)


                #pool = Pool(4)
                #pool.map(self.optimize_nodes, nodes)
                #pool.close()
                #pool.join()


                #for node in nodes:
                    #self.optimize_nodes(node)

                #Rimetto apposto i punti associati ad ogni nodo
                self.classification_tree.build_idxs_of_subtree(X, range(len(X)), T.tree[0])

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
                        left_label_predicted = T.predict_label(X[data_idx].reshape((1, -1)), node.left_node)
                        right_label_predicted = T.predict_label(X[data_idx].reshape((1, -1)), node.right_node)
                        correct_label = y[data_idx]
                        if (left_label_predicted != right_label_predicted and right_label_predicted==correct_label):
                            correct_classification_tuples[data_idx] = 1
                            care_points_idxs.append(data_idx)
                        elif (left_label_predicted != right_label_predicted and left_label_predicted==correct_label):
                            care_points_idxs.append(data_idx)
                            correct_classification_tuples[data_idx] = 0
                    self.TAO_best_parallel_split(node.id, X, y, care_points_idxs, correct_classification_tuples, T)
                    #print("care_points: ", care_points_idxs)

                #print ("indici nodo prima", node.id, "--->", node.data_idxs)
                #prec = node.data_idxs
                #print (set(prec) == set(node.data_idxs))


    def TAO_best_parallel_split(self, node_id, X, y, care_points_idxs, correct_classification_tuples, T):

        error_best = np.inf
        best_t = T.tree[node_id].threshold
        best_feature = T.tree[node_id].feature
        #Itero sulle componenti
        #print ("Ottimizzo nodo:", node_id)
        if len(care_points_idxs) > 0:
            #print(len(care_points_idxs))
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
                actual_loss = self.base_loss(node_id, X, y, sorted_indexes, correct_classification_tuples, thresh)
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
                    sum, k = self.binary_loss(X, j, sorted_indexes, i, correct_classification_tuples)
                    actual_loss += sum


                    #Qualche print per debug
                    '''
                    if j==2 and node_id==3:
                        ones = [correct_classification_tuples[k] for k in sorted_indexes]
                        print("base: ", base, "  n punti:", len(sorted_indexes), "    loss:", actual_loss, "     thresh:", thresh, "     x:", X[sorted_indexes[i], j], "    prediction: ", correct_classification_tuples[sorted_indexes[i]])
                        print(X[sorted_indexes, j])
                        print()
                    '''
                    #actual_loss = self.vecchia_loss(node_id, X, y, care_points_idxs, correct_classification_tuples, thresh)
                    if actual_loss < error_best:
                        error_best = actual_loss
                        best_t = thresh
                        best_feature = j
                    i+=k

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
            #print("finale: ", error_best)
            T.tree[node_id].threshold = best_t
            T.tree[node_id].feature = best_feature



    #Definico la loss base del nodo.
    #Significa che parto considerando come thresh la metà del minimo valore su quella componente.
    #Questo mi consente di evitare un O(n) e quindi
    #di ciclare su ogni punto del dataset per calcolare di nuovo la loss cambiando il
    #thresh. Ad ogni cambio, essendo i dati ordinati per componente la loss cambia
    #al più di una unità +/- 1. OCCORRE FARE ATTENZIONE A PUNTI UGUALI! VEDI BINARY LOSS
    def base_loss(self, node_id, X, y, care_points_idxs, correct_classification_tuples, thresh):
        loss = 0
        T = self.classification_tree.tree
        #Per ogni punto del nodo verifico dove questo lo inoltrerebbe
        #se il punto viene inoltrato su un figlio che porta a misclassificazione
        #è errore
        for data_idx in care_points_idxs:
            if X[data_idx][T[node_id].feature] <= thresh:
                if correct_classification_tuples[data_idx] != 0:
                    loss+=1
            else:
                if correct_classification_tuples[data_idx] != 1:
                    loss+=1
        return loss




    def vecchia_loss(self, node_id, X, y, care_points_idxs, correct_classification_tuples, thresh):
        loss = 0
        T = self.classification_tree.tree
        #Per ogni punto del nodo verifico dove questo lo inoltrerebbe
        #se il punto viene inoltrato su un figlio che porta a misclassificazione
        #è errore
        for data_idx in care_points_idxs:
            if X[data_idx][T[node_id].feature] <= thresh:
                if correct_classification_tuples[data_idx] != 0:
                    loss+=1
            else:
                if correct_classification_tuples[data_idx] != 1:
                    loss+=1
        return loss


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


    def evolve(self, X, y, alfa=0):
        self.restore_tree()
        complexity = self.classification_tree.n_leaves - 1
        T = self.classification_tree.tree
        self.X = X
        self.y = y
        i = 0
        optimized = []

        while (i<1):
            error_prev = self.base_loss(T[0], X, y, T[0].data_idxs) + alfa*complexity

            values = list(self.classification_tree.tree.keys())
            random.shuffle(values)
            #print("values: ", values)
            while (len(values) > 0):
                #print(complexity)
                node_id = values.pop()
                optimized.append(node_id)
                #print("optimizing node:", node_id)
                self.optimize_node_parallel(T[node_id], X, y, alfa, complexity)
                #print("nodo ottimizzato:  ", node_id)
                ids = self.restore_tree()
                complexity = self.classification_tree.n_leaves - 1

                #print("ids: ", ids)
                values = list(set(ids)-set(optimized))
                print("values dopo restore:  ", values)
                self.classification_tree.build_idxs_of_subtree(X, range(len(X)), T[0])
            error_curr = self.base_loss(T[0], X, y, T[0].data_idxs) + alfa*complexity
            #print(self.max_id)
            #print("i-->", i, "node: ", node_id)
            #for node_id in to_delete:
                #self.delete_node(node_id)

            i+=1
            print("Ottimizzato nodi algoritmo ls: ", i, " volte")
            if np.abs(error_curr - error_prev) < 1e-01:
                break



    #Rimette apposto la struttura dati a dizionario usando una DFS
    def restore_tree(self):
        T = self.classification_tree.tree
        root_node = T[0]
        T.clear()
        stack = [root_node]
        ids = []
        depth = 0
        leaves = []
        while(len(stack) > 0):
            actual_node = stack.pop()
            if actual_node.depth > depth:
                depth = actual_node.depth
            T[actual_node.id] = actual_node
            ids.append(actual_node.id)
            if not actual_node.is_leaf:
                stack.append(actual_node.right_node)
                stack.append(actual_node.left_node)
            else:
                leaves.append(actual_node.id)
        self.classification_tree.depth = depth
        self.classification_tree.n_leaves = len(leaves)
        return ids

    #Crea due nuove foglie al nodo, le ottimizza per maggioranza e ritorna il nuovo sottoalbero
    def create_new_children(self, node, feature, threshold):

        node.is_leaf = False
        node.feature = feature
        node.threshold = threshold
        #print(self.max_id)
        id_left = self.max_id+1
        id_right = self.max_id+2
        left_child_node = TreeNode(id_left, node.depth+1, -1, -1, None, None, None, None, True, None)
        right_child_node = TreeNode(id_right, node.depth+1, -1, -1, None, None, None, None, True, None)
        left_child_node.parent_id = node.id
        right_child_node.parent_id = node.id
        node.left_node_id = id_left
        node.right_node_id = id_right
        node.left_node = left_child_node
        node.right_node = right_child_node

        self.classification_tree.build_idxs_of_subtree(self.X, node.data_idxs, node)
        bins = np.bincount(self.y[left_child_node.data_idxs])
        best_class_left = -1
        best_class_right = -1
        if len(bins > 0):
            best_class_left = bins.argmax()
        bins = np.bincount(self.y[right_child_node.data_idxs])
        if len(bins > 0):
            best_class_right = bins.argmax()
        left_child_node.value = best_class_left
        right_child_node.value = best_class_right

    #Mette il sottoalbero con radice in node_B, al posto di quello con radice
    #in node_A
    def replace_node(self, node_A, node_B):
        tree = self.classification_tree.tree

        #Salvo id del padre di A per potergli dire chi è il nuovo figlio
        parent_A_id = node_A.parent_id
        if parent_A_id != -1:
            parent_A = tree[parent_A_id]
            #Occorre capire se A era figlio destro o sinistro
            if parent_A.left_node_id == node_A.id:
                parent_A.left_node_id = node_B.id
                parent_A.left_node = node_B

            elif parent_A.right_node_id == node_A.id:
                parent_A.right_node_id = node_B.id
                parent_A.right_node = node_B

        node_B.parent_id = parent_A_id

        #Rimetto apposto le depth nel sottoalbero con root node_B
        node_B.depth = node_A.depth
        stack = [node_B]
        while (len(stack) > 0):
            actual_node = stack.pop()
            if not actual_node.is_leaf:
                actual_node.left_node.depth = actual_node.depth + 1
                actual_node.right_node.depth = actual_node.depth + 1
                stack.append(actual_node.left_node)
                stack.append(actual_node.right_node)



    def delete_node(self, node_id):
        T = self.classification_tree.tree
        stack = [node_id]
        while (len(stack) > 0):
            actual_node = stack.pop()
            if not T[actual_node].is_leaf:
                stack.append(T[actual_node].left_node_id)
                stack.append(T[actual_node].right_node_id)
            T.pop(actual_node)


    def optimize_node_parallel(self, node, X, y, alfa, complexity):


        error_best = self.base_loss(node, X, y, node.data_idxs) + alfa*complexity

        #print("prima parallel")
        #Provo lo split
        node_para, error_para = self.best_parallel_split(node, X, y, alfa, complexity)
        #print(node.id, "  fatto parallel plit")

        if error_para < error_best:
            #print("error para migliore")
            error_best = error_para
            self.replace_node(node, node_para)

        #Errore sul figlio sinistro, se è migliore allora sostituisco
        error_lower = self.base_loss(node_para.left_node, X, y, node.data_idxs) + alfa*complexity
        if error_lower < error_best:
            #print("error lower migliore")
            self.replace_node(node, node_para.left_node)
            error_best = error_lower

        #Errore sul figlio sinistro, se è migliore allora sostituisco
        error_upper = self.base_loss(node_para.right_node, X, y, node.data_idxs) + alfa*complexity
        if error_upper < error_best:
            #print("error upper migliore")
            self.replace_node(node, node_para.right_node)
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
        if new_node.is_leaf:
            was_leaf = True

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
                i = -1
                while i < len(sorted_indexes):

                    #print("Ottimizzo best parallel: ", i*100/len(sorted_indexes))
                    if i<0:
                        thresh = 0.5*X[sorted_indexes[0], j]
                    if i < len(sorted_indexes)-1:
                        thresh = 0.5*(X[sorted_indexes[i], j]+X[sorted_indexes[i+1], j])
                    else:
                        thresh = 1.5*X[sorted_indexes[i], j]

                    new_node.threshold = thresh
                    #Se il nodo da ottimizzare era una foglia allora dobbiamo creare le foglie figlie
                    #queste vengono ottimizzate subito per maggioranza
                    if was_leaf:
                        self.create_new_children(new_node, j, thresh)
                        actual_loss = self.base_loss(new_node, X, y, sorted_indexes) + alfa*(complexity+1)
                    else:
                        actual_loss = self.base_loss(new_node, X, y, sorted_indexes) + alfa*complexity

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


    #Definico la loss base del nodo.
    def base_loss(self, node, X, y, care_points_idxs):
        #Per ogni punto del nodo verifico la sua predizione
        if len(X[care_points_idxs]) > 0:
            n_misclassified = np.count_nonzero(y[care_points_idxs]-self.classification_tree.predict_label(X[care_points_idxs], node))
            return n_misclassified
        else:
            return 0

    #Definisco la loss in eq (4) per il problema al singolo nodo
    def binary_loss(self, X, node, sorted_indexes, i, targets, prec_tresh):

        y = targets
        #Prima vedo come era la loss totale sul punto/punti, ricorda che possono
        #esserci punti uguali con label differenti
        thresh = node.threshold
        node.threshold = prec_tresh
        sum = 0
        k = 1
        pred = self.classification_tree.predict_label(X[sorted_indexes[i]].reshape((1, -1)), node)
        if pred != y[i]:
            #Era misclassificato
            sum+=1


        #Vedo se ci sono punti uguali che potrebbero dar fastidio con la loss.
        #Se mi fermassi a vedere solo il primo potrei ottenere che lo split migliore
        #sia uno tale che dopo ci sono altri punti uguali che però sono della classe sbagliata
        #se dopo provassi a splittare su quei punti la loss ovviamente si alzerebbe ma
        #nel frattempo, essendo uguali, mi sarei salvato come miglior split proprio il primo.
        #Occorre decidere se conviene eseguire lo split guardando insieme tutti i punti uguali
        #costruendo una loss per maggioranza
        #devo quindi pesare la loss su quanti punti sono classificati bene tra quelli uguali
        #dunque devo guardare i consecutivi
        while k+i < len(sorted_indexes) - 1 and X[sorted_indexes[k+i], node.feature] == X[sorted_indexes[i], node.feature]:
            pred = self.classification_tree.predict_label(X[sorted_indexes[k+i]].reshape((1, -1)), node)
            if pred != y[k+i]:
                #Era misclassificato
                sum+=1
            k+=1
        relative_loss_before = sum

        #Ora guardo cosa succede con il nuovo split
        node.threshold = thresh
        sum = 0
        k = 1
        pred = self.classification_tree.predict_label(X[sorted_indexes[i]].reshape((1, -1)), node)
        if pred != y[i]:
            #Era misclassificato
            sum+=1


        #Vedo se ci sono punti uguali che potrebbero dar fastidio con la loss.
        #Se mi fermassi a vedere solo il primo potrei ottenere che lo split migliore
        #sia uno tale che dopo ci sono altri punti uguali che però sono della classe sbagliata
        #se dopo provassi a splittare su quei punti la loss ovviamente si alzerebbe ma
        #nel frattempo, essendo uguali, mi sarei salvato come miglior split proprio il primo.
        #Occorre decidere se conviene eseguire lo split guardando insieme tutti i punti uguali
        #costruendo una loss per maggioranza
        #devo quindi pesare la loss su quanti punti sono classificati bene tra quelli uguali
        #dunque devo guardare i consecutivi
        while k+i < len(sorted_indexes) - 1 and X[sorted_indexes[k+i], node.feature] == X[sorted_indexes[i], node.feature]:
            pred = self.classification_tree.predict_label(X[sorted_indexes[k+i]].reshape((1, -1)), node)
            if pred != y[k+i]:
                #Era misclassificato
                sum+=1
            k+=1
        relative_loss_after = sum

        return relative_loss_after-relative_loss_before, k


    def predict_label_subtree(x, root_node):

        T = self.classification_tree.tree
        #Parto dalla root e definisco il percorso
        actual_node = root_node

        #Finchè non trovo una foglia
        while(not actual_node.is_leaf):
            #Decido quale sarà il prossimo figlio
            feature = actual_node.feature
            thresh = actual_node.threshold
            if x[feature] <= thresh:
                actual_node = T[actual_node.left_node_id]
            else:
                actual_node = T[actual_node.right_node_id]
        return actual_node.value
