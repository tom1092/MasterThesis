import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import random

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
                self.classification_tree.build_idxs_of_subtree(X, range(len(X)), 0)

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
                        left_label_predicted = T.predict_label(X[data_idx].reshape((1, -1)), node.left_node_id)
                        right_label_predicted = T.predict_label(X[data_idx].reshape((1, -1)), node.right_node_id)
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
        self.X = []
        self.y = []


    def evolve(self, X, y):
        self.X = X
        self.y = y
        i = 0
        to_delete = []
        while (True):
            error_prev = self.classification_tree.misclassification_loss(X, y, 0)
            keys = list(self.classification_tree.tree.keys())
            random.shuffle(keys)
            for node_id in keys:
                nodes_to_del = self.optimize_node_parallel(node_id, X, y)
                print(nodes_to_del)
                for node in nodes_to_del:
                    self.delete_node(node)
                    #to_delete.append(node)
                self.classification_tree.build_idxs_of_subtree(X, range(len(X)), 0)
                error_curr = self.classification_tree.misclassification_loss(X, y, 0)

            #for node_id in to_delete:
                #self.delete_node(node_id)
            if error_curr == error_prev:
                break
        return to_delete


    #Mette il sottoalbero con radice in node_id_B, al posto di quello con radice
    #in node_id_A
    def replace_node(self, node_id_A, node_id_B):
        tree = self.classification_tree.tree

        tree[node_id_A].left_node_id = tree[node_id_B].left_node_id
        tree[node_id_A].right_node_id = tree[node_id_B].right_node_id
        tree[node_id_A].feature = tree[node_id_B].feature
        tree[node_id_A].threshold = tree[node_id_B].threshold
        tree[node_id_A].is_leaf = tree[node_id_B].is_leaf
        tree[node_id_A].value = tree[node_id_B].value
        tree[node_id_A].data_idxs = tree[node_id_B].data_idxs

        #Devo dire ai figli che il padre ha l'id di A, se B non è una foglia
        parent_a = tree[node_id_A].parent_id
        depth_a = tree[node_id_A].depth
        id_a = tree[node_id_A].id
        if not tree[node_id_B].is_leaf:
            left_id = tree[node_id_B].left_node_id
            right_id = tree[node_id_B].right_node_id
            tree[left_id].parent_id = id_a
            tree[right_id].parent_id = id_a
            tree[right_id].depth = depth_a + 1
            tree[left_id].depth = depth_a + 1

        #del tree[node_id_B]


    def delete_node(self, node_id):
        T = self.classification_tree.tree
        stack = [node_id]
        while (len(stack) > 0):
            actual_node = stack.pop()
            if not T[actual_node].is_leaf:
                stack.append(T[actual_node].left_node_id)
                stack.append(T[actual_node].right_node_id)
            T.pop(actual_node)

    def optimize_node_parallel(self, node_id, X, y):

        #Tengo la lista di nodi che sono stati scambiati, scambiandosi con i figli
        #devo eliminarli alla fine
        to_delete = []
        if self.classification_tree.tree[node_id].is_leaf:
            node = self.classification_tree.tree[node_id]
            best_class = np.bincount(y[node.data_idxs]).argmax()
            node.value = best_class
        else:
            node_lower_id = self.classification_tree.tree[node_id].left_node_id
            node_upper_id = self.classification_tree.tree[node_id].right_node_id
            error_best = self.classification_tree.misclassification_loss(X[self.classification_tree.tree[node_id].data_idxs], y[self.classification_tree.tree[node_id].data_idxs], node_id)

            #Provo lo split
            error_para, best_t, best_feature, params_prec = self.best_parallel_split(node_id, node_lower_id, node_upper_id, X, y)
            if error_para < error_best:
                error_best = error_para
            if error_para >= error_best:
                #Rimetto i vecchi parametri al nodo
                self.classification_tree.tree[node_id].feature = params_prec[0]
                self.classification_tree.tree[node_id].threshold = params_prec[1]

            #Errore sul figlio sinistro, se è migliore allora sostituisco
            error_lower = self.classification_tree.misclassification_loss(X[self.classification_tree.tree[node_id].data_idxs], y[self.classification_tree.tree[node_id].data_idxs], node_lower_id)
            if error_lower < error_best:
                self.replace_node(node_id, node_lower_id)
                to_delete.append(node_lower_id)
                error_best = error_lower

            #Errore sul figlio sinistro, se è migliore allora sostituisco
            error_upper = self.classification_tree.misclassification_loss(X[self.classification_tree.tree[node_id].data_idxs], y[self.classification_tree.tree[node_id].data_idxs], node_upper_id)
            if error_upper < error_best:
                self.replace_node(node_id, node_upper_id)
                to_delete.append(node_upper_id)
                error_best = error_upper
        return to_delete


    def best_parallel_split(self, node_id, node_lower_id, node_upper_id, X, y):
        error_best = np.inf
        T = self.classification_tree
        #mi salvo i parametri del nodo precedenti all'ottimizzazione
        params_prec = [T.tree[node_id].feature, T.tree[node_id].threshold]
        for j in range(len(X[0])):
            #Prendo tutte le j-esime componenti del dataset e le ordino
            #in modo crescente
            vals = {}
            for point_idx in T.tree[node_id].data_idxs:
                vals[point_idx] = X[point_idx, j]

            values = sorted(vals.items(), key=lambda x: x[1])
            sorted_indexes = [tuple[0] for tuple in values]
            #plt.scatter(X[sorted_indexes, j], range(len(values)), s=0.4, c=list(correct_classification_tuples.values()))
            #plt.show()
            T.tree[node_id].feature = j
            thresh = 0.5*X[sorted_indexes[0], j]
            actual_loss = self.base_loss(node_id, X, y, sorted_indexes, thresh)
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
                sum, k = self.binary_loss(X, j, sorted_indexes, i, y)
                actual_loss += sum
                if actual_loss < error_best:
                    error_best = actual_loss
                    best_t = thresh
                    best_feature = j
                i+=k

        T.tree[node_id].threshold = best_t
        T.tree[node_id].feature = best_feature
        return error_best, best_t, best_feature, params_prec


    #Definico la loss base del nodo.
    #Significa che parto considerando come thresh la metà del minimo valore su quella componente.
    #Questo mi consente di evitare un O(n) e quindi
    #di ciclare su ogni punto del dataset per calcolare di nuovo la loss cambiando il
    #thresh. Ad ogni cambio, essendo i dati ordinati per componente la loss cambia
    #al più di una unità +/- 1. OCCORRE FARE ATTENZIONE A PUNTI UGUALI! VEDI BINARY LOSS
    def base_loss(self, node_id, X, y, care_points_idxs, thresh):
        loss = 0
        T = self.classification_tree.tree
        #Per ogni punto del nodo verifico dove questo lo inoltrerebbe
        #se il punto viene inoltrato su un figlio che porta a misclassificazione
        #è errore
        for data_idx in care_points_idxs:
            if X[data_idx][T[node_id].feature] <= thresh:
                if y[data_idx] != 0:
                    loss+=1
            else:
                if y[data_idx] != 1:
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
