

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy  as sch
from scipy.spatial.distance import pdist
import pdb
import sys
import time

class Simulation():
    def __init__(self, no_iterations=500, 
                       interactions_per_iteration=5000, 
                       no_of_agents=500, 
                       p_ER_graph=0.1, 
                       friction=.1, 
                       no_of_statements=1, 
                       truth=[], 
                       fraction_passive=.75, 
                       annoyance_threshold_per_interaction=0.8,
                       triadic_closure_prob=.75):
        # save parameters
        self.no_iterations = no_iterations
        self.interactions_per_iteration = interactions_per_iteration
        self.no_of_agents = no_of_agents 
        self.p_ER_graph = p_ER_graph
        self.friction = friction
        self.no_of_statements = no_of_statements 
        self.truth = truth
        self.fraction_passive = fraction_passive
        self.annoyance_threshold_per_interaction = annoyance_threshold_per_interaction
        self.annoyance_threshold = annoyance_threshold_per_interaction * no_of_agents * 1./interactions_per_iteration
        self.triadic_closure_prob = triadic_closure_prob
        
        # initialize statements/truth
        if len(self.truth) < no_of_statements:
            self.truth.append(np.random.binomial(1, .5, 1)[0])
        self.truth = np.asarray(self.truth) * 2 - 1
        assert isinstance(self.truth[0], np.int64)
        
        # initialize agents
        self.beliefs = [np.zeros(self.no_of_agents) for _ in range(self.no_of_statements)]
        self.truth_lookup_capacity = np.random.uniform(0, 1, self.no_of_agents) * np.random \
                                                 .binomial(1, self.fraction_passive, self.no_of_agents)
        # initialize network
        self.adjacency = np.random.binomial(1, self.p_ER_graph * .5, self.no_of_agents**2)  \
                                           .reshape((self.no_of_agents, self.no_of_agents))
        # set main diagonal 0
        for i in range(self.no_of_agents):
            self.adjacency[i][i] = 0 
        # make matrix symmetric, note that because of the maximum function, we need to divide the ER graph p by 2 above
        self.adjacency = np.maximum(self.adjacency, self.adjacency.T)      
        # prepare matrix to collect interaction data; links are disconnected whereever this exceeds a defined threshold
        self.annoyance = np.zeros(self.no_of_agents**2).reshape((self.no_of_agents, self.no_of_agents))
        
        # seed opinions
        for item in range(len(self.truth)):
            i = np.random.randint(self.no_of_agents)
            self.beliefs[item][i] = 1.
            i = np.random.randint(self.no_of_agents)
            self.beliefs[item][i] = -1.
    
    def run(self):  # controls simulation run, defines sequence of events
        ctime = time.time()
        for t in range(self.no_iterations):
            oldtime, ctime = ctime, time.time()
            print("Iteration {0:5d}/{1:d}, Iteration computation time: {2:10f}" \
                                                .format(t+1, self.no_iterations, ctime - oldtime), end="\r")
            # interactions, each with equal probability for all edges
            for _ in range(self.interactions_per_iteration):
                i, j = self.select_edge(existing = True)
                self.interact(i, j)
            
            # modify network: remove edges
            links_cut = self.cut_links()
            
            # modify network: add edges (comment in one of the two method calls)
            self.add_links(links_cut)            # triadic closure part collects all open triads and select from them
            #self.add_links_tc_ad_hoc(links_cut)  # triadic closure part selects random triples and check whether they
                                                  #        ... are open triads. Will be faster for non-sparse networks
            
            # reset annoyance matrix
            self.annoyance = np.zeros(self.no_of_agents**2).reshape((self.no_of_agents, self.no_of_agents))
            
            #save data - TODO (not implemented)
            self.save_data(t)
        
        #create output - TODO (not implemented)
        print("")
        self.evaluate_results()

    def select_edge(self, existing=True): # select existing or missing edge at random 
        adjacency_value = 1 if existing else 0
        criterion = False
        while not criterion:
            i = j = np.random.randint(self.no_of_agents)
            while i==j:
                j = np.random.randint(self.no_of_agents)
            if self.adjacency[i][j] == adjacency_value:
                criterion = True
        return i, j 

    def interact(self, a1, a2): # conduct opinion dynamics interaction between agents #a1 and #a2
        for item in range(len(self.truth)): 
            b1, b2 = self.beliefs[item][a1], self.beliefs[item][a2]
            if b1 * b2 <= 1:
                if self.truth_lookup_capacity[a1] > 0:
                    self.beliefs[item][a1] = b1 + (self.truth[item] - b1) * self.truth_lookup_capacity[a1]
                else:
                    self.beliefs[item][a1] = b1 * (1 - self.friction) + b2 * self.friction
                if self.truth_lookup_capacity[a2] > 0:
                    self.beliefs[item][a2] = b2 + (self.truth[item] - b2) * self.truth_lookup_capacity[a2]
                else:
                    self.beliefs[item][a2] = b2 * (1 - self.friction) + b1 * self.friction
                self.annoyance[a1][a2] += 1
                self.annoyance[a2][a1] += 1
            else:
                self.beliefs[item][a1] = b1 * (1 - self.friction) + b2 * self.friction
                self.beliefs[item][a2] = b2 * (1 - self.friction) + b1 * self.friction                

    def cut_links(self):    # cut every link for which the annoyance threshold has been exceeded
        links_cut = 0
        for i in range(self.no_of_agents):
            for j in range(i + 1, self.no_of_agents):
                if self.annoyance[i][j] >= self.annoyance_threshold:
                    self.adjacency[i][j] = 0
                    self.adjacency[j][i] = 0
                    #self.annoyance[i][j] = 0
                    #self.annoyance[j][i] = 0
                    links_cut += 1
        return links_cut

    def add_links(self, number):    # create <number> new links with triadic closure and at random
        
        if number != 0 and self.triadic_closure_prob != 0:
            # find open triads
            adj_2 = np.dot(self.adjacency, self.adjacency)  # matrix of distance 2 adjacency
            Z = 1 - (self.adjacency + np.identity(len(self.adjacency),dtype=int))   
            open_triads = np.multiply(Z, adj_2) # this is a matrix of the missing links that would close open triads
                                                # entries are 
                                                #   - either 0 (if the link is not missing or would not close any triads
                                                #   - or positive int (indicating the number of triads it would close)
                                                # the matrix entries should be used as weights

        # create links
        target_creation_w_tc = int(round(number * self.triadic_closure_prob))       #using triadic closure
        open_triads_flattened = open_triads.ravel()
        if np.count_nonzero(open_triads_flattened) < target_creation_w_tc:
            print("Warning: Not enough open triads, will add more random links. Continuing ...")
            target_creation_w_tc = np.count_nonzero(open_triads_flattened)
        matrix_len = len(open_triads)      # this is actually self.no_of_agents
        try:
            idxs = np.random.choice(matrix_len**2, target_creation_w_tc, False, p=open_triads_flattened/sum(open_triads_flattened))
        except:
            pdb.set_trace()
        for no in range(len(idxs)):
            i, j = idxs[no] // matrix_len, idxs[no] % matrix_len
            self.adjacency[i][j] = 1
            self.adjacency[j][i] = 1
        for _ in range(len(idxs), number):                                                 # using random connection
            i, j = self.select_edge(existing = False)
            self.adjacency[i][j] = 1
            self.adjacency[j][i] = 1
        
    def add_links_tc_ad_hoc(self, number):    # create <number> new links with triadic closure and at random
        
        # create links
        for no in range(number):
            if np.random.uniform(0, 1) > self.triadic_closure_prob:     # using triadic closure
                #current = np.random.randint(0, len(open_triads))
                #i, j, k = open_triads[i]
                #open_triads = open_triads[:current] + open_triads[current,i+1:]
                success = False
                while not success:
                    i, j, k = np.random.choice(self.no_of_agents, 3, False)
                    if sum([self.adjacency[i][j], self.adjacency[i][k], self.adjacency[j][k]]) == 2:
                        success = True
                # two of these will already be connected, but instead of looking this up, we set all connections 1
                self.adjacency[i][j] = 1
                self.adjacency[j][i] = 1
                self.adjacency[i][k] = 1
                self.adjacency[k][i] = 1
                self.adjacency[k][j] = 1
                self.adjacency[j][k] = 1
            else:                                                       # using random connection
                i, j = self.select_edge(existing = False)
                self.adjacency[i][j] = 1
                self.adjacency[j][i] = 1
        
    def network_metrics(self):      
        '''Method to analyze the current network structure and give graphical output'''
        # TODO: take apart into several methods
        # TODO: allow returning and/or saving of measures
        try:
            # get degree distribution
            degreeDist = [sum(row) for row in self.adjacency]
            dD_sorted = np.asarray(sorted(degreeDist))                              #degree Distribution cCDF x coordinate
            dD_sorted_y_cCDF = 1 - (np.arange(len(dD_sorted)) + 1) / len(dD_sorted)  #degree Distribution cCDF y coordinate
            
            fig, ax = plt.subplots(2, 2, figsize=(15, 9))
            ax[0, 0].plot(dD_sorted, dD_sorted_y_cCDF)
            ax[0, 0].set_title('lin-lin')
            ax[0, 0].set_xscale('linear')
            ax[0, 0].set_yscale('linear')
            ax[0, 0].set_ylabel("cCDF")
            ax[0, 1].plot(dD_sorted, dD_sorted_y_cCDF)
            ax[0, 1].set_title('lin-log')
            ax[0, 1].set_xscale('linear')
            ax[0, 1].set_yscale('log')
            ax[1, 0].plot(dD_sorted, dD_sorted_y_cCDF)
            ax[1, 0].set_title('log-lin')
            ax[1, 0].set_xscale('log')
            ax[1, 0].set_yscale('linear')
            ax[1, 0].set_ylabel("cCDF")
            ax[1, 0].set_xlabel("Node Degree")
            ax[1, 1].plot(dD_sorted, dD_sorted_y_cCDF)
            ax[1, 1].set_title('log-log')
            ax[1, 1].set_xscale('log')
            ax[1, 1].set_yscale('log')
            ax[1, 1].set_xlabel("Node Degree")
            plt.show()

            # transform adjacency matrix to networkx graph
            nxGraph = nx.from_numpy_matrix(self.adjacency, create_using=nx.DiGraph())
            nxGraph = nxGraph.to_undirected()
            #print(nx.dijkstra_path(nxGraph, 0, 1))
            
            # identify connected components
            connectedComponents = nx.connected_component_subgraphs(nxGraph)
            cComDist = [len(item) for item in connectedComponents]
            if len(cComDist) > 1:
                cCD_sorted = np.asarray(sorted(cComDist))                              #degree Distribution cCDF x coordinate
                cCD_sorted_y_cCDF = 1 - (np.arange(len(cCD_sorted)) + 1) / len(cCD_sorted)  #degree Distribution cCDF y coordinate
                
                plt.gca()
                plt.plot(cCD_sorted, cCD_sorted_y_cCDF)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel("Node Degree")
                plt.ylabel("cCDF")
                plt.show()
            else:
                print("Network has only one connected component")
                        
            # get clustering coefficients for all nodes
            clustCoefficients = nx.clustering(nxGraph)
            clustCoeffList = np.asarray(list(clustCoefficients.values()))
            print("Average clustering coefficient: {0:f}".format(sum(clustCoeffList)/len(clustCoeffList)))
            
            # get centrality measures
            bCentrality = nx.betweenness_centrality(nxGraph)            # betweenness centrality
            cCentrality = nx.closeness_centrality(nxGraph)              # closeness centrality
            eCentrality = nx.eigenvector_centrality(nxGraph)            # eigenvector centrality
            bcList = np.asarray(list(bCentrality.values()))
            ccList = np.asarray(list(cCentrality.values()))
            ecList = np.asarray(list(eCentrality.values()))
            
            fig, ax = plt.subplots(2, 2, figsize=(15, 9))
            ax[0, 0].scatter(ecList, bcList)
            ax[0, 0].set_ylabel("Betweenness Centrality")
            ax[0, 1].scatter(ecList, ccList)
            ax[0, 1].set_ylabel("Closeness Centrality")
            ax[1, 0].scatter(ecList, clustCoeffList)
            ax[1, 0].set_xlabel("Eigenvector Centrality")
            ax[1, 0].set_ylabel("Clustering Coefficient")
            ax[1, 1].scatter(ecList, degreeDist)
            ax[1, 1].set_xlabel("Eigenvector Centrality")
            ax[1, 1].set_ylabel("Node Degree")
            plt.show()
            
            # get distance matrix
            distanceDict = nx.all_pairs_shortest_path_length(nxGraph)
            
            distance = np.asarray([distanceDict[i][j] for i in range(len(distanceDict)) for j in \
                                  range(len(distanceDict))]).reshape(len(distanceDict),len(distanceDict))
            
            # get clustering for dendrogram
            dist_upper_triangle = pdist(distance)
            hierarchicalClustering = sch.linkage(dist_upper_triangle, method='ward')
            flat_cluster = sch.fcluster(hierarchicalClustering, 0.2*distance.max(), criterion='distance')
            
            plt.gca()
            sch.dendrogram(hierarchicalClustering, color_threshold=1, show_leaf_counts=True) #,labels=["","",...])
            plt.show()
            
        except:
            print(sys.exc_info())
            pdb.set_trace()
        
    def save_data(self, time):  # TODO (not implemented, should save data on iteration)
        pass
 
    def evaluate_results(self): # TODO (not implemented, should create output, save results, etc.)
        self.network_metrics()
        #pass
        
# main entry point
if __name__ == "__main__":
    S = Simulation()
    S.run()
