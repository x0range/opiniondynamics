

import numpy as np
#import networkx
#import pdb

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
        for t in range(self.no_iterations):
            print("Iteration {0:5d}/{1:d}".format(t,self.no_iterations), end="\r")
            # interactions, each with equal probability for all edges
            for _ in range(self.interactions_per_iteration):
                i, j = self.select_edge(existing = True)
                self.interact(i, j)
            
            # modify network
            links_cut = self.cut_links()
            self.add_links(links_cut)
            
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
        
        # finding all open triads is too computationally intensive (provided the network is sufficiently dense),
        #                                                                       so just choose at random below
        #if number != 0 and self.triadic_closure_prob != 0:
        #    # find open triads
        #    open_triads = []
        #    for i in range(self.no_of_agents):
        #        for j in range(i + 1, self.no_of_agents):
        #            for k in range(j + 1, self.no_of_agents):
        #                if sum([self.adjacency[i][j], self.adjacency[i][k], self.adjacency[j][k]]) == 2:
        #                    open_triads.append([i, j, k])
        
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
        
    def save_data(self, time):  # TODO (not implemented, should save data on iteration)
        pass
 
    def evaluate_results(self): # TODO (not implemented, should create output, save results, etc.)
        pass
        
# main entry point
if __name__ == "__main__":
    S = Simulation()
    S.run()
