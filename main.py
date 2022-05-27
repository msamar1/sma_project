import random
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import sys
import argparse



class Hissg_Graphs:
    def __init__(self):
        self.CASCADE_COEF=0.1
        self.colors_list = ["lightcoral", "gray", "lightgray", "firebrick", "red", "chocolate", "darkorange", "moccasin",
                       "gold","chartreuse", "forestgreen", "lime", "mediumaquamarine", "teal", "blue", "slateblue",
                       "cadetblue","turquoise", "blueviolet", "magenta", "lightsteelblue"]
        self.icm_current_infectious={}

    def draw_graph(self,graph):
        pos = nx.spring_layout(graph, seed=225)
        colors=nx.get_node_attributes(graph,'node_color')
        nx.draw_networkx(graph, pos=pos, node_size=100, node_color=list(colors.values()), font_size=10, width=0.5) #'#A0CBE2'
        plt.show()

    def read_mention_graph(self,weight_var_name='weight',number_of_rows=0):
        if number_of_rows == 0:
            df = pd.read_csv("./data/higgs-mention_network.edgelist/higgs-mention_network.edgelist", sep=" ", header=None,names=["side_a","side_b","weight"])
        else:
            df = pd.read_csv("./data/higgs-mention_network.edgelist/higgs-mention_network.edgelist", sep=" ", header=None,names=["side_a","side_b","weight"],nrows=number_of_rows)
        self.mention_graph = nx.from_pandas_edgelist(df,source="side_b",target="side_a",edge_attr="weight",create_using=nx.DiGraph())
        print(f"\nmention_graph:\nNumber of Nodes: {self.mention_graph.number_of_nodes()}\t\tnumber of Edges:{self.mention_graph.number_of_edges()}\n")
        # for nd in self.mention_graph.nodes:
        #     self.mention_graph.nodes[nd]["node_color"] = self.colors_list[random.randint(0,len(self.colors_list)-1)]
        # bcc = sorted(nx.strongly_connected_components(self.mention_graph), key=len, reverse=True)
        # print(f"biggest sub graph -->size:{len(bcc[0])} {bcc[0]}")
        # self.mention_biggest_subgraph = self.mention_graph.subgraph(bcc[0])  #
        # print(f"\tmention_biggest_subgraph:\nNumber of Nodes: {self.mention_biggest_subgraph.number_of_nodes()}\t\tnumber of Edges:{self.mention_biggest_subgraph.number_of_edges()}\n")
        return self.mention_graph

    def read_reply_graph(self,weight_var_name='weight', number_of_rows=0):
        if number_of_rows == 0:
            df = pd.read_csv("./data/higgs-reply_network.edgelist/higgs-reply_network.edgelist", sep=" ", header=None,names=["side_a","side_b",weight_var_name])
        else:
            df = pd.read_csv("./data/higgs-reply_network.edgelist/higgs-reply_network.edgelist", sep=" ", header=None,names=["side_a","side_b",weight_var_name],nrows=number_of_rows)
        self.reply_graph = nx.from_pandas_edgelist(df, source="side_b", target="side_a", edge_attr=weight_var_name,create_using=nx.DiGraph())
        print(f"\nreply_graph:\nNumber of Nodes: {self.reply_graph.number_of_nodes()}\t\tnumber of Edges:{self.reply_graph.number_of_edges()}\n")
        # for nd in self.reply_graph.nodes:
        #     self.reply_graph.nodes[nd]["node_color"] = self.colors_list[random.randint(0, len(self.colors_list) - 1)]
        # bcc = sorted(nx.strongly_connected_components(self.reply_graph), key=len, reverse=True)
        # print(f"biggest sub graph -->size:{len(bcc[0])} {bcc[0]}")
        # self.reply_biggest_subgraph = self.reply_graph.subgraph(bcc[0])  #
        # print(f"\treply_biggest_subgraph:\nNumber of Nodes: {self.reply_biggest_subgraph.number_of_nodes()}\t\tnumber of Edges:{self.reply_biggest_subgraph.number_of_edges()}\n")
        return self.reply_graph

    def read_retweet_graph(self,number_of_rows=0,weight_var_name="weight"):
        if number_of_rows==0:
            df = pd.read_csv("./data/higgs-retweet_network.edgelist/higgs-retweet_network.edgelist", sep=" ", header=None,names=["side_a","side_b","weight"])
        else:
            df = pd.read_csv("./data/higgs-retweet_network.edgelist/higgs-retweet_network.edgelist", sep=" ",header=None, names=["side_a", "side_b", "weight"], nrows=number_of_rows)
        self.retweet_graph = nx.from_pandas_edgelist(df, source="side_b", target="side_a", edge_attr=weight_var_name,create_using=nx.DiGraph())
        print(f"\nretweet_graph:\nNumber of Nodes: {self.retweet_graph.number_of_nodes()}\t\tnumber of Edges:{self.retweet_graph.number_of_edges()}\n")
        # for nd in self.retweet_graph.nodes:
        #     self.retweet_graph.nodes[nd]["node_color"] = self.colors_list[random.randint(0, len(self.colors_list) - 1)]
        # bcc = sorted(nx.strongly_connected_components(self.retweet_graph), key=len, reverse=True)
        # Gcc = sorted(nx.connected_components(self.retweet_graph), key=len, reverse=True)
        # print(f"biggest sub graph -->size:{len(bcc[0])} {bcc[0]}")
        # self.retweet_biggest_subgraph = self.retweet_graph.subgraph(bcc[0])  #
        # print(f"\tretweet_biggest_subgraph:\nNumber of Nodes: {self.retweet_biggest_subgraph.number_of_nodes()}\t\tnumber of Edges:{self.retweet_biggest_subgraph.number_of_edges()}\n")
        return self.retweet_graph

    def read_social_network_graph(self,number_of_rows=0,weight_var_name="weight",random_weight=True):
        if number_of_rows==0:
            df = pd.read_csv("./data/higgs-social_network.edgelist/higgs-social_network.edgelist", sep=" ", header=None,names=["side_a", "side_b"])
        else:
            df = pd.read_csv("./data/higgs-social_network.edgelist/higgs-social_network.edgelist", sep=" ",header=None, names=["side_a", "side_b"], nrows=number_of_rows)
        np.random.seed(20)


        if random_weight:
            df[weight_var_name] = np.random.randint(1, 5, df.shape[0])
            g = nx.from_pandas_edgelist(df, source="side_a", target="side_b", edge_attr=weight_var_name,create_using=nx.DiGraph())
        else:
            df[weight_var_name] = 0
            g = nx.from_pandas_edgelist(df, source="side_a", target="side_b", edge_attr=weight_var_name,create_using=nx.DiGraph())
            min_ = np.min([((g.in_degree[side_a] - g.out_degree[side_a]) - (g.in_degree[side_b] - g.out_degree[side_b])) for side_a, side_b, _ in g.edges(data=True)])
            max_ = np.max([((g.in_degree[side_a] - g.out_degree[side_a]) - (g.in_degree[side_b] - g.out_degree[side_b])) for  side_a, side_b, _ in g.edges(data=True)])
            for side_a, side_b, _ in g.edges(data=True):
                # val=abs((g.out_degree[side_a] - g.out_degree[side_b]) / (g.out_degree[side_b]- g.in_degree[side_b]))
                val = ((g.in_degree[side_a] - g.out_degree[side_a]) - (g.in_degree[side_b] - g.out_degree[side_b]))
                nx.set_edge_attributes(g, {(side_a, side_b): {'weight': (val - min_) / (max_ - min_) + 1}})

        # self.social_network_graph = nx.from_pandas_edgelist(df, source="side_a", target="side_b",edge_attr=weight_var_name,create_using=nx.DiGraph())
        # for nd in self.social_network_graph.nodes:
        #     self.social_network_graph.nodes[nd]["node_color"] = self.colors_list[random.randint(0, len(self.colors_list) - 1)]
        self.social_network_graph=g
        print(f"\nsocial_network_graph:\nNumber of Nodes: {self.social_network_graph.number_of_nodes()}\t\tnumber of Edges:{self.social_network_graph.number_of_edges()}\n")
        return self.social_network_graph

    def create_social_working_subgraph(self):
        # lst=[*range(1, 30)]
        # lst=[1,2,3,8,24,25,26,30,45,48,68,69,71,88,92,93,100,110,111,112,113,115,120]
        lst=self.retweet_biggest_subgraph.nodes
        print(lst)
        self.social_working_subgraph = self.social_network_graph.subgraph(lst)  #
        print(f"\tsocial_working_subgraph:\nNumber of Nodes: {self.social_working_subgraph.number_of_nodes()}\t\tnumber of Edges:{self.social_working_subgraph.number_of_edges()}\n")
        return self.social_working_subgraph

    def read_activity_graph(self, number_of_rows=0):

        if number_of_rows==0:
            df = pd.read_csv("./data/higgs-activity_time.txt/higgs-activity_time.txt", sep=" ",header=None, names=["side_a", "side_b","time","action"])
        else:
            df = pd.read_csv("./data/higgs-activity_time.txt/higgs-activity_time.txt", sep=" ", header=None,names=["side_a", "side_b", "time", "action"], nrows=number_of_rows)
        print("\nactivity graph:\n")
        print(Counter(df.loc[:,"action"]))
        rt_df=df.loc[df["action"]=="RT",["side_a","side_b","time"]]
        mt_df=df.loc[df["action"]=="MT",["side_a","side_b","time"]]
        re_df=df.loc[df["action"]=="RE",["side_a","side_b","time"]]


        print("RT:",rt_df.shape)
        print("MT:",mt_df.shape)
        print("RE:",re_df.shape)
        # self.activity_graph = nx.from_pandas_edgelist(df, source="side_a", target="side_b",edge_attr=["time","action"],create_using=nx.DiGraph())
        # print(f"Number of Nodes: {self.activity_graph.number_of_nodes()}\t\tnumber of Edges:{self.activity_graph.number_of_edges()}")
        # for nd in self.activity_graph.nodes:
        #     self.activity_graph.nodes[nd]["node_color"] = self.colors_list[random.randint(0, len(self.colors_list) - 1)]


        self.retweet_activity_graph = nx.from_pandas_edgelist(rt_df, source="side_a", target="side_b",edge_attr=["time"], create_using=nx.DiGraph())
        print(f"Nodes # RT: {self.retweet_activity_graph.number_of_nodes()}\t\tEdge#:{self.retweet_activity_graph.number_of_edges()}")
        for nd in self.retweet_activity_graph.nodes:
            self.retweet_activity_graph.nodes[nd]["node_color"] = self.colors_list[random.randint(0, len(self.colors_list) - 1)]

        self.mention_activity_graph = nx.from_pandas_edgelist(mt_df, source="side_a", target="side_b", edge_attr=["time"], create_using=nx.DiGraph())
        print(f"Nodes # MT: {self.mention_activity_graph.number_of_nodes()}\t\tEdge#:{self.mention_activity_graph.number_of_edges()}")
        for nd in self.mention_activity_graph.nodes:
            self.mention_activity_graph.nodes[nd]["node_color"] = self.colors_list[random.randint(0, len(self.colors_list) - 1)]

        self.reply_activity_graph = nx.from_pandas_edgelist(re_df, source="side_a", target="side_b",edge_attr=["time"], create_using=nx.DiGraph())
        print(f"Nodes # RE: {self.reply_activity_graph.number_of_nodes()}\t\tEdge#:{self.reply_activity_graph.number_of_edges()}")
        for nd in self.reply_activity_graph.nodes:
            self.reply_activity_graph.nodes[nd]["node_color"] = self.colors_list[random.randint(0, len(self.colors_list) - 1)]

    def add_edge_weights(self,G, A):
        for u, v, hdata in A.edges(data=True):  # G.edges_iter
            if G.get_edge_data(u,v)["owner"] != hdata["owner"]:
                nx.set_edge_attributes(G, {(u, v): {'weight': hdata["weight"] + G.get_edge_data(u,v)["weight"] }})

            # attr = dict( (key, value) for key,value in hdata.items())
            # get data from G or use empty dict if no edge in G
            # gdata = G[u].get(v,{})
            # add data from g
            # sum shared items
            # shared = set(gdata) & set(hdata)
            # attr.update(dict((key, attr[key] + gdata[key]) for key in shared))
            # non shared items
            # non_shared = set(gdata) - set(hdata)
            # attr.update(dict((key, gdata[key]) for key in non_shared))
            # yield u,v,attr
        return G

    def read_and_combine_all_graphs(self,weight_var_name='weight',number_of_rows=0):
        graph_reply = self.read_reply_graph(weight_var_name=weight_var_name, number_of_rows=number_of_rows)
        nx.set_edge_attributes(graph_reply,"REP","owner")

        graph_retweet = self.read_retweet_graph(weight_var_name=weight_var_name, number_of_rows=number_of_rows)
        nx.set_edge_attributes(graph_retweet, "RET", "owner")

        graph_mention = self.read_mention_graph(weight_var_name=weight_var_name, number_of_rows=number_of_rows)
        nx.set_edge_attributes(graph_mention, "MEN", "owner")

        t1=time.time()
        print("composing reply and retweet")
        G = nx.compose(graph_reply, graph_retweet)
        print(f"{time.time()-t1}   -- composing ... and mention")
        t1 = time.time()
        G = nx.compose(G, graph_mention)
        print(f"{time.time()-t1}   --  adding edge weights ... reply")

        G2=self.add_edge_weights(G,graph_reply)
        print("adding edge weighhts ... retweet")
        G3=self.add_edge_weights(G2,graph_retweet)

        return G3

    def independent_cascade(self,graph,t, infection_times):
        np.random.seed(10)
        max_weight = max([e[2]['weight'] for e in graph.edges(data=True)])
        icm_current_infectious = [n for n in infection_times if infection_times[n] == t]
        for n in icm_current_infectious:
            for v in graph.neighbors(n):
                if v not in infection_times:
                    if graph.get_edge_data(n, v)['weight'] >= (np.random.random() * max_weight)/self.CASCADE_COEF: # change the multiplier
                        infection_times[v] = t + 1
        return infection_times

    def independent_cascade_with_immunity(self,graph,t, situation_times,opposite_situation_times,situation_overrides):
        np.random.seed(10)
        max_weight = max([e[2]['weight'] for e in graph.edges(data=True)])
        icm_current_situation = [n for n in situation_times if situation_times[n] == t]
        icm_future_opposite_situation = [n for n in opposite_situation_times if opposite_situation_times[n] == t+1]
        icm_old_opposite_situation = [n for n in opposite_situation_times if opposite_situation_times[n] <= t]
        for n in icm_current_situation:
            for v in graph.neighbors(n):
                if situation_overrides: #for ex the immunity can be applied on current infected nodes but not vice-versa
                    if (v not in situation_times) and (v not in icm_old_opposite_situation):
                        if graph.get_edge_data(n, v)['weight'] >= (np.random.random() * max_weight)/self.CASCADE_COEF: # change the multiplier
                            if v in icm_future_opposite_situation:
                                opposite_situation_times.pop(v)
                                print("  ^^^^  a bomb defused ^^^^ !!!!  : ",v)
                            situation_times[v] = t + 1
                else: # situation_times == infected _times
                    if (v not in situation_times) and (v not in opposite_situation_times):
                        if graph.get_edge_data(n, v)['weight'] >= ( np.random.random() * max_weight) / self.CASCADE_COEF:  # change the multiplier
                            situation_times[v] = t + 1
        return situation_times

    def icm_draw_graph(self,graph, pos,weighted_degrees, infection_times, t,total,infected,draw_edges=True):
        current_infectious = [n for n in infection_times if infection_times[n] == t]
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.title(f'Total={total}, Infected={infected}  ({100*infected/total:.2f}%) of nodes are infected at t={t} ', fontsize=22)

        for node in graph.nodes():
            size = 25 * weighted_degrees[node] ** 0.5
            if node in current_infectious:
                ns = nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=size, node_color='Red')  # yellow #feba02
            elif infection_times.get(node,9999999) < t:  # it was infected before node't is smaller than current t   999999 for if key is not found and for these cases next else will be triggered
                ns = nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=size,
                                            node_color='Gray')  # white '#f2f6fa'
            else:
                ns = nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=size,
                                            node_color='Cyan')  # not infected yet  #Blue #009fe3
            ns.set_edgecolor('#f2f6fa')
        nx.draw_networkx_labels(graph, pos,{n: "N:"+str(n) for n in graph.nodes()},font_size=6) #if weighted_degrees[n] > 100
        if draw_edges:
            for e in graph.edges(data=True):
                if e[2]['weight'] > 0: # consider min edge weight minimum_node_w_degree
                    nx.draw_networkx_edges(graph, pos, [e], width=e[2]['weight'] / 10, edge_color='#707070')
        plt.show()

    def icm_draw_graph_with_immunity(self,graph, pos,weighted_degrees, infection_times, immunity_times,t,total,infected,immuned,draw_edges=True):
        current_infectious = [n for n in infection_times if infection_times[n] == t]
        current_immuned = [n for n in immunity_times if immunity_times[n] == t]
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.title(f'Total={total}, Infected={infected} ({100*infected/total:.2f}%)  ,Immune={immuned}({100*immuned/total:.2f}%)  at t={t} ', fontsize=20)

        for node in graph.nodes():
            size = 100 * weighted_degrees[node] ** 0.5
            if node in current_infectious:
                ns = nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=size, node_color='Red')  # yellow #feba02
            elif infection_times.get(node,9999999) < t:  # it was infected before node't is smaller than current t   999999 for if key is not found and for these cases next else will be triggered
                ns = nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=size, node_color='Gray')
            else:
                if node in current_immuned:
                    ns = nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=size, node_color='Green')
                elif immunity_times.get(node,9999999) < t:  # it was immuned before
                    ns = nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=size,node_color='#8efac2')
                else:
                    ns = nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=size,  node_color='#7de5ff')  # not infected yet  #Blue #009fe3
            ns.set_edgecolor('#f2f6fa')
        nx.draw_networkx_labels(graph, pos,{n: str(n) for n in graph.nodes()},font_size=6) #if weighted_degrees[n] > 100
        if draw_edges:
            for e in graph.edges(data=True):
                if e[2]['weight'] > 0: # consider min edge weight minimum_node_w_degree
                    nx.draw_networkx_edges(graph, pos, [e], width=e[2]['weight'] / 10, edge_color='#707070')
        plt.show()

    def run_icm_with_immunity(self, graph,infection_list,immunity_list,weight_var_name='weight',minimum_node_w_degree=1,draw_graph=False,draw_edge=False,spread_coef=1,delay=0):
        self.CASCADE_COEF = spread_coef
        icm_weighted_degrees = dict(nx.degree(graph, weight=weight_var_name))
        sub_g = graph.subgraph([n for n in icm_weighted_degrees if icm_weighted_degrees[n] >= minimum_node_w_degree])
        if draw_graph:
            pos = nx.spring_layout(sub_g, weight='weight', iterations=20, k=4)
        immunity_times={}
        for j in immunity_list:
            immunity_times[j]=delay

        infection_times={}
        for i in infection_list:
            infection_times[i]=0
        new_infected=True
        t=0
        total_number_infected=0
        total_number_immuned=0
        percent_result_infected=[]
        percent_result_immuned=[]
        immune_list=[]
        #for t in range(5):
        while new_infected:

            res=list(l for l in infection_times.keys() if infection_times[l]==t)
            print("\t\tt:",t,"  new ones:",len(res))
            total_number_infected+=len(res)
            if t>= delay:
                immune_list=list(l for l in immunity_times.keys() if immunity_times[l]==t)

            total_number_immuned+=len(immune_list)

            if draw_graph:
                self.icm_draw_graph_with_immunity (graph=sub_g,pos= pos,weighted_degrees= icm_weighted_degrees,infection_times= infection_times,immunity_times=immunity_times,
                                                   t=t,total=sub_g.number_of_nodes(),infected=total_number_infected,immuned=total_number_immuned,draw_edges=draw_edge)
            # print(f" T=={t}:")

            percent_result_infected.append(total_number_infected/sub_g.number_of_nodes())
            percent_result_immuned.append(total_number_immuned /sub_g.number_of_nodes())

            cnt_before=len(infection_times)
            infection_times = self.independent_cascade_with_immunity(sub_g, t,situation_times= infection_times,opposite_situation_times=immunity_times,situation_overrides=False)
            if t >= delay:
                immunity_times=self.independent_cascade_with_immunity(sub_g,t,situation_times= immunity_times,opposite_situation_times=infection_times,situation_overrides=True)
            if len(infection_times) == cnt_before:
                # print(" NO MORE PROGRESS!!")
                new_infected=False
            t+=1
        return percent_result_infected,percent_result_immuned

    def run_icm(self, graph,infection_list,weight_var_name='weight',minimum_node_w_degree=1,draw_graph=False,spread_coef=1):
        self.CASCADE_COEF = spread_coef
        icm_weighted_degrees = dict(nx.degree(graph, weight=weight_var_name))
        sub_g = graph.subgraph([n for n in icm_weighted_degrees if icm_weighted_degrees[n] >= minimum_node_w_degree])
        if draw_graph:
            pos = nx.spring_layout(sub_g, weight='weight', iterations=20, k=4)
        infection_times={}
        for i in infection_list:
            infection_times[i]=0
        new_infected=True
        t=0
        total_number_infected=0
        percent_result=[]

        #for t in range(5):
        while new_infected:
            res=list(l for l in infection_times.keys() if infection_times[l]==t)
            total_number_infected+=len(res)
            if draw_graph:
                self.icm_draw_graph(graph=sub_g,pos= pos,weighted_degrees= icm_weighted_degrees,infection_times= infection_times,t=t,total=sub_g.number_of_nodes(),infected=total_number_infected,draw_edges=False)
            # print(f" T=={t}:")
            # print(f"\t\t Count in this round: {len(res)},  total percentage ({total_number_infected}/{sub_g.number_of_nodes()}) : {100*total_number_infected/sub_g.number_of_nodes():.2f}% ")
            percent_result.append(total_number_infected/sub_g.number_of_nodes())
            # print(f"\t\t Count in this round: {len(res)},  total percentage ({total_number_infected}/{sub_g.number_of_nodes()}) : {100*total_number_infected/sub_g.number_of_nodes():.2f}%  --> {res}")


            cnt_before=len(infection_times)
            infection_times = self.independent_cascade(sub_g, t, infection_times)
            if len(infection_times) == cnt_before:
                # print(" NO MORE PROGRESS!!")
                new_infected=False
            t+=1
        return percent_result

    def run_icm_ndlib(self,graph,infection_list,max_number_iteration,weight_coef,weight_var_name='weight',minimum_node_w_degree=1,spread_coef=1):
        self.CASCADE_COEF = spread_coef
        icm_weighted_degrees = dict(nx.degree(graph, weight=weight_var_name))
        sub_g = graph.subgraph([n for n in icm_weighted_degrees if icm_weighted_degrees[n] >= minimum_node_w_degree])
        percent_result = []

        model = ep.IndependentCascadesModel(sub_g)
        config = mc.Configuration()

        config.add_model_initial_configuration("Infected", infection_list)

        for e in sub_g.edges(data=True):
            threshold= e[2]['weight'] * spread_coef *weight_coef
            config.add_edge_configuration("threshold", (e[0],e[1]), threshold)
        # for i in g.nodes():
        #     config.add_edge_configuration("threshold", i, threshold)
        model.set_initial_status(config)

        # iterations = model.iteration_bunch(5)
        print("\nldlib ICM: ")
        percent_result.append(len(infection_list)/sub_g.number_of_nodes())
        for i in range(max_number_iteration):
            itr = model.iteration()
            # print(itr['node_count'])
            print(f"{(itr['node_count'][1] + itr['node_count'][2]) / (itr['node_count'][0] + itr['node_count'][1] + itr['node_count'][2]):.3f}")
            percent_result.append((itr['node_count'][1] + itr['node_count'][2]) / (itr['node_count'][0] + itr['node_count'][1] + itr['node_count'][2]))
        return percent_result

    def run_ltm_ndlib(self,graph,infection_list,max_number_iteration,weight_coef,weight_var_name='weight',minimum_node_w_degree=1,spread_coef=1):
        self.CASCADE_COEF = spread_coef
        icm_weighted_degrees = dict(nx.degree(graph, weight=weight_var_name))
        sub_g = graph.subgraph([n for n in icm_weighted_degrees if icm_weighted_degrees[n] >= minimum_node_w_degree])
        percent_result = []

        # model = ep.IndependentCascadesModel(sub_g)
        model = ep.ThresholdModel(sub_g)
        config = mc.Configuration()

        config.add_model_initial_configuration("Infected", infection_list)

        for i in sub_g.nodes():
            thr=spread_coef * weight_coef * (0.0001 + sub_g.in_degree[i]/(sub_g.out_degree[i]+0.0001))
            # print(i," --> ",thr)
            config.add_node_configuration("threshold", i, thr)
        model.set_initial_status(config)

        # iterations = model.iteration_bunch(5)
        print("\nldlib LTM: ")
        percent_result.append(len(infection_list)/sub_g.number_of_nodes())
        for i in range(max_number_iteration):
            itr = model.iteration()
            print(f"{(itr['node_count'][1]) / (itr['node_count'][0] + itr['node_count'][1]):.3f}")
            percent_result.append((itr['node_count'][1] ) / (itr['node_count'][0] + itr['node_count'][1]))
        return percent_result

def test_init_graphs():
    higgs=Hissg_Graphs()
    reply_graph=higgs.read_reply_graph(weight_var_name='weight', number_of_rows=500)
    bcc = sorted(nx.strongly_connected_components(reply_graph), key=len, reverse=True)
    print(f"biggest reply sub graph -->size:{len(bcc[0])} ")
    reply_biggest_subgraph = reply_graph.subgraph(bcc[0])  #
    # higgs.read_mention_graph(100)
    # higgs.read_retweet_graph()
    # higgs.read_social_network_graph()
    #Number of Nodes: 456626		number of Edges:14855842
    # higgs.create_social_working_subgraph()
    # higgs.read_activity_graph()
    # higgs.draw_graph(higgs.social_network_graph)
    # higgs.draw_graph(higgs.social_working_subgraph)
    higgs.draw_graph(reply_biggest_subgraph)

def find_random_candidates(number_of_candidates,graph):
    if number_of_candidates < graph.number_of_nodes():
        return random.sample(graph.nodes, number_of_candidates)
    else:
        return random.sample(graph.nodes, 1)

def  find_candidates(number_of_candidates,graph,use_closeness=True,use_degree=True,use_eigen=True,use_between=True):

    if use_closeness:
        closeness = nx.closeness_centrality(graph)
        used_dic=closeness
    if use_between:
        betweenness = nx.betweenness_centrality(graph)
        used_dic = betweenness
    if use_eigen:
        eigen_vector_centrality=nx.eigenvector_centrality(graph)
        used_dic = eigen_vector_centrality
    if use_degree:
        degree_centrality=nx.degree_centrality(graph)
        used_dic = degree_centrality
    combined={}
    for node in used_dic.keys():
        combined[node]=0
        if use_closeness:
            combined[node] +=closeness[node]
        if use_degree:
            combined[node] +=degree_centrality[node]
        if use_eigen:
            combined[node] +=eigen_vector_centrality[node]
        if use_between:
            combined[node] +=betweenness[node]
    sorted_combined={k: v for k, v in sorted(combined.items(), key=lambda item: item[1],reverse=True)}
    st6 = time.time()
    # for i in range(number_of_candidates):
    #     result.append(sorted_combined.items[i])

    # for item in sorted_combined:
    #      print("node: " + str(item) + " combined centrality: " + str(sorted_combined[item]))
    return [*sorted_combined][:number_of_candidates]

def test_icm_on_social_graph(number_of_rows=20000,number_of_infected_candidates=5,test_over_connected_sub=1,draw_graph=False, draw_edge=False,spread_coef=1):

    higgs = Hissg_Graphs()
    # graph = higgs.read_reply_graph(weight_var_name='weight', number_of_rows=2000)
    graph = higgs.read_social_network_graph (weight_var_name='weight', number_of_rows=number_of_rows,random_weight=True)
    bcc = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    print("top 10 connected components:")
    for i in bcc[:test_over_connected_sub]:
        print("C C S size:",len(i))
    # biggest_subgraph = graph.subgraph(bcc[0])

    #infection_list = [13808, 92274,30478,6241,48963] #reply list
    # infection_list = [995,626,28,26,98] #network list
    for sub_graph in bcc[:test_over_connected_sub]:
        print(f"connected sub graph size:{len(sub_graph)} ")
        print("finding candidates ...")
        current_subgraph = graph.subgraph(sub_graph)
        infection_list=find_candidates(number_of_candidates=number_of_infected_candidates,graph=current_subgraph,use_between=False)
        print("Candidates:",infection_list)
        print("Running icm")
        res_top_candidates_percent=higgs.run_icm(graph=current_subgraph,weight_var_name='weight',
                                                 infection_list=infection_list,minimum_node_w_degree=1,
                                                 draw_graph=draw_graph,spread_coef=spread_coef)
        print("done")
        res_random_percent = []
        print("Calc for random init nodes...")
        for i in range(20):
            print("i:",i)
            random_infection_list = find_random_candidates(number_of_candidates=number_of_infected_candidates, graph=current_subgraph)
            res_random_percent.append(higgs.run_icm(graph=current_subgraph, weight_var_name='weight',
                                                    infection_list=random_infection_list, minimum_node_w_degree=1,
                                                    draw_graph=False,spread_coef=spread_coef))

        max_len = len(max(res_random_percent, key=len))
        random_res = np.array([np.array(i + (max_len - len(i)) * [i[-1]]) for i in res_random_percent])
        # print(random_res)
        random_res = np.average(random_res, axis=0)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 1)
        plt.plot(random_res, label='random')
        for i, j in enumerate(random_res):
            ax.annotate(f"{j:.2f}", xy=(i, j))

        plt.plot(res_top_candidates_percent, label='Greedy')
        for i, j in enumerate(res_top_candidates_percent):
            ax.annotate(f"{j:.3f}", xy=(i, j))
        plt.legend()
        plt.show()

def test_icm_on_action_graphs(number_of_rows=100000,spread_coef=0.1,number_of_candidates=5,draw_graph=False, draw_edge=False):
    higgs = Hissg_Graphs()
    graph = higgs.read_and_combine_all_graphs(weight_var_name='weight', number_of_rows=number_of_rows)
    print(f"\ncombined graph:\nNumber of Nodes: {graph.number_of_nodes()}\t\tnumber of Edges:{graph.number_of_edges()}\n")

    print( "finding biggest subgraph!!")

    bcc = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)

    print(f"biggest  sub graph -->size:{len(bcc[0])} ")
    comb_biggest_subgraph = graph.subgraph(bcc[0])

    t1 = time.time()
    print("Finding infection candidates!!")
    # infection_list=find_candidates(number_of_candidates=5,graph= comb_biggest_subgraph,use_between=False)

    top_infection_list = find_candidates(number_of_candidates=number_of_candidates, graph=comb_biggest_subgraph, use_between=False)
    res_top_candidates_percent = higgs.run_icm(graph=comb_biggest_subgraph, weight_var_name='weight',
                                               infection_list=top_infection_list, minimum_node_w_degree=1,
                                               draw_graph=draw_graph,spread_coef=spread_coef)

    random_infection_list=[]
    # print(f"elapsed time:{time.time()-t1:.5f}  Infected lst: {random_infection_list}")
    res_random_percent=[]
    for i in range(20):
        random_infection_list = find_random_candidates(number_of_candidates=number_of_candidates, graph=comb_biggest_subgraph)
        res_random_percent.append(higgs.run_icm(graph=comb_biggest_subgraph, weight_var_name='weight',
                                           infection_list=random_infection_list, minimum_node_w_degree=1,
                                           draw_graph=False,spread_coef=spread_coef))

    max_len = len(max(res_random_percent, key=len))
    random_res = np.array([np.array(i + (max_len - len(i)) * [i[-1]]) for i in res_random_percent])
    # print(random_res)
    random_res=np.average(random_res, axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1)
    plt.plot(random_res, label='random')
    for i, j in enumerate(random_res):
        ax.annotate(f"{j:.2f}", xy=(i, j))

    plt.plot(res_top_candidates_percent , label='Greedy Choice')
    for i, j in enumerate(res_top_candidates_percent):
        ax.annotate(f"{j:.3f}", xy=(i, j))
    plt.legend()
    plt.show()

def test_icm_on_action_graphs_with_immunity(number_of_infected_candidates,number_of_immuned_candidates,number_of_rows=100000,spread_coef=0.1,delay=0,draw_graph=False, draw_edge=False):
    higgs = Hissg_Graphs()
    graph = higgs.read_and_combine_all_graphs(weight_var_name='weight', number_of_rows=number_of_rows)
    print(f"\ncombined graph:\nNumber of Nodes: {graph.number_of_nodes()}\t\tnumber of Edges:{graph.number_of_edges()}\n")

    print("finding biggest subgraph!!")

    bcc = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)

    print(f"biggest  sub graph -->size:{len(bcc[0])} ")
    comb_biggest_subgraph = graph.subgraph(bcc[0])

    t1 = time.time()
    print("Finding infection candidates!!")
    # infection_list=find_candidates(number_of_candidates=5,graph= comb_biggest_subgraph,use_between=False)

    top_candidates_list = find_candidates(number_of_candidates=number_of_infected_candidates+number_of_immuned_candidates, graph=comb_biggest_subgraph, use_between=False)
    res_top_candidates_percent,res_top_immuned_percent = higgs.run_icm_with_immunity(graph=comb_biggest_subgraph, weight_var_name='weight',
                                               infection_list=top_candidates_list[:number_of_infected_candidates],immunity_list=top_candidates_list[number_of_infected_candidates:number_of_infected_candidates+number_of_immuned_candidates]
                                                             ,minimum_node_w_degree=1, draw_graph=draw_graph,spread_coef=spread_coef,delay=delay, draw_edge=draw_edge)

    random_infection_list=[]

    # res_random_percent=[]
    # for i in range(20):
    #     random_infection_list = find_random_candidates(number_of_candidates=number_of_candidates, graph=comb_biggest_subgraph)
    #     res_random_percent.append(higgs.run_icm(graph=comb_biggest_subgraph, weight_var_name='weight',
    #                                        infection_list=random_infection_list, minimum_node_w_degree=1,
    #                                        draw_graph=False,spread_coef=spread_coef))

    # max_len = len(max(res_random_percent, key=len))
    # random_res = np.array([np.array(i + (max_len - len(i)) * [i[-1]]) for i in res_random_percent])
    # print(random_res)
    # random_res=np.average(random_res, axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1)

    plt.plot(res_top_immuned_percent, label='immuned')
    for i, j in enumerate(res_top_immuned_percent):
         ax.annotate(f"{j:.2f}", xy=(i, j))

    plt.plot(res_top_candidates_percent , label='infected')
    for i, j in enumerate(res_top_candidates_percent):
        ax.annotate(f"{j:.3f}", xy=(i, j))
    plt.legend()
    plt.show()

def test_ndlib_icm_on_action_graphs(number_of_rows=100000,number_of_candidates=5,max_number_iteration=30):
    weight_coef=0.25
    higgs = Hissg_Graphs()
    graph = higgs.read_and_combine_all_graphs(weight_var_name='weight', number_of_rows=number_of_rows)
    print(f"\ncombined graph:\nNumber of Nodes: {graph.number_of_nodes()}\t\tnumber of Edges:{graph.number_of_edges()}\n")
    print("finding biggest subgraph!!")
    bcc = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    print(f"biggest  sub graph -->size:{len(bcc[0])} ")
    comb_biggest_subgraph = graph.subgraph(bcc[0])
    t1 = time.time()
    print("Finding infection candidates!!")
    # infection_list=find_candidates(number_of_candidates=5,graph= comb_biggest_subgraph,use_between=False)
    top_infection_list = find_candidates(number_of_candidates=number_of_candidates, graph=comb_biggest_subgraph, use_between=False)
    res_top_candidates_percent = higgs.run_icm_ndlib(graph=comb_biggest_subgraph, weight_var_name='weight',
                                               infection_list=top_infection_list, minimum_node_w_degree=1,max_number_iteration=max_number_iteration,weight_coef=weight_coef)

    print("Random init")
    res_random_percent=[]
    for i in range(10):
        random_infection_list = find_random_candidates(number_of_candidates=number_of_candidates, graph=comb_biggest_subgraph)
        res_random_percent.append(higgs.run_icm_ndlib(graph=comb_biggest_subgraph, weight_var_name='weight',infection_list=random_infection_list, minimum_node_w_degree=1,
                                                      max_number_iteration=max_number_iteration,weight_coef=weight_coef))

    max_len = len(max(res_random_percent, key=len))
    random_res = np.array([np.array(i + (max_len - len(i)) * [i[-1]]) for i in res_random_percent])
    # print(random_res)
    random_res=np.average(random_res, axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1)
    plt.plot(random_res, label='random')
    for i, j in enumerate(random_res):
        ax.annotate(f"{j:.2f}", xy=(i, j))

    plt.plot(res_top_candidates_percent , label='Greedy Choice')
    for i, j in enumerate(res_top_candidates_percent):
        ax.annotate(f"{j:.3f}", xy=(i, j))
    plt.legend()
    plt.show()

def test_ndlib_ltm_on_action_graphs(number_of_rows=100000,number_of_candidates=5,max_number_iteration=15):
    weight_coef=0.15
    higgs = Hissg_Graphs()
    graph = higgs.read_and_combine_all_graphs(weight_var_name='weight', number_of_rows=number_of_rows)
    print(f"\ncombined graph:\nNumber of Nodes: {graph.number_of_nodes()}\t\tnumber of Edges:{graph.number_of_edges()}\n")
    print("finding biggest subgraph!!")
    bcc = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    print(f"biggest  sub graph -->size:{len(bcc[0])} ")
    comb_biggest_subgraph = graph.subgraph(bcc[0])
    t1 = time.time()
    print("Finding infection candidates!!")
    # infection_list=find_candidates(number_of_candidates=5,graph= comb_biggest_subgraph,use_between=False)
    top_infection_list = find_candidates(number_of_candidates=number_of_candidates, graph=comb_biggest_subgraph, use_between=False)
    res_top_candidates_percent = higgs.run_ltm_ndlib(graph=comb_biggest_subgraph, weight_var_name='weight',
                                               infection_list=top_infection_list, minimum_node_w_degree=1,max_number_iteration=max_number_iteration,weight_coef=weight_coef)

    print("Random init")
    res_random_percent=[]
    for i in range(10):
        random_infection_list = find_random_candidates(number_of_candidates=number_of_candidates, graph=comb_biggest_subgraph)
        res_random_percent.append(higgs.run_ltm_ndlib(graph=comb_biggest_subgraph, weight_var_name='weight',infection_list=random_infection_list, minimum_node_w_degree=1,
                                                      max_number_iteration=max_number_iteration,weight_coef=weight_coef))

    max_len = len(max(res_random_percent, key=len))
    random_res = np.array([np.array(i + (max_len - len(i)) * [i[-1]]) for i in res_random_percent])
    # print(random_res)
    random_res=np.average(random_res, axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1)
    plt.plot(random_res, label='random')
    for i, j in enumerate(random_res):
        ax.annotate(f"{j:.2f}", xy=(i, j))

    plt.plot(res_top_candidates_percent , label='Greedy Choice')
    for i, j in enumerate(res_top_candidates_percent):
        ax.annotate(f"{j:.3f}", xy=(i, j))
    plt.legend()
    plt.show()

def compare_centralities(spread_coef=1,init_infected_count=5, number_of_rows=100000):
    higgs = Hissg_Graphs()
    # graph = higgs.read_reply_graph(weight_var_name='weight', number_of_rows=2000)
    graph = higgs.read_social_network_graph(weight_var_name='weight', number_of_rows=number_of_rows,random_weight=True)
    bcc = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    print(f"biggest graph sub graph -->size:{len(bcc[0])} ")
    biggest_subgraph = graph.subgraph(bcc[0])

    infection_list_closeness = find_candidates(init_infected_count, biggest_subgraph,use_closeness=True,use_degree=False,use_eigen=False,use_between=False)
    infection_list_degree = find_candidates(init_infected_count, biggest_subgraph,use_closeness=False,use_degree=True,use_eigen=False,use_between=False)
    infection_list_eigen = find_candidates(init_infected_count, biggest_subgraph,use_closeness=False,use_degree=False,use_eigen=True,use_between=False)
    infection_list_between = find_candidates(init_infected_count, biggest_subgraph,use_closeness=False,use_degree=False,use_eigen=False,use_between=True)

    lst_closeness=higgs.run_icm(graph=biggest_subgraph, weight_var_name='weight', infection_list=infection_list_closeness,minimum_node_w_degree=1, draw_graph=False,spread_coef=spread_coef)
    lst_degree=higgs.run_icm(graph=biggest_subgraph, weight_var_name='weight', infection_list=infection_list_degree,minimum_node_w_degree=1, draw_graph=False,spread_coef=spread_coef)
    lst_eigen=higgs.run_icm(graph=biggest_subgraph, weight_var_name='weight', infection_list=infection_list_eigen,minimum_node_w_degree=1, draw_graph=False,spread_coef=spread_coef)
    lst_between=higgs.run_icm(graph=biggest_subgraph, weight_var_name='weight', infection_list=infection_list_between,minimum_node_w_degree=1, draw_graph=False,spread_coef=spread_coef)

    max_length = max(max(len(lst_closeness), len(lst_degree)), len(lst_eigen),len(lst_between))
    lst_closeness += [lst_closeness[-1]] * (max_length - len(lst_closeness))
    lst_degree += [lst_degree[-1]] * (max_length - len(lst_degree))
    lst_eigen += [lst_eigen[-1]] * (max_length - len(lst_eigen))
    lst_between += [lst_between[-1]] * (max_length - len(lst_between))

    df=pd.DataFrame()
    df["closeness centrality"]=lst_closeness
    df["degree centrality"]=lst_degree
    df["eigen_vector centrality"]=lst_eigen
    df["betweeness"]=lst_between
    df.plot()
    plt.legend(fontsize=12)
    plt.ylabel('Infection Rate', fontsize=16)
    plt.xlabel('rounds', fontsize=16)
    plt.show()

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1", "ok")
if __name__ == "__main__":
    #test_icm_on_social_graph(number_of_rows=5000000,random_weight=False)  #
    # test_icm_on_action_graphs(number_of_rows=100000,spread_coef=30,number_of_candidates=5)
    # compare_centralities(spread_coef=0.1,init_infected_count=20,number_of_rows=500000)
    # test_ndlib_icm_on_action_graphs(number_of_rows=500000,number_of_candidates=5,max_number_iteration=20)
    # test_ndlib_ltm_on_action_graphs(number_of_rows=500000,number_of_candidates=5,max_number_iteration=20)
    #test_icm_on_action_graphs_with_immunity(number_of_rows=20000,spread_coef=20,number_of_infected_candidates=5,number_of_immuned_candidates=5,delay=0)
    parser = argparse.ArgumentParser()
    parser.add_argument("-fid", "--function_id", required=True)
    parser.add_argument("-r", "--nrows", required=True)
    parser.add_argument("-c", "--ncand", required=True)

    parser.add_argument("-sc", "--spcoef", required=False)
    parser.add_argument("-d", "--delay", required=False)
    parser.add_argument("-ic", "--nimmune", required=False)

    parser.add_argument("-dw", "--draw", required=False)
    parser.add_argument("-de", "--drawedge", required=False)



    args = parser.parse_args()
    print(f' function ID is  {args.function_id }')
    print(f' number of rows is  {args.nrows}')
    print(f' number of initial active nodes is  {args.ncand}')
    print(f' draw graph is  {args.draw}')

    # python main.py -fid 1 -r 10000 -c 5 -sc 0.2 -d 0 -ic 5 -dw False -de False
    if int(args.function_id) ==1:
        print("** test_icm_on_action_graphs_with_immunity")
        print(f' spread coefficient: {args.spcoef}')
        print(f' delay: {args.delay}')
        print(f' N. immune: {args.nimmune}')
        spread_coef_coef=50
        test_icm_on_action_graphs_with_immunity(number_of_rows=int(args.nrows), spread_coef=spread_coef_coef*float(args.spcoef),
                                                number_of_infected_candidates=int(args.ncand),
                                                number_of_immuned_candidates=int(args.nimmune), delay=int(args.delay),
                                                draw_graph=str2bool(args.draw), draw_edge=str2bool(args.drawedge))
    # python main.py -fid 2 -r 10000 -c 5 -sc 0.3 -dw True
    if int(args.function_id) ==2:
        print("** test_icm_on_social_graph")
        spread_coef_coef = 1
        test_icm_on_social_graph(number_of_rows=int(args.nrows),
                                 number_of_infected_candidates=int(args.ncand),
                                 draw_graph=str2bool(args.draw),spread_coef=spread_coef_coef*float(args.spcoef))

    #python main.py -fid 3 -r 500000 -c 5 -sc 0.5 -dw False
    if int(args.function_id) ==3:
        print("** test_icm_on_action_graphs")
        spread_coef_coef = 30
        test_icm_on_action_graphs(number_of_rows=int(args.nrows),
                                 number_of_candidates=int(args.ncand),
                                 draw_graph=str2bool(args.draw),spread_coef=spread_coef_coef*float(args.spcoef))
