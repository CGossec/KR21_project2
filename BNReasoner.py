import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net
            print(net)

    def determine_reach(self, reachable_nodes, seen):
        new_reachable_nodes = list(set(reachable_nodes) - set(seen))
        for node in new_reachable_nodes:
            for child in self.bn.get_children(node):
                if child not in reachable_nodes:
                    reachable_nodes.append(child)
            seen.append(node)
        if list(set(reachable_nodes) - set(seen)) != []:
            #Recursive part: Find the children of the children of the current set
            reachable_nodes = self.determine_reach(reachable_nodes, seen)
        return reachable_nodes

    def d_separation(self , X, Y, Z):
        #Prune the graph
        all_variables = (self.bn.get_all_variables())

        for variable in all_variables:
            if variable not in (X + Y +Z) and self.bn.get_children(variable) == []:
                self.bn.del_var(variable)
        for variable in Z:
            children = self.bn.get_children(variable)
            for child in children:
                self.bn.del_edge([variable, child])
        #Find the reachable nodes from X
        X_reachables = []
        for variable in X:
            reachable_nodes = []
            seen = []
            reachable_nodes.extend(self.bn.get_children(variable))
            seen.extend(variable)
            reachable_nodes = self.determine_reach(reachable_nodes, seen)
            X_reachables.append(reachable_nodes)
        for x_reach in X_reachables:
            for y in Y:
                if y in x_reach:
                    print(X, 'and', Y, 'are not d-separated by', Z)
                    return False
        print(X, 'and', Y, 'are d-separated by', Z)
        return True

    def pruning(self, query: List[str], evidence: Dict[str, bool]) -> BayesNet:
        """
        Given a graph G and disjoint set of nodes in the query and an evidence, return a pruned graph G' such that:
            * There are no leaf nodes that do not belong to Query + evidence
            * All edges outgoing from nodes in evidence are deleted

        :param query: List of nodes
        :param evidence: Dictionary of variables with their assigned truth values
        :return: Copy of self.net, with appropriate nodes and edge removed, and CPTs updated
        """
        pruned_graph = copy.copy(self.bn)
        modified = True
        while modified:
            modified = False
            for var in pruned_graph.get_all_variables():
                if pruned_graph.get_children(var) == [] and var not in query and var not in evidence:
                    pruned_graph.del_var(var)
                    modified = True
            CPTs = pruned_graph.get_all_cpts()
            for given in evidence:
                for variable in pruned_graph.get_all_cpts():
                    cpt = pruned_graph.get_all_cpts()[variable]
                    if given not in cpt.columns:
                        continue
                    indices_to_drop = cpt[cpt[given] == (not evidence[given])].index
                    new_cpt = cpt.drop(indices_to_drop)
                    new_cpt = new_cpt.reset_index(drop=True)
                    pruned_graph.update_cpt(variable, new_cpt)
                for child in pruned_graph.get_children(given):
                    pruned_graph.del_edge((given, child))
                    modified = True
        return pruned_graph

    def min_degree(self) -> List[Tuple[str, int]]:
        """
        Ordering nodes on the basis of the smallest degree.

        :return: a list the variables and their degree ordered according to the heuristic
        """
        interaction_graph = self.bn.get_interaction_graph()
        degrees = sorted(list(interaction_graph.degree()), key=lambda x: x[1])
        return degrees

    def min_fill(self) -> List[Tuple[str, int]]:
        """
        Ordering nodes on the basis where elimination leads to smallest number of edges.

        :return: a list the variables and their degree ordered according to the heuristic
        """
        interaction_graph = self.bn.get_interaction_graph()
        added_edges = []
        for node in interaction_graph:
            count = 0
            neighbors = interaction_graph.neighbors(node)
            for neighbor in neighbors:
                for elm in neighbors:
                    if neighbor is elm:
                        continue
                    if elm not in interaction_graph.neighbors(neighbor):
                        count += 1
            added_edges.append((node, count))
        return sorted(added_edges, key=lambda x: x[1])


    def marginal_distributions(self, query: List[str], evidence: Optional[Dict[str, bool]], ordering: List[Tuple[str, int]]) -> Dict[str, pd.DataFrame]:
        """
        Computes the marginal distribution of the given query w.r.t. evidence.

        :param query: List of nodes
        :param evidence: if None, this computes the prior marginal. Otherwise, this is a dictionary of values with
            respective truth assignments.
        :return: a dictionary matching the variables of the query with their respective CPTs as pd.DataFrame
        """
        for elm in query:
            for tup in ordering:
                if elm in tup:
                    ordering.remove(tup)
        if evidence is None:
            evidence = {}
        pruned_graph = self.pruning(bnr.bn.get_all_variables(), evidence=evidence)
        print(f"Ordering: {ordering}")
        S = self.bn.get_all_cpts()
        factors = {}
        print([S[cpt] for cpt in [_ for _ in S]], "\n=======================================\n")
        for (node, _) in ordering:
            factor = None
            for variable in S:
                cpt = S[variable]
                if node in cpt.columns:
                    if factor is None:
                        factor = cpt
                        continue
                    S[variable] = multiply_factors(factor, cpt)
            factors[node] = factor
            for variable in S:
                if S[variable].empty:
                    continue
                if S[variable].columns[-2] == node:
                    S[variable] = pd.DataFrame()
                    continue
                if node in S[variable].columns:
                    S[variable] = sum_out_variable(S[variable], node)
        result = {var: S[var] for var in query}
        for var in result:
            if len(result[var]) > 2:
                vars_to_remove = list(result[var].columns[:-2])
                for other in vars_to_remove:
                    result[var] = sum_out_variable(multiply_factors(result[var], result[other]), other)
        return {var: self.normalize_with_evidence(result[var], evidence, factors) for var in result}

    def normalize_with_evidence(self, cpt: pd.DataFrame, evidence: Dict[str, bool], factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        For every value in the result, we normalize according to the given evidence to reflect proper probabilities.

        :param cpt: the dataframe we normalize. Modification is not done inplace.
        :param evidence: A dictionary of the variables and their assigned truth values
        :param factors: the factors we computed. I'm not sure whether this could be changed or not.
        :return: A new CPT, normalized according to the evidence.
        """
        res = cpt.copy()
        for var in evidence:
            evidence_cpt = factors[var]
            proba_of_evidence = float(evidence_cpt[evidence_cpt[var] == evidence[var]]["p"])
            res["p"] = res["p"] / proba_of_evidence
        return res

    def reduce_cpts(self, cpts, evidence):
        for cpt in cpts:
            for ev in evidence:
                if ev in cpts[cpt]:
                    for i in range(0, len(cpts[cpt].index)):
                        if cpts[cpt].loc[i,ev] == (not evidence[ev]):
                            cpts[cpt] = cpts[cpt].drop(labels=i, axis=0)
        return cpts

    def map(self, variables, evidence):
        res, cpts = {} , self.bn.get_all_cpts()
        cpts = self.bn.get_all_cpts()
        cpts = self.reduce_cpts(cpts, evidence)
        pos_marg = self.marginal_distributions(variables, evidence, self.min_fill())
        bnr.bn.draw_structure()
        print("\n\n\n\n", pos_marg)
        for instance in pos_marg:
            idx = pos_marg[instance]['p'].idxmax()
            res[pos_marg[instance].columns[-2]] = pos_marg[instance].loc[idx, pos_marg[instance].columns[-2]]
        return res

    def mpe(self, query: List[str], evidence):
        prunned_net = self.pruning(query, evidence)


def multiply_factors(cpt1: pd.DataFrame, cpt2: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplies two CPTs, taking the common variables and multiplying the values where it can.
    If no variables are in common, the algorithm won't do anything.

    :param cpt1: One of the CPTs to multiply
    :param cpt2: The other of the CPTs to multiply
    :return: a CPT containing the multiplication result of both CPTs
    """
    res = cpt2 if len(cpt2.columns) > len(cpt1.columns) else cpt1
    other = cpt1 if len(cpt2.columns) > len(cpt1.columns) else cpt2
    for var in other.columns[:-1]:
        if var not in res.columns:
            continue
        for _, row in other.iterrows():
            truth_value = row[var]
            res.loc[res[var] == truth_value, "p"] *= row["p"]
    return res

def sum_out_variable(cpt: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Given a CPT, we want to remove a variable from it by summing all values opposed with said variable.
    If the table looks like:
        A      B      p
    0  False False  0.015
    1  False True   0.485
    2  True  False  0.495
    3  True  True   0.005
    Then we sum out lines 0 and 2, and lines 1 and 3 to get
        B      p
    0  False  0.51
    1  True   0.49
    We summed every line where B was False, 0.015 + 0.495 = 0.51, and did similarly for B is True, disregarding
    the value of variable A.

    :param cpt: the conditional probability table that we want to simplify through a summing-out
    :param variable: The variable that will be removed through summing out
    :return: the new conditional probability table
    """
    def find_complementary_row(cpt: pd.DataFrame, entry_row: pd.Series, index_to_switch: int) -> Tuple[pd.Series, int]:
        """
        Featuring the most complex list comprehension I have ever written, without a doubt.

        This function creates an array out of the values of the current row, excluding the probability row;
        it then switches the value of the variable to remove, before individually checking every row of the
        given CPT for the row that matches every value (there can only be one).
        It returns the appropriate row, as well as the index of that row, to easily drop it in the main function.

        :param cpt: the CPT through which we iterate to find the complementary value
        :param entry_row: the row of which we want to find the complementary value
        :param index_to_switch: the column index of the row we want to switch. In example above, this would be 0,
            since we want to add up probabilities for switched values of A
        :return: the complementary row and its index, or None if there were no complementaries (so the variable to sum out was given
            as evidence)
        """
        complementary_values = list(entry_row)[:-1]
        complementary_values[index_to_switch] = not complementary_values[index_to_switch]
        row_matches_conditions = [row.all() # every condition needs to evaluate to true
            for row in np.array( # We use numpy so can transpose
                [cpt[cpt.columns[i]] == complementary_values[i] # Compare cpt[SOME_COLUMN] with complementary_value[SOME_COLUMN]
                for i in range(len(complementary_values))] # for every column that we want to match
                ).T]
        try:
            index = row_matches_conditions.index(True)
            complementary_row = cpt.iloc[index]
            return complementary_row, index
        except:
            return None, None

    print(f"Summing out variable {variable} from \n{cpt}")
    res = cpt.copy()
    cols = list(res.columns)
    var_to_remove = cols.index(variable)
    indices_to_drop = []
    for ii, row in res.iterrows():
        if row[variable] == True:
            opposite_row, index = find_complementary_row(cpt, row, var_to_remove)
            if opposite_row is not None:
                res.loc[ii, "p"] += opposite_row["p"]
                indices_to_drop.append(index)
    res = res.drop(indices_to_drop)
    res = res.reset_index(drop=True)
    if res.columns[-2] != variable:
        res = res.drop(columns=[variable])
    print(f"End result is: \n{res}\n====================================")
    return res

if __name__ == "__main__":
    bifxml_path = os.getcwd() + "/testing/lecture_example2.BIFXML"
    bnr = BNReasoner(bifxml_path)
    print(bnr.marginal_distributions(['O', 'Y', 'X'], {}, bnr.min_degree()))
