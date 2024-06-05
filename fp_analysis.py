#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import random 
import pickle5 as pickle
from pathlib import Path
from collections import defaultdict

import itertools

import unittest
  
class fpa():
    def findsubsets(s, n):
        
    #   A helper function that you can use to list of all subsets of size n. Do not make any changes to this code block.
    #   Input: 
    #       1. s - A python list of items
    #       2. n - Size of each subset
    #   Output:
    #       1. subsets - A python list containing the subsets of size n.
        
        subsets = list(sorted((itertools.combinations(s,n))))
        return subsets

    def items_from_frequent_itemsets(frequent_itemset):

    #   A helper function that you can use to get the sorted items from the frequent itemsets. Do not make any changes
    #   to this code block
    #   Input:
    #       1. Frequent Itemsets
    #   Output:
    #       1. Sorted list of items

        items = list()
        for keys in frequent_itemset.keys():
            for item in list(keys):
                items.append(item)
        return sorted(list(set(items)))

    def generate_frequent_itemsets(dataset, support, items, n=1, frequent_items={}):
        
    #   Input:
    #       1. dataset - A python dictionary containing the transactions.
    #       2. support - A floating point variable representing the min_support value for the set of transactions.
    #       3. items - A python list representing all the items that are part of all the transactions.
    #       4. n - An integer variable representing what frequent item pairs to generate.
    #       5. frequent_items - A dictionary representing k-1 frequent sets. 
    #   Output:
    #       1. frequent_itemsets - A dictionary representing the frequent itemsets and their corresponding support counts.
        
        len_transactions = len(dataset)
        if n == 1:
            # initialize frequent_items with 0
            frequent_items = {item:0 for item in items}
            
            # iterate over the transactions and update the count of each item in frequent_items
            for item in items:
                for key, transactions in dataset.items():
                    if item in transactions:
                        frequent_items[item] += 1
            # filter out the items that have support less than the min_support
            #frequent_itemsets = {key: value/len_transactions for key, value in frequent_items.items() if value/len_transactions >= support}
            frequent_itemsets = {key: value for key, value in frequent_items.items() if value/len_transactions >= support}
            
        
        else:
            
            # get items that are in the frequent items as starting point for combinations
            fpa_items = items_from_frequent_itemsets(frequent_items)
            # find all possible subsets of size n from the frequent_items
            subsets = findsubsets(fpa_items, n)
            # initialize frequent_items with 0
            frequent_itemsets = {key: 0 for key in subsets}
            # iterate over the transactions and update the count of each item in frequent_items
            for key in frequent_itemsets.keys():
                for t_key, transaction in dataset.items():
                    # check if the key is a subset of the transaction
                    if set(key).issubset(set(transaction)):
                        frequent_itemsets[key] += 1
            # filter out the items that have support less than the min_support
            #frequent_itemsets = {key: value/len_transactions for key, value in frequent_items.items() if value/len_transactions >= support}
            frequent_itemsets = {key: value for key, value in frequent_itemsets.items() if value/len_transactions >= support}
        
        return frequent_itemsets

    def item_support(dataset, min_support):
        
    #   A helper function that returns the support count of each item in the dataset.
    #   Input:
    #       1. dataset - A python dictionary containing the transactions. 
    #       2. min_support - A floating point variable representing the min_support value for the set of transactions.
    #   Output:
    #       1. support_dict - A dictionary representing the support count of each item in the dataset.
        
        len_transactions = len(dataset)
        support_dict = dict()
        for key, value in dataset.items():
            
            for item in list(value):
                if item in support_dict:
                    support_dict[item] += 1
                else:
                    support_dict[item] = 1
        
        ### For reference only
        sorted_support = dict(sorted(support_dict.items(), key=lambda item: item[1], reverse=True))
        pruned_support = {key:val for key, val in sorted_support.items() if val/len_transactions >= min_support}
        ###
        
        return support_dict

    def reorder_transactions(dataset, min_support):
        
    #   A helper function that reorders the transaction items based on maximum support count. It is important that you finish
    #   the code in the previous cells since this function makes use of the support count dictionary calculated above.
    #   Input:
    #       1. dataset - A python dictionary containing the transactions. 
    #       2. min_support - A floating point variable representing the min_support value for the set of transactions.
    #   Output:
    #       1. updated_dataset - A dictionary representing the transaction items in sorted order of their support counts.

        pruned_support = item_support(dataset, min_support) 
        updated_dataset = dict()
        
        # This loop sorts the transaction items based on the item support counts
        for key, value in dataset.items():
            updated_dataset[key] = sorted(value, key=pruned_support.get, reverse=True)
        
        # Update the following loop to remove items that do not belong to the pruned_support dictionary
        for key, value in updated_dataset.items():
            updated_values = list()
            for item in value:
                
                # reordering the items based on the support count
                # get the support count of the item
                item_support_count = pruned_support.get(item)
                # add item and support count to updated_values
                if item_support_count:
                    updated_values.append((item, item_support_count))
                    
            # sort updated values by item support count descending but keep only the item
            updated_values = [item[0] for item in sorted(updated_values, key=lambda x: x[1], reverse=True)]
                
            updated_dataset[key] = updated_values

        return updated_dataset
    def build_fp_tree(updated_dataset):
    
    #   Input: 
    #       1. updated_dataset - A python dictionary containing the updated set of transactions based on the pruned support dictionary.
    #   Output:
    #       1. fp_tree - A dictionary representing the fp_tree. Each node should have a count and children attribute.
    #        
    #   HINT:
    #       1. Loop over each transaction in the dataset and make an update to the fp_tree dictionary. 
    #       2. For each loop iteration store a pointer to the previously visited node and update it's children in the next pass.
    #       3. Update the root pointer when you start processing items in each transaction.
    #       4. Reset the root pointer for each transaction.
    #
    #   Sample Tree Output:
    #   {'Y': {'count': 3, 'children': {'V': {'count': 1, 'children': {}}}},
    #    'X': {'count': 2, 'children': {'R': {'count': 1, 'children': {'F': {'count': 1, 'children': {}}}}}}}
        
        fp_tree = dict()
        for key, value in updated_dataset.items():
            
            # your code here
            root_item = value[0]
            # check if item is in the fp_tree
            if root_item in fp_tree:
                fp_tree[root_item]['count'] += 1
            else:
                # add item to the fp_tree with spot for the children
                fp_tree[root_item] = {'count': 1, 'children': {}}
                
            # store pointer to the current node
            # set current node to the item
            current_node = fp_tree[root_item]
            root_node = fp_tree[root_item]
            for item in value:
                # check if item matches the root node
                if item != root_item:
                    if item in current_node['children']:
                        current_node['children'][item]['count'] += 1
                    else:
                        current_node['children'][item] = {'count': 1, 'children': {}}
                    current_node = current_node['children'][item]
            
        return fp_tree

if __name__ == "__main__":

    fpa_test = fpa()
    
    class TestX(unittest.TestCase):
        def setUp(self):
            self.min_support = 0.5
            self.items = ['A', 'B', 'C', 'D', 'E']
            self.dataset = dict()
            self.dataset["T1"] = ['A', 'B', 'D']
            self.dataset["T2"] = ['A', 'B', 'E']
            self.dataset["T3"] = ['B', 'C', 'D']
            self.dataset["T4"] = ['B', 'D', 'E']        
            self.dataset["T5"] = ['A', 'B', 'C', 'D']
            
        def test0(self):
            frequent_1_itemsets = fpa_test.generate_frequent_itemsets(self.dataset, self.min_support, self.items)
            print (frequent_1_itemsets)
            frequent_1_itemsets_solution = dict()
            frequent_1_itemsets_solution['A'] = 3
            frequent_1_itemsets_solution['B'] = 5
            frequent_1_itemsets_solution['D'] = 4

            print ("Test 1: frequent 1 itemsets")
            assert frequent_1_itemsets == frequent_1_itemsets_solution

            frequent_2_itemsets = fpa_test.generate_frequent_itemsets(self.dataset, self.min_support, self.items, 2, frequent_1_itemsets)
            print (frequent_2_itemsets)
            frequent_2_itemsets_solution = dict()
            frequent_2_itemsets_solution[('A', 'B')] = 3
            frequent_2_itemsets_solution[('B', 'D')] = 4
            
            print ("Test 1: frequent 2 itemsets")
            assert frequent_2_itemsets == frequent_2_itemsets_solution

            frequent_3_itemsets = fpa_test.generate_frequent_itemsets(self.dataset, self.min_support, self.items, 3, frequent_2_itemsets)
            print (frequent_3_itemsets)
            frequent_3_itemsets_solution = dict()

            print ("Test 1: frequent 3 itemsets")
            assert frequent_3_itemsets == frequent_3_itemsets_solution         
    
    tests = TestX()
    tests_to_run = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(tests_to_run)
    
    dataset = dict()
    with open('./data/dataset.pickle', 'rb') as handle:
        dataset = pickle.load(handle)

    support_dict = fpa_test.item_support(dataset, 0.5)
    support_dict_expected = {'C': 7, 'D': 9, 'E': 5, 'B': 6, 'A': 6}

    print(f'The expected support_dict value for the given dataset is: {support_dict_expected}')
    print(f'Your support_dict value is: {support_dict}')

    try:
        assert support_dict == support_dict_expected
        print("Visible tests passed!")
    except:
        print("Visible tests failed!")
        
    import pprint
    import json
    pp = pprint.PrettyPrinter(depth=4)

    dataset = dict()
    with open('./data/dataset.pickle', 'rb') as handle:
        dataset = pickle.load(handle)

    updated_dataset = fpa_test.reorder_transactions(dataset, 0.5)
    updated_dataset_expected = {'T1': ['D', 'C', 'E'], 'T2': ['D', 'C', 'B'], 'T3': ['D', 'C', 'A'],
                                'T4': ['D', 'C', 'A', 'E'], 'T5': ['D', 'C', 'A', 'B'], 'T6': ['B'],
                                'T7': ['D', 'E'], 'T8': ['D', 'C', 'A', 'B'], 'T9': ['D', 'A', 'B', 'E'], 'T10': ['D', 'C', 'A', 'B', 'E']}

    print(f'The expected updated_dataset value for the given dataset is:')
    pp.pprint(updated_dataset_expected)
    print(f'Your updated_dataset value is:')
    pp.pprint(updated_dataset)

    try:
        assert updated_dataset == updated_dataset_expected
        print("Visible tests passed!")
    except:
        print("Visible tests failed!")
        

    import pprint
    pp = pprint.PrettyPrinter(depth=8)

    dataset = dict()
    with open('./data/dataset.pickle', 'rb') as handle:
        dataset = pickle.load(handle)

    updated_dataset = fpa_test.reorder_transactions(dataset, 0.5)

    fp_tree = fpa_test.build_fp_tree(updated_dataset)
    fp_tree_expected = {'D': {'count': 9,
    'children': {'C': {'count': 7,
        'children': {'E': {'count': 1, 'children': {}},
        'B': {'count': 1, 'children': {}},
        'A': {'count': 5,
        'children': {'E': {'count': 1, 'children': {}},
        'B': {'count': 3, 'children': {'E': {'count': 1, 'children': {}}}}}}}},
    'E': {'count': 1, 'children': {}},
    'A': {'count': 1,
        'children': {'B': {'count': 1,
        'children': {'E': {'count': 1, 'children': {}}}}}}}},
    'B': {'count': 1, 'children': {}}}

    print(f'The expected fp_tree value for the given dataset is:')
    pp.pprint(fp_tree_expected)
    print(f'\nYour fp_tree value is:')
    pp.pprint(fp_tree)

    try:
        assert fp_tree == fp_tree_expected
        print("Visible tests passed!")
    except:
        print("Visible tests failed!")