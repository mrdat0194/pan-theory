#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:16:49 2019

@author: petern
"""
import pandas as pd



#
# data = {'id': [1, 2, 3, 4], 'name': ['Alice', 'Bob', 'Charlie', 'David']}
# df = pd.DataFrame(data)
# print(df)
# # Drop the second row
# df = df.drop(index=1)
#
# # Reset the index
# df = df.reset_index(drop=True)
#
# print(df)

#
# # # #675
# # # This problem was asked by Google.
# # #
# # # You are given a set of synonyms, such as (big, large) and (eat, consume). Using this set, determine if two sentences with the same number of words are equivalent.
# # #
# # # For example, the following two sentences are equivalent:
# # #
# # # "He wants to eat food."
# # # "He wants to consume food."
# # # Note that the synonyms (a, b) and (a, c) do not necessarily imply (b, c): consider the case of (coach, bus) and (coach, teacher).
# # #
# # # Follow-up: what if we can assume that (a, b) and (a, c) do in fact imply (b, c)?
#
# if __name__ == '__main__':
#
#     # Best answer
#     class DisjointSet(object):
#         def __init__(self):
#             self.parents = {}
#
#         def get_root(self, w):
#             words_traversed = []
#             while w in self.parents and self.parents[w] != w:
#                 words_traversed.append(w)
#                 w = self.parents[w]
#             for word in words_traversed:
#                 self.parents[word] = w
#             return w
#
#         def add_synonyms(self, w1, w2):
#             if w1 not in self.parents:
#                 self.parents[w1] = w1
#             if w2 not in self.parents:
#                 self.parents[w2] = w2
#
#             w1_root = self.get_root(w1)
#             w2_root = self.get_root(w2)
#             if w1_root < w2_root:
#                 w1_root, w2_root = w2_root, w1_root
#             self.parents[w2_root] = w1_root
#
#         def are_synonymous(self, w1, w2):
#             return self.get_root(w1) == self.get_root(w2)
#
#
#     def preprocess_synonyms(synonym_words):
#         ds = DisjointSet()
#         for w1, w2 in synonym_words:
#             ds.add_synonyms(w1, w2)
#         return ds
#
#
#     def synonym_queries(synonym_words, queries):
#         '''
#         synonym_words: iterable of pairs of strings representing synonymous words
#         queries: iterable of pairs of strings representing queries to be tested for
#                  synonymous-ness
#         '''
#         synonyms = preprocess_synonyms(synonym_words)
#         print(synonyms.get_root("big"))
#
#         output = []
#         for q1, q2 in queries:
#             q1, q2 = q1.split(), q2.split()
#             print(q1)
#             if len(q1) != len(q2):
#                 output.append(False)
#                 continue
#             result = True
#             for i in range(len(q1)):
#                 w1, w2 = q1[i], q2[i]
#                 if w1 == w2:
#                     continue
#                 elif synonyms.are_synonymous(w1, w2):
#                     continue
#                 else:
#                     result = False
#                     break
#             output.append(result)
#         return output
#
#     dictionary = [('big', 'large'),('eat', 'consume')]
#     queries = [("He wants to eat big food.","He wants to consume huge food.")]
#     print(synonym_queries(dictionary, queries))
#


