#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:03:34 2019

@author: petern
"""

# Python program to reverse a singly linked list


# Node class
class Node:

    # Constructor to initialise data and next
    def __init__(self, data=None):
        self.data = data
        self.next = None

class SinglyLinkedList:

    # Constructor to initialise head
    def __init__(self):
        self.head = None

    # Function to reverse a linked list
    def bubble_sort_exdata(self):
        
        end = None
        
        while end != self.head.next:
            p = self.head
            while p.next != end:
                q = p.next
                if p.data > q.data:
                        p.data,q.data = q.data,p.data
                p = p.next
            end = p
            
    def bubble_sort_exlink(self):
            
        end = None
        
        while end != self.head.next:
            
            r = p = self.head
            while p.next != end:
                q= p.next
                
                if  p.data > q.data:
                        p.next = q.next
                        q.next = p
                        if p != self.head:
                             r.next = q
                        else:
                            self.head = q
                        p,q = q,p
                r = p
                p = p.next
            end = p
    

    # Function to Insert data at the beginning of the linked list
    def insert_at_beg(self, data):
        node = Node(data)
        node.next = self.head
        self.head = node

    # Function to print the linked list
    def print_data(self):
        current = self.head
        while current is not None:
            print(current.data, '-> ', end='')
            current = current.next
        print('None')

if __name__ == '__main__':
   linked_list = SinglyLinkedList()
   linked_list.insert_at_beg(4)
   linked_list.insert_at_beg(6)
   linked_list.insert_at_beg(5)
   linked_list.insert_at_beg(7)
   linked_list.insert_at_beg(3)
   linked_list.insert_at_beg(2)
   linked_list.insert_at_beg(1)
   linked_list.print_data()
   # call the reverse function
   linked_list.bubble_sort_exdata()
   # linked_list.bubble_sort_exlink()
   # print the reversed list
   linked_list.print_data()
