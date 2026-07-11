# Shared Stack and Node primitives for Lesson exercises, adapted from Master repository

import sys
import os

datastructure_dir = os.path.dirname(os.path.abspath(__file__))
master_path = os.path.join(datastructure_dir, "Data_Structures_Algorithms_In_Python-master")
if master_path not in sys.path:
    sys.path.insert(0, master_path)

from Stack_and_Queue.Stack import Stack as MasterStack, StackNode as MasterStackNode


class Node(MasterStackNode):
    """Adapter for StackNode supporting 'value' alias for compatibility."""
    def __init__(self, value=None, next=None):
        super().__init__(value)
        self.next = next

    @property
    def value(self):
        return self.data

    @value.setter
    def value(self, val):
        self.data = val


class Stack(MasterStack):
    """Adapter for Master Stack supporting 'top' alias, silencing stdout prints."""
    def __init__(self):
        super().__init__()

    @property
    def top(self):
        return self.root

    @top.setter
    def top(self, val):
        self.root = val

    def push(self, value):
        # Override to avoid the print statement in MasterStack.push
        newNode = Node(value)
        newNode.next = self.root
        self.root = newNode

    def pop(self):
        # Override to return None when empty instead of float("-inf")
        if self.isEmpty():
            return None
        temp = self.root
        self.root = self.root.next
        return temp.data

    def peek(self):
        # Override to return None when empty instead of float("-inf")
        if self.isEmpty():
            return None
        return self.root.data

    def __len__(self):
        curr = self.root
        size = 0
        while curr:
            size += 1
            curr = curr.next
        return size

    def __str__(self):
        curr = self.root
        nodes = []
        while curr:
            nodes.append(str(curr.data))
            curr = curr.next
        nodes.append("None")
        return ",".join(nodes)
