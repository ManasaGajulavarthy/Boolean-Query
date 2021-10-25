'''
@author: Sougata Saha
Institute: University at Buffalo
'''

import math


class Node:

    def __init__(self, value=None, next=None, tf = 0, tfidf = 0, skipConn = None):
        """ Class to define the structure of each node in a linked list (postings list).
            Value: document id, Next: Pointer to the next node
            Add more parameters if needed.
            Hint: You may want to define skip pointers & appropriate score calculation here"""
        self.value = value
        self.next = next
        self.tf = tf
        self.tfidf = tfidf
        self.skipConn = skipConn

class LinkedList:
    """ Class to define a linked list (postings list). Each element in the linked list is of the type 'Node'
        Each term in the inverted index has an associated linked list object.
        Feel free to add additional functions to this class."""
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def update_tf_idf(self, idf):
        if self.start_node is None:
            print("List has no element")
            return
        else:
            n = self.start_node
            while n is not None:
                if n.value == 69375:
                    x = 0
                n.tfidf = n.tf*idf
                n = n.next
        return

    def traverse_list(self):
        traversal = []
        if self.start_node is None:
            print("List has no element")
            return
        else:
            n = self.start_node
            # Start traversal from head, and go on till you reach None
            while n is not None:
                traversal.append(n.value)
                n = n.next
        return traversal

    def traverse_skips(self):
        traversal = []
        if self.start_node is None:
            return
        else:
            """ Write logic to traverse the linked list using skip pointers.
                To be implemented."""
            n = self.start_node
            while n is not None:
                if n.skipConn :
                    if n.value not in traversal:
                        traversal.append(n.value)
                    traversal.append(n.skipConn)
                n = n.next
            return traversal

    def add_skip_connections(self):
        n_skips = math.floor(math.sqrt(self.length))
        if n_skips * n_skips == self.length:
            n_skips = n_skips - 1
        """ Write logic to add skip pointers to the linked list. 
            This function does not return anything.
            To be implemented."""
        self.skip_length = round(math.sqrt(self.length), 0)
        node = self.start_node
        if self.length > 2:
            for i in range(n_skips):
                j = 0
                temp_node = node
                #print("temp_node" + str(temp_node.value))
                while j < self.skip_length and temp_node:
                    temp_node = temp_node.next
                    j += 1
                #print("temp_node_skip" + str(temp_node.value))
                node.skipConn = temp_node.value
                node = temp_node
        return

    def insert_at_end(self, value, total_tokens):
        """ Write logic to add new elements to the linked list.
            Insert the element at an appropriate position, such that elements to the left are lower than the inserted
            element, and elements to the right are greater than the inserted element.
            To be implemented. """
        #value= value,params.skipConnections,params.tf_idf
        n = self.start_node

        if self.start_node is None:
            new_node = Node(value=value)
            self.length += 1
            self.start_node = new_node
            self.end_node = new_node
            new_node.tf += (1/total_tokens)
            return

        elif self.start_node.value > value:
            new_node = Node(value=value)
            self.length += 1
            self.start_node = new_node
            self.start_node.next = n
            new_node.tf += (1/total_tokens)
            return

        elif self.end_node.value < value:
            new_node = Node(value=value)
            self.length += 1
            self.end_node.next = new_node
            self.end_node = new_node
            new_node.tf += (1/total_tokens)
            return

        elif self.start_node.value == value:
            self.start_node.tf += 0
            self.start_node.tf += (1/total_tokens)

        elif self.end_node.value == value:
            self.start_node.tf += 0
            self.end_node.tf += (1/total_tokens)

        else:
            while n.value <= value <= self.end_node.value and n.next is not None:
                if n.value == value:
                    n.tf += (1/total_tokens)
                    return
                n = n.next

            new_node = Node(value=value)
            self.length += 1
            m = self.start_node
            while m.next != n and m.next is not None:
                m = m.next
            m.next = new_node
            new_node.next = n
            new_node.tf += (1/total_tokens)
            return

