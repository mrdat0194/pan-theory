# class of a node
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


# class of the signly linked list
class SinglyLinkedList:
    def __init__(self):
        self.head = None

    # function to calculate size of the list
    def size(self):
        # initialize temporary variable and size to zero
        current = self.head
        size = 0

        # count until current does not reach the end of the list i.e. NULL or None
        while current is not None:
            size += 1
            current = current.next
        return size

    # function to insert at the end of the list
    def insert_at_end(self, data):
        # Create the new node
        node = Node(data)

        # If the list is empty
        if self.head is None:
            self.head = node
        else:
            current = self.head

            # Otherwise move to the last node of the list
            while current.next is not None:
                current = current.next

            # Point the last node of the list to the new node
            # So that the new node gets added to the end of the list
            current.next = node

    # function to insert at the beginning of the list
    def insert_at_beg(self, data):
        node = Node(data)
        # Next pointer of the new node points to the head
        node.next = self.head

        # Updated the head as new node
        self.head = node

    # function to delete from the end of the list
    def delete_from_end(self):
        current = self.head
        previous = None

        if current is None:
            print("Linked List Underflow!!")
        else:
            while current.next is not None:
                previous = current
                current = current.next

            if previous is None:
                self.head = None
            else:
                previous.next = None

    # function to delete from the beginning of the list
    def delete_from_beg(self):
        current = self.head

        if current is None:
            print("Linked List Underflow!!")
        else:
            self.head = current.next

    # function to print the linked list data
    def print_data(self):
        current = self.head

        while current is not None:
            print(current.data, '->', end='')
            current = current.next
        print('End of list')
        
    def index_valuefind(self,k):
        current = self.head
        i = 1
        while i < k and current is not None:
            current = current.next
            i+=1
            
        abc = current.data
        print(abc)    


# Main program:
if __name__ == '__main__': 
    linked_list = SinglyLinkedList()
    linked_list.insert_at_beg(3)
    linked_list.insert_at_beg(2)
    linked_list.insert_at_beg(1)
    linked_list.insert_at_beg(9)
    linked_list.print_data()
    linked_list.index_valuefind(2)
