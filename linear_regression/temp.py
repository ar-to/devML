
class Node:
  def __init__(self, value):
      self.value = value
      self.next = None
  def set_next(self, next_node):
      self.next = next_node

node1 = Node(1)
node2 = Node(2)
node1.set_next(node2)

print(node1) 
print(node1.value)  # Output: 1
print(node1.next.value)  # Output: 2

# for n in node1:
#     print(n.value)

# This technically iterates backwards. The node sets the head, which is the final. Node and iterates backwards until there is no more note in the dot next. 
curr = node1
while curr is not None:
    print(curr.value)
    curr = curr.next

# ai
# Build: 1 -> 2 -> 3
head = Node(1)
head.next = Node(2)
head.next.next = Node(3)

# Iterate
current = head
while current:
    print(current.value)
    current = current.next

print("Done iterating through linked list.")

print(node1) 
print(head) 

def merge_lists(node1, head):
  a = node1
  b = head
  c = None # New node. 
  while a is not None or b is not None:
      # val_a = a.value if a else None
      # val_b = b.value if b else None
      # print(val_a, val_b)
      print(a.value if a else None, b.value if b else None)
      if c is None:
          if a is not None and b is not None:
              if a.value <= b.value:
                  c = Node(a.value)
                  a = a.next
              else:
                  c = Node(b.value)
                  b = b.next
          elif a is not None:
          if a is not None and a.value <= b.value:
          if a is not None and a.value <= (b.value if b else float('inf')):
              c = Node(a.value)
              a = a.next
          if a and (not b or a.value <= b.value):
              c = Node(a.value)
              a = a.next
          elif b:
              c = Node(b.value)
              b = b.next
      if a: a = a.next
      if b: b = b.next