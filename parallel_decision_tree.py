from abc import ABC

class ParallelTree:
    def __init__(self,dataSet) -> None:
        self.root = None
        self.data = dataSet
        
class Node(ABC):
    def __init__(self) -> None:
        pass
class InternalNode(Node):
    def __init__(self) -> None:
        pass
class LeafNode(Node):
    def __init__(self) -> None:
        pass