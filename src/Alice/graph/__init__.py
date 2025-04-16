# fmt: off
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(parent_dir)

from Alice.graph.graph import create_graph

# fmt: on

graph_builder = create_graph()
