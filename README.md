# rs-cactusgraph
rs-cactusgraph is a Rust library for handling Cactus Graphs and related data structures (such as Biedged Graphs), as described in the paper **Superbubbles, Ultrabubbles, and Cacti by BENEDICT PATEN et al.** which can be found [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6067107/). You can also refer to [this link](https://gsocgraph.blogspot.com/) for further details on its implementation.

## Definitions
- A **Biedged Graph** is a graph with two types of edges: *black edges* and *gray edges*, such that each vertex is incident with at most one black edge.
- A **Cactus Graph** is a graph in which any two vertices are at most [*2-edge-connected*](https://en.wikipedia.org/wiki/K-edge-connected_graph). It can be obtained from a Biedged Graph.

## How to use this library
Currently rs-cactusgraph is not hosted on crates.io, so you will have to add it to your project as a git dependency. Therefore, in your **Cargo.toml** file, add the following dependency:

```
cactusgraph = { git = "https://github.com/HopedWall/rs-cactusgraph" }
```

You can also download this repository and add it as a local dependency, but it is not recommended.

## Available features
- Build a Biedged Graph from scratch, or from a [rs-handlegraph](https://github.com/chfi/rs-handlegraph)
- Export a Biedged Graph as a Dot file; [Graphviz](https://graphviz.org/) can then be used to obtain a graphical representation of the graph
- Convert a Biedged Graph into a Cactus Graph

## How to construct a Cactus Graph from a Biedged Graph
rs-cactusgraph implements the Cactus Graph construction algorithm as described in the paper linked above, which works as follows:
1. **Contract all the gray edges.** More information on edge contraction can be found [here](https://en.wikipedia.org/wiki/Edge_contraction).
2. **Merge each 3-connected-component into a single node.** This step relies on [chfi's rs-3-edge](https://github.com/chfi/rs-3-edge).
3. **Merge each cyclic-component into a single node.** This is done by using a DFS visit, and checking for back edges. Still a WIP.
