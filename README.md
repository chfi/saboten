# rs-cactusgraph
rs-cactusgraph is a Rust library for handling Cactus Graphs and related data structures (such as Biedged Graphs), as described in the paper Superbubbles, Ultrabubbles, and Cacti by BENEDICT PATEN et al. which can be found [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6067107/). You can also refer to [this link](https://gsocgraph.blogspot.com/) for further details on its implementation.

## How to use this library
Currently rs-cactusgraph is not hosted on crates.io, so you will have to add it to your project as a git dependency. Therefore, in your Cargo.toml file, add the following dependency:

```
cactusgraph = { git = "https://github.com/HopedWall/rs-cactusgraph" }
```

You can also download this repository and add it as a local dependency, but this is not recommended.

## Available features
1. Build a Biedged Graph from scratch, or from a [rs-handlegraph](https://github.com/chfi/rs-handlegraph)
2. Export a Biedged Graph as a Dot file; [Graphviz](https://graphviz.org/) can then be used to obtain a graphical representation of the graph
3. Convert a Biedged Graph into a Cactus Graph

## How to construct a Cactus Graph
rs-cactusgraph implements the Cactus Graph construction algorithm as described in the paper linked above, which works as follows:
1. **Create a Biedged Graph**
2. **Contract all the gray edges**
3. **Merge each 3-connected-component into a single node**
4. **Merge each cyclic-component into a single node**
