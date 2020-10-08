Saboten - cactus graphs and ultrabubbles in Rust
==============================================================

A Rust implementation of the algorithm described in [Superbubbles, Ultrabubbles and Cacti paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6067107/).

Supports transforming a GFA graph into a biedged graph, and further
into cactus graphs, cactus trees, and bridge forests.

These structures can then be used to find the ultrabubbles in the
graph (see the paper for more details).

This crate can be used either as a command line application, or as a
library in other Rust projects.

### Command line usage

```bash
$ saboten --help
saboten 0.0.1

USAGE:
    saboten [FLAGS] <gfa>

FLAGS:
    -h, --help       Prints help information
    -j, --json       Output JSON
    -V, --version    Prints version information

ARGS:
    <gfa>    Path to input GFA file
```


All the command line tool needs is the path to the input GFA.

The default output is one ultrabubble per line, with a tab between the
start and end of the bubble, for example:

```
7	9
3	11
12	15
0	3
```

If the `--json` flag is turned on, output will instead be a JSON
array, where each element is an object with a `start` and an `end`
field, denoting the ultrabubble start and end nodes:

```json
[{"start":7,"end":9},
 {"start":3,"end":11},
 {"start":12,"end":15},
 {"start":0,"end":3}]
```


### Use as a library

Library usage will look very much like the `main()` function of the
command line tool, transforming a GFA into a biedged graph, and then
building the rest of the graph structures off of that, before using
the cactus tree and bridge forest with the `find_ultrabubbles_par`
function.


### Limitations

Input graphs must have all segment names as unsigned integers, and
tightly packed, e.g. from 0 to N-1, if there are N segments, however
the numbering does not have to start from zero.
