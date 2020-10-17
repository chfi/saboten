Saboten - cactus graphs and ultrabubbles in Rust
==============================================================

A Rust implementation of the algorithm described in [Superbubbles, Ultrabubbles and Cacti paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6067107/).

Supports transforming a GFA graph into a biedged graph, and further
into cactus graphs, cactus trees, and bridge forests.

These structures can then be used to find the ultrabubbles in the
graph (see the paper for more details).

For use of this library in a command line application, see
[gfautil](https://crates.io/crates/gfautil).

### Limitations

Input graphs must have all segment names as unsigned integers, and
tightly packed, e.g. from 0 to N-1, if there are N segments, however
the numbering does not have to start from zero.
