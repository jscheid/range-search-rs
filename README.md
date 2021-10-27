# `range-search`

**Implementation of a Layered Range Tree for two-dimensional data.**

[![](https://docs.rs/range-tree/badge.svg)](https://docs.rs/range-tree/)
[![](https://img.shields.io/crates/v/range-tree.svg)](https://crates.io/crates/range-tree)
[![](https://img.shields.io/crates/d/range-tree.svg)](https://crates.io/crates/range-tree)
[![Build Status](https://github.com/fitzgen/range-tree/workflows/Rust/badge.svg)](https://github.com/jscheid/range-tree/actions?query=workflow%3ARust)

## Range Search

A [Range Tree][https://en.wikipedia.org/wiki/range_tree] is an ordered tree data
structure to hold a list of points. It allows all points within a given
orthogonal (axis-aligned) range to be reported efficiently, and is typically
used in two or higher dimensions.

This crate currently only provides an implementation for the case of
two dimensions; support for other dimensionalities might be added later on.

Range trees are a good choice for when the queried range is small compared to
the total range of values, and when the same set of points will be queried more
than a couple of times: think finding all towns in a latitute/longitude range
that covers only a few square miles, although this data structure can obviously
be useful for non-geometrical problems.

Given this example it's easy to see why a range tree wouldn't be a good choice
when the queried range covers the whole planet: every point would be reported
which is much more efficient to do with a simpler data structure, such as an
array.

Likewise, if the tree will only be queried once or twice, the relatively large
cost of construction will outweigh the benefits of fast queries and you're again
better off using a simpler data structure.
