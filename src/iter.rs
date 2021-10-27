// Copyright 2021 Julian Scheid.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::tree::{AssocEntry, Index, Internal, Node, Point};
use crate::Coord;
use alloc::vec::Vec;

trait Branch<T: Coord> {
    fn offset<I: Index>(entry: &AssocEntry<T, I>) -> usize;
}

pub struct RangeTree2dIterator<'storage, 'iterator, T: Coord, I: Index> {
    stack: Vec<(&'iterator Internal<'storage, T, I>, usize)>,
    state: IteratorState<'storage, 'iterator, T, I>,
}

struct LeftBranch {}

impl<T: Coord> Branch<T> for LeftBranch {
    fn offset<I: Index>(entry: &AssocEntry<T, I>) -> usize {
        entry.left.try_into().unwrap_or_else(|_| unreachable!())
    }
}

struct RightBranch {}

impl<T: Coord> Branch<T> for RightBranch {
    fn offset<I: Index>(entry: &AssocEntry<T, I>) -> usize {
        entry.right.try_into().unwrap_or_else(|_| unreachable!())
    }
}

#[doc(hidden)]
struct LeftIteratorState<'storage, 'iterator, T: Coord, I: Index> {
    x_end: &'iterator T,
    y_end: &'iterator T,
    it: &'iterator Node<'storage, T, I>,
    y_index: usize,
    initial_y_index: usize,
    split_node: &'iterator Internal<'storage, T, I>,
}

#[doc(hidden)]
struct RightIteratorState<'storage, 'iterator, T: Coord, I: Index> {
    x_end: &'iterator T,
    y_end: &'iterator T,
    it: &'iterator Node<'storage, T, I>,
    y_index: usize,
}

enum IteratorState<'storage, 'iterator, T: Coord, I: Index> {
    Left(LeftIteratorState<'storage, 'iterator, T, I>),
    Right(RightIteratorState<'storage, 'iterator, T, I>),
    Just(&'iterator Point<T>),
    Finished,
}

impl<'storage, 'iterator, T: Coord, I: Index> RangeTree2dIterator<'storage, 'iterator, T, I> {
    pub fn from_split_node(
        split_node: &'iterator Internal<'storage, T, I>,
        initial_y_index: usize,
        x_start: &'iterator T,
        x_end: &'iterator T,
        y_end: &'iterator T,
    ) -> Self {
        let mut stack: Vec<(&'iterator Internal<'storage, T, I>, usize)> = vec![];
        let mut it: &Node<T, I> = &split_node.left;
        let mut y_index =
            follow_layer::<T, I, LeftBranch>(split_node, &split_node.left, initial_y_index, y_end);

        while let Node::Internal(internal) = it {
            if internal.left_max_x < *x_start {
                y_index =
                    follow_layer::<T, I, RightBranch>(internal, &internal.right, y_index, y_end);
                it = &internal.right;
            } else {
                stack.push((internal, y_index));
                y_index =
                    follow_layer::<T, I, LeftBranch>(internal, &internal.left, y_index, y_end);
                it = &internal.left;
            }
        }

        RangeTree2dIterator {
            stack,
            state: IteratorState::Left(LeftIteratorState {
                x_end,
                y_end,
                it,
                y_index,
                split_node,
                initial_y_index,
            }),
        }
    }

    pub fn just(point: &'iterator Point<T>) -> Self {
        RangeTree2dIterator {
            stack: vec![],
            state: IteratorState::Just(point),
        }
    }

    pub fn none() -> Self {
        RangeTree2dIterator {
            stack: vec![],
            state: IteratorState::Finished,
        }
    }
}

fn iterate_left<'storage, 'iterator, T: Coord, I: Index>(
    state: &LeftIteratorState<'storage, 'iterator, T, I>,
    stack: &mut Vec<(&'iterator Internal<'storage, T, I>, usize)>,
) -> (IteratorState<'storage, 'iterator, T, I>, Option<Point<T>>) {
    let mut y_index = state.y_index;
    let mut it = state.it;
    let y_end = state.y_end;

    loop {
        if y_index == 0 {
            if let Node::Leaf(leaf) = it {
                break (
                    IteratorState::Left(LeftIteratorState {
                        initial_y_index: state.initial_y_index,
                        split_node: state.split_node,
                        x_end: state.x_end,
                        y_end,
                        y_index: 1,
                        it,
                    }),
                    Some((*leaf).clone()),
                );
            } else {
                unreachable!();
            }
        }

        if let Some(head_node) = stack.pop() {
            it = &head_node.0.right;
            y_index = follow_layer::<T, I, RightBranch>(
                head_node.0,
                &head_node.0.right,
                head_node.1,
                y_end,
            );

            while let Node::Internal(internal) = it {
                stack.push((internal, y_index));
                y_index =
                    follow_layer::<T, I, LeftBranch>(internal, &internal.left, y_index, y_end);
                it = &internal.left;
            }
        } else {
            let y_end = state.y_end;

            let split_node = &state.split_node;
            let mut it = &split_node.right;
            let mut y_index = follow_layer::<T, I, RightBranch>(
                split_node,
                &split_node.right,
                state.initial_y_index,
                y_end,
            );
            let x_end = state.x_end;

            while let Node::Internal(internal) = it {
                if internal.left_max_x < *x_end {
                    stack.push((internal, y_index));
                }
                it = &internal.left;
                y_index =
                    follow_layer::<T, I, LeftBranch>(internal, &internal.left, y_index, y_end);
            }

            break (
                IteratorState::Right(RightIteratorState {
                    it,
                    x_end,
                    y_end,
                    y_index,
                }),
                None,
            );
        }
    }
}

fn iterate_right<'storage, 'iterator, T: Coord, I: Index>(
    state: &RightIteratorState<'storage, 'iterator, T, I>,
    stack: &mut Vec<(&'iterator Internal<'storage, T, I>, usize)>,
) -> (IteratorState<'storage, 'iterator, T, I>, Option<Point<T>>) {
    let mut y_index = state.y_index;
    let mut it = state.it;
    let y_end = state.y_end;
    let x_end = state.x_end;

    loop {
        if y_index == 0 {
            if let Node::Leaf(leaf) = it {
                if leaf.0 < *x_end {
                    break (
                        IteratorState::Right(RightIteratorState {
                            x_end,
                            y_end,
                            y_index: 1,
                            it,
                        }),
                        Some((*leaf).clone()),
                    );
                }
            } else {
                unreachable!();
            }
        }

        if let Some(head_node) = stack.pop() {
            it = &head_node.0.right;
            y_index = follow_layer::<T, I, RightBranch>(
                head_node.0,
                &head_node.0.right,
                head_node.1,
                y_end,
            );

            while let Node::Internal(internal) = it {
                if internal.left_max_x <= *x_end {
                    stack.push((internal, y_index));
                }
                it = &internal.left;
                y_index =
                    follow_layer::<T, I, LeftBranch>(internal, &internal.left, y_index, y_end);
            }
        } else {
            break (IteratorState::Finished, None);
        }
    }
}

impl<'storage, 'iterator, T: Coord, I: Index> Iterator
    for RangeTree2dIterator<'storage, 'iterator, T, I>
{
    type Item = Point<T>;

    fn next(&mut self) -> Option<Point<T>> {
        loop {
            let (next_state, result) = match &self.state {
                IteratorState::Finished => {
                    return None;
                }
                IteratorState::Just(value) => (IteratorState::Finished, Some((*value).clone())),
                IteratorState::Left(state) => iterate_left(state, &mut self.stack),
                IteratorState::Right(state) => iterate_right(state, &mut self.stack),
            };

            self.state = next_state;
            if result.is_some() {
                return result;
            }
        }
    }
}

fn follow_layer<'storage, 'iterator, T: Coord, I: Index, BRANCH: Branch<T>>(
    parent: &Internal<'storage, T, I>,
    child: &Node<'storage, T, I>,
    parent_y_index: usize,
    y_end: &'iterator T,
) -> usize {
    let mut iter = parent
        .assoc
        .iter()
        .skip(parent_y_index)
        .take_while(|entry| entry.value < *y_end);
    match child {
        Node::Internal(internal) => {
            if let Some(assoc) = iter.find(|entry| {
                let offset = BRANCH::offset(*entry);
                if offset == 0 {
                    false
                } else if let Some(item) = internal.assoc.get(offset - 1) {
                    item.value == entry.value
                } else {
                    false
                }
            }) {
                BRANCH::offset(assoc) - 1
            } else {
                internal.assoc.len()
            }
        }
        Node::Leaf(leaf) => {
            if let Some(assoc) = iter.find(|entry| leaf.1 == entry.value) {
                let offset = BRANCH::offset(assoc);
                if offset == 0 {
                    1
                } else {
                    offset - 1
                }
            } else {
                1
            }
        }
        Node::Empty => unreachable!(),
    }
}
