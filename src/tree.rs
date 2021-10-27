// Copyright 2021 Julian Scheid.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::iter::RangeTree2dIterator;
use crate::util::{cmp_x, cmp_y};
use crate::{Coord, RangeTree2dStorage};
use alloc::vec::Vec;
use bumpalo::boxed::Box;
use core::cmp::Ordering;
use core::convert::{TryFrom, TryInto};
use core::marker::PhantomData;
use core::ops::{Add, Range, Sub};

pub type Point<T> = (T, T);

#[derive(Debug, Default, Copy, Clone)]
pub struct AssocEntry<T: Coord, I: Index> {
    pub value: T,
    pub left: I,
    pub right: I,
}

#[derive(Debug)]
pub struct Internal<'storage, T: Coord, I: Index> {
    pub left_max_x: T,
    pub left: Node<'storage, T, I>,
    pub right: Node<'storage, T, I>,
    pub assoc: &'storage [AssocEntry<T, I>],
}

#[derive(Debug)]
pub enum Node<'storage, T: Coord, I: Index> {
    Leaf(
        // Note: storing the point data itself as the edge, rather
        // than a pointer to the data, may seem a somewhat unorthodox
        // choice but works well for f32 data on 64-bit machines and
        // wastes only a moderate amount of space for f64 data.  In
        // comparison, using a pointer wastes a larger amount of space
        // for f32 data and adds a level of indirection that reduces
        // query speed.
        //
        // Perhaps in a future implementation this could be abstracted
        // away so that the user can make this trade-off depending on
        // the size of the data type and of machine words.
        Point<T>,
    ),
    Internal(Box<'storage, Internal<'storage, T, I>>),
    Empty,
}

#[doc(hidden)]
pub trait Index:
    Copy
    + TryInto<usize>
    + TryFrom<usize>
    + From<u8>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Default
{
}

impl<T> Index for T where
    T: Copy
        + TryInto<usize>
        + TryFrom<usize>
        + From<u8>
        + Add<Output = Self>
        + Sub<Output = Self>
        + Default
{
}

#[doc(hidden)]
pub struct RangeTree2dImpl<'storage, T: Coord, I: Index>(Node<'storage, T, I>);

#[derive(Debug, PartialEq)]
pub enum RangeTree2dError {
    /// Range trees can't handle duplicates, and we don't want to
    /// have to de-duplicate all incoming data as this would be
    /// wasteful.
    DuplicatePoints,

    Overflow,
}

struct BuildIterator<'temp, T: Coord, I: Index, const TOP_LEVEL: bool> {
    y_sorted: &'temp mut [&'temp Point<T>],
    index: usize,
    median: &'temp Point<T>,
    prev_point: Option<&'temp Point<T>>,
    error: Option<RangeTree2dError>,
    left_count: usize,
    right_count: usize,
    scratch_left: &'temp mut [&'temp Point<T>],
    scratch_right: &'temp mut [&'temp Point<T>],
    index_type: PhantomData<I>,
}

impl<'temp, T: Coord, I: Index, const TOP_LEVEL: bool> Iterator
    for BuildIterator<'temp, T, I, TOP_LEVEL>
{
    type Item = AssocEntry<T, I>;

    fn next(&mut self) -> Option<AssocEntry<T, I>> {
        match self.y_sorted.get(self.index) {
            Some(point) => {
                self.index += 1;

                if TOP_LEVEL {
                    if let Some(prev_point) = self.prev_point {
                        if cmp_y(prev_point, point) == Ordering::Equal {
                            self.error = Some(RangeTree2dError::DuplicatePoints);
                            return Some(AssocEntry {
                                value: self.median.1.clone(),
                                left: I::from(0u8),
                                right: I::from(0u8),
                            });
                        }
                    }
                }
                self.prev_point = Some(point);

                let left = I::try_from(self.left_count + 1)
                    .unwrap_or_else(|_| panic!("Internal error: overflow"));
                let right = I::try_from(self.right_count + 1)
                    .unwrap_or_else(|_| panic!("Internal error: overflow"));

                let is_same_y = if let Some(prev_point) = self.prev_point {
                    point.1 == prev_point.1
                } else {
                    false
                };

                if cmp_x(point, self.median) == Ordering::Greater {
                    self.scratch_right[self.right_count] = *point;
                    self.right_count += 1;
                } else {
                    self.scratch_left[self.left_count] = *point;
                    self.left_count += 1;
                }

                Some(AssocEntry {
                    value: point.1.clone(),
                    left: if is_same_y { left } else { I::from(0u8) },
                    right: if is_same_y { right } else { I::from(0u8) },
                })
            }
            None => None,
        }
    }
}

fn build_recursive<'storage, 'temp, T: Coord, I: Index, const TOP_LEVEL: bool>(
    storage: &'storage RangeTree2dStorage,
    x_sorted: &'temp [Point<T>],
    y_sorted: &'temp mut [&'temp Point<T>],
    y_scratch: &'temp mut [&'temp Point<T>],
) -> Result<Node<'storage, T, I>, RangeTree2dError> {
    let mut x_sorted_iter = x_sorted.iter();
    let x_sorted_first = x_sorted_iter.next();
    if let Some(element) = x_sorted_first {
        if x_sorted_iter.next().is_none() {
            return Ok(Node::Leaf((*element).clone()));
        }
    } else {
        return Ok(Node::Empty);
    }

    let (x_left, x_right) = x_sorted.split_at((x_sorted.len() + 1) / 2);
    let median = x_left
        .last()
        .expect("Should have at least one element on the left");

    let (scratch_left, scratch_right) = y_scratch.split_at_mut(x_left.len());

    let num_points = y_sorted.len();
    let mut build_it: BuildIterator<'temp, T, I, TOP_LEVEL> = BuildIterator {
        y_sorted,
        index: 0,
        median,
        prev_point: None,
        error: None,
        left_count: 0,
        right_count: 0,
        scratch_left,
        scratch_right,
        index_type: PhantomData,
    };

    // Note: not using `bumpalo::Bump::alloc_slice_fill_iter` here
    // since it wants an `Iterator` rather than a reference to one,
    // and causing all kinds of lifetime problems.
    let assoc = storage
        .level2
        .alloc_slice_fill_with(num_points, |_| build_it.next().unwrap());

    if let Some(err) = build_it.error {
        return Err(err);
    }

    let (next_scratch_left, next_scratch_right) = build_it.y_sorted.split_at_mut(x_left.len());

    let left =
        build_recursive::<T, I, false>(storage, x_left, build_it.scratch_left, next_scratch_left);

    let right = build_recursive::<T, I, false>(
        storage,
        x_right,
        build_it.scratch_right,
        next_scratch_right,
    );

    Ok(Node::Internal(Box::new_in(
        Internal {
            left_max_x: median.0.clone(),
            left: left.expect("Shouldn't be able to find duplicates in nested nodes"),
            right: right.expect("Shouldn't be able to find duplicates in nested nodes"),
            assoc,
        },
        &storage.level1,
    )))
}

impl<'storage, T: Coord, I: Index> RangeTree2dImpl<'storage, T, I> {
    pub fn new(
        mut points: Vec<Point<T>>,
        storage: &'storage RangeTree2dStorage,
    ) -> Result<Self, RangeTree2dError> {
        points.sort_unstable_by(|a, b| cmp_x(a, b));
        let mut y_sorted = Vec::<&Point<T>>::with_capacity(points.len() * 2);
        y_sorted.extend(points.iter().into_iter());
        y_sorted.sort_unstable_by(|a, b| cmp_y(*a, *b));
        let num_points = points.len();

        assert_eq!(y_sorted.len(), num_points);

        unsafe {
            y_sorted.set_len(y_sorted.capacity());
        }

        let (y_sorted, y_scratch) = y_sorted.split_at_mut(num_points);
        let result = build_recursive::<T, I, true>(storage, points.as_slice(), y_sorted, y_scratch);

        result.map(|root| RangeTree2dImpl(root))
    }

    pub fn query<'iterator>(
        &'iterator self,
        x_range: &'iterator Range<T>,
        y_range: &'iterator Range<T>,
    ) -> RangeTree2dIterator<'storage, 'iterator, T, I> {
        if x_range.end <= x_range.start || y_range.end <= y_range.start {
            return RangeTree2dIterator::none();
        }

        let mut node = &self.0;

        loop {
            match node {
                Node::Internal(internal) => {
                    node = if x_range.start > internal.left_max_x {
                        &internal.right
                    } else if x_range.end <= internal.left_max_x {
                        &internal.left
                    } else {
                        let index = match internal
                            .assoc
                            .binary_search_by_key(&&y_range.start, |elt| &elt.value)
                        {
                            Ok(index) => index,
                            Err(index) => index,
                        };

                        return RangeTree2dIterator::from_split_node(
                            internal,
                            index,
                            &x_range.start,
                            &x_range.end,
                            &y_range.end,
                        );
                    };
                }
                Node::Leaf(leaf) => {
                    return if x_range.contains(&leaf.0) && y_range.contains(&leaf.1) {
                        RangeTree2dIterator::just(leaf)
                    } else {
                        RangeTree2dIterator::none()
                    }
                }
                Node::Empty => return RangeTree2dIterator::none(),
            }
        }
    }
}

pub fn get_capacities<T: Coord, I: Index>(num_points: usize) -> (usize, usize) {
    let tree_height = ceil_log2(num_points);
    let num_assoc_entries = num_points * tree_height;

    (
        num_points
            .checked_mul(core::mem::size_of::<Internal<T, I>>())
            .expect("Out of memory"),
        num_assoc_entries
            .checked_mul(core::mem::size_of::<AssocEntry<T, I>>())
            .expect("Out of memory"),
    )
}

fn ceil_log2(val: usize) -> usize {
    const BITS_PER_BYTE: usize = 8;

    BITS_PER_BYTE * core::mem::size_of::<usize>() - (val.leading_zeros() as usize)
}
