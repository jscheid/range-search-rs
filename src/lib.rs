// Copyright 2021 Julian Scheid.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Efficient lookup of points inside a two-dimensional range.
//!
//! This crate implements a 2D [layered range tree], a data structure
//! that takes up a comparatively large amount of space [^space] and
//! construction time, but subsequently enables fast lookup of points
//! inside a given orthogonal (axis-aligned) range: queries take O(log
//! n + k) time, where _n_ is the number of points and _k_ is a number
//! of reported points much smaller than _n_.
//!
//! ```
//! use range_search::RangeTree2dBuilder;
//!
//! let points = (1..5)
//!     .flat_map(|y| (1..5).map(move |x| (x * 10, y * 10)))
//!     .collect::<Vec<_>>();
//! let builder = RangeTree2dBuilder::from(points);
//! let storage = builder.alloc_storage();
//! let tree = builder.build_in(&storage).unwrap();
//! let result: Vec<(u32, u32)> = tree.query(&(25..45), &(18..38)).collect();
//! assert_eq!(result, &[(30, 20), (30, 30), (40, 20), (40, 30)]);
//!
//! // Fast deallocation
//! std::mem::forget(tree);
//! std::mem::drop(storage);
//! ```
//!
//! Range trees can support any number of dimensions but this
//! implementation is optimized for, and only supports,
//! two-dimensional data.
//!
//! Also, range trees can support dynamic updates but this
//! implementation currently only supports static trees: any change to
//! the source point set requires rebuilding the whole tree.
//!
//! [^space]: This data structure takes up about 1-2 orders of
//! magnitude more space compared to a simple array, depending on the
//! total number of points and the size of the underlying data type.
//! Space complexity is O(n log n).
//!
//! [Layered Range Tree]: https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-851-advanced-data-structures-spring-2012/calendar-and-notes/MIT6_851S12_L3.pdf#page=7

#![no_std]
#![deny(missing_docs)]

#[macro_use]
extern crate alloc;

use alloc::vec::Vec;
use bumpalo::Bump;
use core::convert::TryFrom;
use core::ops::Range;

mod iter;
mod tree;
mod util;

use tree::{get_capacities, Point, RangeTree2dError, RangeTree2dImpl};

/// Required trait for the X and Y coordinates of points.
///
/// Coordinates must be [totally ordered] and support cloning.
///
/// The total ordering requirement means that you can't use [f32] or
/// [f64], but you can use an [ordered-float] or [noisy_float], say.
///
/// [totally ordered]: https://en.wikipedia.org/wiki/Total_order
/// [ordered-float]: https://docs.rs/crate/ordered-float/
/// [noisy_float]: https://docs.rs/crate/noisy_float/
pub trait Coord: Ord + Clone {}

impl<T> Coord for T where T: Ord + Clone {}

/// Knows how to build [RangeTree2d]s and [RangeTree2dStorage].
///
pub struct RangeTree2dBuilder<T: Coord>(Vec<Point<T>>);

enum RangeTree2dDispatch<'storage, T: Coord> {
    U16(RangeTree2dImpl<'storage, T, u16>),
    U32(RangeTree2dImpl<'storage, T, u32>),
    Usize(RangeTree2dImpl<'storage, T, usize>),
}

/// Layered Range Tree providing fast querying of points in a 2D range.
pub struct RangeTree2d<'storage, T: Coord>(RangeTree2dDispatch<'storage, T>);

/// Contiguous memory regions for storing [RangeTree2d] data.
///
/// Using allocation arenas here has several benefits: individual
/// allocations are much faster, which matters as we will have to
/// perform _n_ of them.  Deallocation is much faster, as we can just
/// drop the arenas in bulk.  Finally, cache locatity is improved.
///
/// Two different arenas are used, one for the first nesting level and
/// one for the second.  This again has multiple benefits: the data
/// stored in each of the two levels has different alignment
/// requirements so that we would have to waste space, or complicate
/// construction, by storing both in a single arena.  Also, cache
/// locality improves with two separate arenas.
///
/// You are free to create this struct and allocate the arenas
/// yourself using any strategy you see fit, but for the simple case
/// of storing a single range tree you are advised to use
/// `RangeTree2dBuilder::alloc_storage`.  Doing so wastes a small
/// percentage of memory due to pessimistic estimates but guarantees
/// that each arena uses only a single chunk of memory, which improves
/// cache locality and reduces deconstruction time slightly.
///
/// There is nothing speaking against reusing a `RangeTree2dStorage`
/// when rebuilding a tree from scratch, or for storing multiple
/// trees. In these cases you might want to use a different allocation
/// strategy since `RangeTree2dBuilder::alloc_storage` will not be
/// able to make a useful estimate.
pub struct RangeTree2dStorage {
    /// The arena used to store the nodes on the first level of the
    /// nested range tree (on X coordinates.)
    pub level1: Bump,

    /// The arena used to store the associated nodes on the second
    /// level of the nested range tree (on Y coordinates.)
    pub level2: Bump,
}

impl<T: Coord> From<Vec<Point<T>>> for RangeTree2dBuilder<T> {
    /// Create this builder from the given set of points, consuming it.
    ///
    /// The points you pass in must not contain any duplicates or
    /// [RangeTree2dBuilder::build_in] will error.
    fn from(points: Vec<Point<T>>) -> Self {
        RangeTree2dBuilder(points)
    }
}

impl<T: Coord> RangeTree2dBuilder<T> {
    // /// Create this builder from the given set of points, consuming it.
    // ///
    // /// The points you pass in must not contain any duplicates or
    // /// [RangeTree2dBuilder::build_in] will error.
    // pub fn new(points: Vec<Point<T>>) -> Self {
    //     RangeTree2dBuilder(points)
    // }

    /// Allocate storage suitable for storing a tree built from the
    /// points in this builder.
    pub fn alloc_storage(&self) -> RangeTree2dStorage {
        Self::alloc_storage_for(self.0.len())
    }

    /// Allocate storage suitable for storing a tree with the given
    /// number of points.
    pub fn alloc_storage_for(num_points: usize) -> RangeTree2dStorage {
        let (level1_capacity, level2_capacity) = {
            let index_max = num_points / 2 + 1;

            if u16::try_from(index_max).is_ok() {
                get_capacities::<T, u16>(num_points)
            } else if u32::try_from(index_max).is_ok() {
                get_capacities::<T, u32>(num_points)
            } else {
                get_capacities::<T, usize>(num_points)
            }
        };

        let level2 = Bump::with_capacity(level2_capacity);
        let level1 = Bump::with_capacity(level1_capacity);

        RangeTree2dStorage { level1, level2 }
    }

    /// Build the Range Tree using the given storage.
    pub fn build_in(
        self,
        storage: &RangeTree2dStorage,
    ) -> Result<RangeTree2d<T>, RangeTree2dError> {
        let index_max = self.0.len() / 2 + 1;

        // Note: it would be possible to use u8 for small data sets
        // (of 510 points or less) but in practice u16 appears to
        // perform better.  It doesn't matter either way since for
        // that small a data set the benefits of using a range tree
        // are questionable to begin with.
        //
        // Note: we don't have a u64 case: its role is met by usize on
        // 64-bit systems, and on 32-bit systems there is no need for
        // it since we can't store enough points in the address space
        // anyway.
        if u16::try_from(index_max).is_ok() {
            Ok(RangeTree2d(RangeTree2dDispatch::U16(RangeTree2dImpl::<
                T,
                u16,
            >::new(
                self.0, storage
            )?)))
        } else if u32::try_from(index_max).is_ok() {
            Ok(RangeTree2d(RangeTree2dDispatch::U32(RangeTree2dImpl::<
                T,
                u32,
            >::new(
                self.0, storage
            )?)))
        } else {
            Ok(RangeTree2d(RangeTree2dDispatch::Usize(RangeTree2dImpl::<
                T,
                usize,
            >::new(
                self.0, storage
            )?)))
        }
    }
}

impl<'storage, T: Coord> RangeTree2d<'storage, T> {
    /// Return an iterator lazily yielding all points that lie inside
    /// both the given X and Y range.
    ///
    /// You can perform an emptiness query by asking the iterator only
    /// for its first element; since the iterator is lazy, this is
    /// more efficient than collecting all points.
    pub fn query<'iterator>(
        &'iterator self,
        x_range: &'iterator Range<T>,
        y_range: &'iterator Range<T>,
    ) -> alloc::boxed::Box<dyn Iterator<Item = Point<T>> + 'iterator> {
        match &self.0 {
            RangeTree2dDispatch::U16(tree) => alloc::boxed::Box::new(tree.query(x_range, y_range)),
            RangeTree2dDispatch::U32(tree) => alloc::boxed::Box::new(tree.query(x_range, y_range)),
            RangeTree2dDispatch::Usize(tree) => {
                alloc::boxed::Box::new(tree.query(x_range, y_range))
            }
        }
    }
}

impl RangeTree2dStorage {
    /// Return the total number of bytes currently allocated in this
    /// storage.
    pub fn allocated_bytes(&self) -> usize {
        self.level1.allocated_bytes() + self.level2.allocated_bytes()
    }
}

#[cfg(test)]
pub mod utilities {
    use crate::tree::Point;
    use alloc::vec::Vec;
    use core::ops::Range;
    use ordered_float::NotNan;
    use rand::Rng;
    use rand_pcg::Pcg64;

    pub fn not_nan_range(start: f32, end: f32) -> Range<NotNan<f32>> {
        NotNan::new(start).unwrap()..NotNan::new(end).unwrap()
    }

    pub fn not_nan_tuple(x: f32, y: f32) -> (NotNan<f32>, NotNan<f32>) {
        (NotNan::new(x).unwrap(), NotNan::new(y).unwrap())
    }

    pub fn infinite_range() -> Range<NotNan<f32>> {
        NotNan::new(core::f32::NEG_INFINITY).unwrap()..NotNan::new(core::f32::INFINITY).unwrap()
    }

    pub fn random_point_vec(rng: &mut Pcg64, num_points: usize) -> Vec<(NotNan<f32>, NotNan<f32>)> {
        let mut arr = vec![];
        for _ in 0..num_points {
            arr.push(not_nan_tuple(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            ));
        }
        return arr;
    }

    pub fn naive_query<'a, T: Ord + Copy>(
        points: &'a Vec<Point<T>>,
        x_range: &Range<T>,
        y_range: &Range<T>,
    ) -> Vec<Point<T>> {
        points
            .iter()
            .filter(|p| x_range.contains(&p.0) && y_range.contains(&p.1))
            .map(|p| *p)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::util::cmp_x;
    use crate::utilities::{
        infinite_range, naive_query, not_nan_range, not_nan_tuple, random_point_vec,
    };
    use crate::RangeTree2dBuilder;
    use alloc::vec::Vec;
    use ordered_float::NotNan;
    use rand::prelude::*;
    use rand_pcg::Pcg64;

    #[test]
    fn it_works() {
        let points = vec![
            not_nan_tuple(3.0, 10.0),
            not_nan_tuple(7.0, 6.0),
            not_nan_tuple(11.0, 14.0),
            not_nan_tuple(16.0, 12.0),
            not_nan_tuple(23.0, 5.0),
            not_nan_tuple(30.0, 24.0),
            not_nan_tuple(42.0, 20.0),
        ];
        let points_copy = points.clone();

        let builder = RangeTree2dBuilder::from(points);
        let storage = builder.alloc_storage();
        let tree = builder.build_in(&storage).unwrap();

        let ranges = vec![
            (infinite_range(), infinite_range()),
            (not_nan_range(4.0, 25.0), infinite_range()),
            (not_nan_range(4.0, 21.0), infinite_range()),
            (not_nan_range(4.0, 12.0), infinite_range()),
            (not_nan_range(4.0, 9.0), infinite_range()),
            (not_nan_range(4.0, 5.0), infinite_range()),
            (not_nan_range(8.0, 12.0), infinite_range()),
            (not_nan_range(15.0, 18.0), infinite_range()),
            (not_nan_range(20.0, 25.0), infinite_range()),
            (not_nan_range(10.0, 17.0), not_nan_range(0.0, 13.0)),
            (not_nan_range(10.0, 17.0), not_nan_range(0.0, 11.0)),
            (infinite_range(), not_nan_range(0.0, 11.0)),
            (infinite_range(), not_nan_range(8.0, 22.0)),
        ];

        ranges.into_iter().for_each(|(x_range, y_range)| {
            let result: Vec<(NotNan<f32>, NotNan<f32>)> = tree.query(&x_range, &y_range).collect();
            assert_eq!(result, naive_query(&points_copy, &x_range, &y_range));
        });
    }

    #[test]
    fn case1() {
        let points = vec![
            not_nan_tuple(6.704619, 7.179145),
            not_nan_tuple(7.0622587, 7.1050587),
            not_nan_tuple(8.868534, 3.5678291),
            not_nan_tuple(2.0606363, 3.031677),
            not_nan_tuple(0.7817769, 2.810601),
            not_nan_tuple(0.26633263, 6.1269197),
            not_nan_tuple(3.5122561, 2.5835037),
            not_nan_tuple(5.509269, 6.2513885),
            not_nan_tuple(0.12477517, 6.3275075),
            not_nan_tuple(1.0669255, 1.4070714),
            not_nan_tuple(8.950378, 4.587141),
            not_nan_tuple(1.3949537, 5.3900967),
            not_nan_tuple(2.63363, 8.645783),
            not_nan_tuple(6.18145, 1.0271776),
            not_nan_tuple(0.44353604, 4.4920993),
            not_nan_tuple(0.73534966, 0.5263984),
            not_nan_tuple(2.713592, 3.6874294),
            not_nan_tuple(1.8319213, 8.172796),
            not_nan_tuple(0.10519266, 2.4926174),
            not_nan_tuple(2.2924244, 4.361762),
            not_nan_tuple(4.4110737, 4.49432),
            not_nan_tuple(3.0965877, 1.5774536),
            not_nan_tuple(0.7044816, 6.187668),
            not_nan_tuple(1.7017412, 9.313786),
            not_nan_tuple(0.2771914, 8.048183),
            not_nan_tuple(8.299381, 0.77507377),
            not_nan_tuple(5.7726836, 0.3043735),
            not_nan_tuple(7.208949, 8.360238),
            not_nan_tuple(0.9985328, 3.2439315),
        ];
        let points_clone = points.clone();

        let builder = RangeTree2dBuilder::from(points);
        let storage = builder.alloc_storage();
        let tree = builder.build_in(&storage).unwrap();

        let x_range = not_nan_range(0.25, 0.75);
        let y_range = not_nan_range(0.25, 0.75);

        let result: Vec<(NotNan<f32>, NotNan<f32>)> = tree.query(&x_range, &y_range).collect();
        assert_eq!(result, naive_query(&points_clone, &x_range, &y_range));
    }

    #[test]
    fn with_random_data() {
        for i in 0..50 {
            let mut rng = Pcg64::seed_from_u64(i);
            let points = random_point_vec(&mut rng, 10_000);
            let mut points_clone = points.clone();
            points_clone.sort_unstable_by(|a, b| cmp_x(a, b));

            let builder = RangeTree2dBuilder::from(points);
            let storage = builder.alloc_storage();
            let tree = builder.build_in(&storage).unwrap();

            let x_range = not_nan_range(0.25, 0.75);
            let y_range = not_nan_range(0.25, 0.75);

            let result: Vec<(NotNan<f32>, NotNan<f32>)> = tree.query(&x_range, &y_range).collect();
            assert_eq!(result, naive_query(&points_clone, &x_range, &y_range));
        }
    }

    #[test]
    fn with_duplicates() {
        let builder =
            RangeTree2dBuilder::from(vec![not_nan_tuple(1.0, 2.0), not_nan_tuple(1.0, 2.0)]);
        let storage = builder.alloc_storage();
        assert_eq!(
            builder.build_in(&storage).err(),
            Some(super::RangeTree2dError::DuplicatePoints)
        );
    }

    #[test]
    fn empty() {
        let builder = RangeTree2dBuilder::from(vec![]);
        let storage = builder.alloc_storage();
        let tree = builder.build_in(&storage).unwrap();
        let range = not_nan_range(0.25, 0.75);
        assert_eq!(tree.query(&range, &range).count(), 0);
    }

    #[test]
    fn single() {
        let builder = RangeTree2dBuilder::from(vec![not_nan_tuple(0.0, 0.0)]);
        let storage = builder.alloc_storage();
        let tree = builder.build_in(&storage).unwrap();

        {
            let range = not_nan_range(0.0, 1.0);
            assert_eq!(tree.query(&range, &range).count(), 1);
        }

        {
            let range = not_nan_range(0.01, 1.0);
            assert_eq!(tree.query(&range, &range).count(), 0);
        }

        {
            let range = not_nan_range(-1.0, 0.0);
            assert_eq!(tree.query(&range, &range).count(), 0);
        }

        {
            let range = not_nan_range(0.0, 0.0);
            assert_eq!(tree.query(&range, &range).count(), 0);
        }

        {
            let range = not_nan_range(1.0, 0.0);
            assert_eq!(tree.query(&range, &range).count(), 0);
        }
    }
}
