// Copyright 2021 Julian Scheid.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::tree::Point;
use crate::Coord;
use core::cmp::Ordering;

pub fn cmp_x<T: Coord>(a: &Point<T>, b: &Point<T>) -> Ordering {
    // Note: a.0.cmp(b.0).then(a.1.cmp(b.1)) is prettier, but slower
    if a.0 < b.0 {
        Ordering::Less
    } else if a.0 > b.0 {
        Ordering::Greater
    } else if a.1 < b.1 {
        Ordering::Less
    } else if a.1 > b.1 {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

pub fn cmp_y<T: Coord>(a: &Point<T>, b: &Point<T>) -> Ordering {
    if a.1 < b.1 {
        Ordering::Less
    } else if a.1 > b.1 {
        Ordering::Greater
    } else if a.0 < b.0 {
        Ordering::Less
    } else if a.0 > b.0 {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}
