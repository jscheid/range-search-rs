// Copyright 2021 Julian Scheid.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ordered_float::NotNan;
use rand::prelude::*;
use rand::Rng;
use rand_pcg::Pcg64;
use range_search::RangeTree2dBuilder;
use std::ops::Range;

pub fn not_nan_range(start: f32, end: f32) -> Range<NotNan<f32>> {
    NotNan::new(start).unwrap()..NotNan::new(end).unwrap()
}

pub fn not_nan_tuple(x: f32, y: f32) -> (NotNan<f32>, NotNan<f32>) {
    (NotNan::new(x).unwrap(), NotNan::new(y).unwrap())
}

pub fn infinite_range() -> Range<NotNan<f32>> {
    NotNan::new(std::f32::NEG_INFINITY).unwrap()..NotNan::new(std::f32::INFINITY).unwrap()
}

pub fn random_point_collection(
    rng: &mut Pcg64,
    num_points: usize,
) -> Vec<(NotNan<f32>, NotNan<f32>)> {
    let mut arr = vec![];
    for _ in 0..num_points {
        arr.push(not_nan_tuple(
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
        ));
    }
    return arr;
}

pub fn construction_time(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("construction time");
    group.sample_size(10);

    for num_points_total in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_points_total),
            num_points_total,
            |b, &num_points_total| {
                let mut rng = Pcg64::seed_from_u64(2);
                b.iter_batched(
                    || random_point_collection(&mut rng, num_points_total),
                    |points| {
                        let builder = RangeTree2dBuilder::from(points);
                        let storage = builder.alloc_storage();
                        let tree = builder.build_in(&storage).unwrap();
                        black_box(tree);
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
}

#[derive(Clone, Copy)]
struct QueryBenchmarkConfig {
    approx_num_reported: usize,
    num_points_total: usize,
}

impl std::fmt::Display for QueryBenchmarkConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "approx_num_reported={}, num_points_total={}",
            self.approx_num_reported, self.num_points_total
        )
    }
}

pub fn query_time(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("query");

    for config in [
        QueryBenchmarkConfig {
            approx_num_reported: 10,
            num_points_total: 1_000,
        },
        QueryBenchmarkConfig {
            approx_num_reported: 10,
            num_points_total: 10_000,
        },
        QueryBenchmarkConfig {
            approx_num_reported: 10,
            num_points_total: 100_000,
        },
        QueryBenchmarkConfig {
            approx_num_reported: 10,
            num_points_total: 1_000_000,
        },
        QueryBenchmarkConfig {
            approx_num_reported: 100,
            num_points_total: 1_000_000,
        },
        QueryBenchmarkConfig {
            approx_num_reported: 1000,
            num_points_total: 1_000_000,
        },
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::new("RangeTree2d", config),
            config,
            |b, &config| {
                let mut rng = Pcg64::seed_from_u64(2);
                let builder = RangeTree2dBuilder::from(random_point_collection(
                    &mut rng,
                    config.num_points_total,
                ));
                let storage = builder.alloc_storage();
                let tree = builder.build_in(&storage).unwrap();

                let n = (config.approx_num_reported as f32).sqrt()
                    / (config.num_points_total as f32).sqrt();
                let xy_range = not_nan_range(0.5 - n / 2.0, 0.5 + n / 2.0);

                b.iter(|| tree.query(&xy_range, &xy_range).collect::<Vec<_>>())
            },
        );
        group.bench_with_input(BenchmarkId::new("Naive", config), config, |b, &config| {
            let mut rng = Pcg64::seed_from_u64(2);
            let points = random_point_collection(&mut rng, config.num_points_total);

            let n = (config.approx_num_reported as f32).sqrt()
                / (config.num_points_total as f32).sqrt();
            let xy_range = not_nan_range(0.5 - n / 2.0, 0.5 + n / 2.0);

            b.iter(|| {
                points
                    .iter()
                    .filter(|p| xy_range.contains(&p.0) && xy_range.contains(&p.1))
                    .collect::<Vec<_>>()
            })
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = construction_time, query_time
}

criterion_main!(benches);
