const std = @import("std");

pub fn random_in(rng: std.Random.Random, min: f32, max: f32) f32 {
    return min + (max - min) * rng.float(f32);
}
