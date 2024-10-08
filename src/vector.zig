const std = @import("std");
const rand = @import("random.zig");

pub const X = 0;
pub const Y = 1;
pub const Z = 2;
pub const Color = @Vector(3, f32);
pub const Position = Color;
pub const Direction = Position;

pub fn dot(a: @Vector(3, f32), b: @Vector(3, f32)) f32 {
    return @reduce(.Add, a * b);
}

pub fn cross(a: @Vector(3, f32), b: @Vector(3, f32)) @Vector(3, f32) {
    return @Vector(3, f32){
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

pub fn magnitude(a: @Vector(3, f32)) f32 {
    return std.math.sqrt(@reduce(.Add, a * a));
}

pub fn normalize(a: @Vector(3, f32)) @Vector(3, f32) {
    return scale(a, 1 / magnitude(a));
}

pub fn scale(a: @Vector(3, f32), factor: f32) @Vector(3, f32) {
    return a * @as(@Vector(3, f32), @splat(factor));
}

pub fn equal(a: @Vector(3, f32), b: @Vector(3, f32)) bool {
    return @reduce(.And, a == b);
}

pub fn lerp(start: @Vector(3, f32), end: @Vector(3, f32), a: f32) @Vector(3, f32) {
    std.debug.assert(a >= 0 and a <= 1);
    return scale(start, 1.0 - a) + scale(end, a);
}

pub fn random(rng: std.Random.Random) @Vector(3, f32) {
    return @Vector(3, f32){
        rng.float(f32),
        rng.float(f32),
        rng.float(f32),
    };
}

pub fn random_in(rng: std.Random.Random, min: f32, max: f32) @Vector(3, f32) {
    return @Vector(3, f32){
        rand.random_in(rng, min, max),
        rand.random_in(rng, min, max),
        rand.random_in(rng, min, max),
    };
}

pub fn random_on_unit_sphere(rng: std.Random.Random) @Vector(3, f32) {
    while (true) {
        const v = random_in(rng, -1.0, 1.0);
        const mag = magnitude(v);
        if (mag * mag <= 1) {
            return normalize(v);
        }
    }
}

pub fn random_on_hemisphere(rng: std.Random.Random, normal: Direction) @Vector(3, f32) {
    const v = random_on_unit_sphere(rng);
    if (dot(v, normal) > 0.0) {
        return v;
    } else {
        return -v;
    }
}

pub fn near_zero(vec: @Vector(3, f32)) bool {
    const e: @Vector(3, f32) = @splat(1e-8);
    return @reduce(.And, @abs(vec) < e);
}

pub fn reflect(a: @Vector(3, f32), b: @Vector(3, f32)) @Vector(3, f32) {
    return a - scale(b, 2 * dot(a, b));
}

test "vector scale works" {
    const v = Position{ -2.0, 4.0, -4.0 };
    const res = scale(v, 2.0);
    try std.testing.expect(equal(res, Position{ -4.0, 8.0, -8.0 }));
}

test "vector magnitude works" {
    const v = Position{ -2.0, 4.0, -4.0 };
    const mag = magnitude(v);
    try std.testing.expect(mag == 6);
}

test "vector dot product works" {
    const a = Position{ 1.0, 3.0, -5.0 };
    const b = Position{ 4.0, -2.0, -1.0 };
    const c = dot(a, b);
    try std.testing.expect(c == 3.0);
}

test "vector cross product works" {
    const a = Position{ 1.0, 3.0, -5.0 };
    const b = Position{ 4.0, -2.0, -1.0 };
    const c = cross(a, b);
    try std.testing.expect(equal(c, Position{ -13.0, -19.0, -14.0 }));
}

test "vector normalization works" {
    const a = Position{ -2.0, 4.0, -4.0 };
    const aNorm = normalize(a);

    try std.testing.expect(equal(aNorm, Position{ -1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0 }));
}

test "vector normalization on a normalized vector is identity" {
    const a = Position{ -2.0, 4.0, -4.0 };
    const aNorm = normalize(a);
    const normedAgain = normalize(aNorm);

    try std.testing.expect(equal(aNorm, normedAgain));
}
