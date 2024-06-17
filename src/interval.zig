const std = @import("std");

pub const Interval = struct {
    min: f32,
    max: f32,

    pub const EMPTY = Interval{
        .min = std.math.inf(f32),
        .max = -std.math.inf(f32),
    };

    pub const UNIVERSE = Interval{
        .min = -std.math.inf(f32),
        .max = std.math.inf(f32),
    };

    pub fn of(min: f32, max: f32) Interval {
        return .{ .min = min, .max = max };
    }

    pub fn size(self: Interval) f32 {
        self.max - self.min;
    }

    pub fn contains(self: Interval, x: f32) bool {
        return self.min <= x and x <= self.max;
    }

    pub fn surrounds(self: Interval, x: f32) bool {
        return self.min < x and x < self.max;
    }

    pub fn clamp(self: Interval, x: f32) f32 {
        if (x < self.min) {
            return self.min;
        }
        if (x > self.max) {
            return self.max;
        }

        return x;
    }
};
