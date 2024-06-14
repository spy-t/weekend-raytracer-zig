// A ray tracer in a weekend https://raytracing.github.io/books/RayTracingInOneWeekend.html
// Dragons observed:
// - Zig integer division produces an integer result so take extra care to use float literals when computing a literal division

const std = @import("std");
const Allocator = std.mem.Allocator;
const heap = std.heap;

const X = 0;
const Y = 1;
const Z = 2;
const Color = @Vector(3, f32);
const Position = Color;
const Direction = Position;

fn dot(a: @Vector(3, f32), b: @Vector(3, f32)) f32 {
    return @reduce(.Add, a * b);
}

fn cross(a: @Vector(3, f32), b: @Vector(3, f32)) @Vector(3, f32) {
    return @Vector(3, f32){
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn magnitude(a: @Vector(3, f32)) f32 {
    return std.math.sqrt(@reduce(.Add, a * a));
}

fn normalize(a: @Vector(3, f32)) @Vector(3, f32) {
    return scale(a, 1 / magnitude(a));
}

fn scale(a: @Vector(3, f32), factor: f32) @Vector(3, f32) {
    return a * @as(@Vector(3, f32), @splat(factor));
}

fn equal(a: @Vector(3, f32), b: @Vector(3, f32)) bool {
    return @reduce(.And, a == b);
}

fn lerp(start: @Vector(3, f32), end: @Vector(3, f32), a: f32) @Vector(3, f32) {
    std.debug.assert(a >= 0 and a <= 1);
    return scale(start, 1.0 - a) + scale(end, a);
}

const PpmImage = struct {
    buffer: std.ArrayList(u8),
    width: u32,
    height: u32,
    current_row: u32 = 0,
    current_column: u32 = 0,

    const RGBColor = struct {
        r: u8,
        g: u8,
        b: u8,
        pub fn fromVector(v: Color) @This() {
            return .{ .r = @intFromFloat(255.999 * v[0]), .g = @intFromFloat(255.999 * v[1]), .b = @intFromFloat(255.999 * v[2]) };
        }
    };

    fn init(allocator: Allocator, width: u32, height: u32) PpmImage {
        return PpmImage{ .buffer = std.ArrayList(u8).init(allocator), .width = width, .height = height };
    }

    fn pushPixel(self: *PpmImage, color: RGBColor) !void {
        const w = self.buffer.writer();
        if (self.current_column == self.width) {
            if (self.current_row == self.height) {
                return error.ImageFull;
            }
            self.current_column = 0;
            self.current_row += 1;
            try w.writeByte('\n');
        }
        try std.fmt.format(self.buffer.writer(), " {} {} {}", .{ color.r, color.g, color.b });
        self.current_column += 1;
    }

    fn writeTo(self: *PpmImage, writer: anytype) !void {
        // First write the header
        try std.fmt.format(writer, "P3\n{} {}\n255\n", .{ self.width, self.height });

        // Then write the pixel values
        try writer.writeAll(self.buffer.items);
    }
};

const Hittable = union(enum) {
    sphere: *Sphere,

    const Hit = struct {
        point: Position,
        normal: Direction,
        t: f32,
        frontFace: bool,
    };

    pub fn ray_hit(self: Hittable, ray: Ray, tmin: f32, tmax: f32) ?Hit {
        switch (self) {
            inline else => |s| return s.ray_hit(ray, tmin, tmax),
        }
    }
};

const Ray = struct {
    origin: Position,
    direction: Direction,

    pub fn at(self: Ray, t: f32) Position {
        return self.origin + scale(self.direction, t);
    }

    pub fn color(self: Ray) Color {
        const s = Sphere{ .origin = Position{ 0.0, 0.0, -1.0 }, .radius = 0.5 };
        if (s.ray_hit(self, 0.0, std.math.inf(f32))) |h| {
            return scale(h.normal + @as(Color, @splat(1.0)), 0.5);
        }

        const unitDirection = normalize(self.direction);
        // the unit vector produced here will have components in range [-1, 1].
        // We want components to be in [0, 1] so we:
        // - +1 it so the range becomes [0, 2]
        // - /2 it so the range becomes [0, 1]
        const a = 0.5 * (unitDirection[Y] + 1.0);
        return lerp(Color{ 1.0, 1.0, 1.0 }, Color{ 0.5, 0.7, 1.0 }, a);
    }

    // TODO(spyros): This might need rework because it always normalizes the
    // normal and it might be redundant for some shapes. Otherwise we could
    // expose a "I know what I do version" that such shapes could use.
    //
    // Here we make an important choice. We calculate the orientation of a face
    // at geometric computation time. This is done because we will have more
    // material types than geometry types.
    //
    /// Called when we have determined that the ray has hit something. This
    /// takes care of calculating face orientation.
    pub fn hit(self: Ray, t: f32, targetOrigin: Position) Hittable.Hit {
        const intersectionPoint = self.at(t);

        // This calculation always produces an outwards normal
        const normal = normalize(intersectionPoint - targetOrigin);
        if (dot(self.direction, normal) > 0.0) {
            // The ray comes from inside the target so flip the normal and mark this as a back face
            return Hittable.Hit{ .normal = -normal, .point = intersectionPoint, .t = t, .frontFace = false };
        } else {
            // The ray comes from outside the target so keep the normal and mark this as a front face
            return Hittable.Hit{ .normal = normal, .point = intersectionPoint, .t = t, .frontFace = true };
        }
    }
};

const Sphere = struct {
    origin: Position,
    radius: f32,

    // A sphere equation is x^2 + y^2 + z^2 = r^2.
    // If it is centered at C this becomes (Cx - x)^2 + (Cy - y)^2 + (Cz - z)^2 = r^2
    //
    // To test if a ray hits the sphere we need to find the roots of the following equation:
    // (C - P(t)) * (C - P(t)) = r^2 => (C - (Q + td)) * (C - (Q + td)) = r^2
    // t is the variable of that equation which after applying some properties expands to:
    // t^2 * d - 2td * (C - Q) + (C - Q) * (C - Q) - r^2 = 0
    // which is a regular quadratic equation.
    //
    // If this equation has no roots
    // then the ray does not intersect the sphere at any point. If it has 1
    // root it is perpendicular to the sphere at the root and if it has 2 roots
    // it intersects the sphere at the 2 roots. Thus finding the discriminant
    // is enough to answer if a ray hits the sphere.
    pub fn ray_hit(self: Sphere, ray: Ray, tmin: f32, tmax: f32) ?Hittable.Hit {
        // This actually does some further simplification to reduce the number of calculations.
        // - The dot product of a vector with itself is the squared magnitude.
        // - Setting b = -2h simplifies some disciminant calculations

        // oc = (C - Q)
        const oc = self.origin - ray.origin;
        const ocMag = magnitude(oc);

        const rayDirectionMag = magnitude(ray.direction);
        const a = rayDirectionMag * rayDirectionMag;

        const h = dot(ray.direction, oc);
        const c = ocMag * ocMag - self.radius * self.radius;

        const discriminant = h * h - a * c;
        if (discriminant < 0) {
            return null;
        }

        const dsqrt = @sqrt(discriminant);
        var t = (h - dsqrt) / a;
        if (t <= tmin or t >= tmax) {
            // Maybe the other root is in bounds?
            t = (h + dsqrt) / a;
            if (t <= tmin or t >= tmax) {
                return null;
            }
        }

        return ray.hit(t, self.origin);
    }
};

pub fn main() !void {
    var arena = heap.ArenaAllocator.init(heap.page_allocator);
    defer arena.deinit();

    // Image
    const width = 860.0;
    const aspectRatio = 16.0 / 9.0;
    const height = height: {
        const h = width / aspectRatio;
        if (h < 1) {
            break :height 1;
        } else {
            break :height h;
        }
    };

    // Camera
    const focalLength = 1.0;
    const viewportHeight = 2.0;
    const viewportWidth = viewportHeight * (width / height);
    const cameraCenter: Position = @splat(0.0);

    // Viewport
    const viewportRight = Direction{ viewportWidth, 0.0, 0.0 };
    const viewportDown = Direction{ 0.0, -viewportHeight, 0.0 };

    const pixelDeltaRight = scale(viewportRight, 1.0 / width);
    const pixelDeltaDown = scale(viewportDown, 1.0 / height);

    const viewportUpperLeft = cameraCenter - Direction{ 0.0, 0.0, focalLength } - scale(viewportRight, 1.0 / 2.0) - scale(viewportDown, 1.0 / 2.0);
    const firstPixelPosition = viewportUpperLeft + scale(pixelDeltaRight + pixelDeltaDown, 0.5);

    var ppm_image = PpmImage.init(arena.allocator(), @intFromFloat(width), @intFromFloat(height));

    for (0..@intFromFloat(height)) |h| {
        for (0..@intFromFloat(width)) |w| {
            const wf = @as(f32, @floatFromInt(w));
            const hf = @as(f32, @floatFromInt(h));

            const pixelCenter = firstPixelPosition + scale(pixelDeltaRight, wf) + scale(pixelDeltaDown, hf);
            const ray = Ray{ .direction = pixelCenter - cameraCenter, .origin = cameraCenter };

            try ppm_image.pushPixel(PpmImage.RGBColor.fromVector(ray.color()));
        }
    }

    const cwd = std.fs.cwd();
    var image_file = try cwd.createFile("image.ppm", .{});
    try ppm_image.writeTo(image_file.writer());
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
