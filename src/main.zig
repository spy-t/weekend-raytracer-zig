// A ray tracer in a weekend https://raytracing.github.io/books/RayTracingInOneWeekend.html
// Dragons observed:
// - Zig integer division produces an integer result so take extra care to use float literals when computing a literal division

const std = @import("std");
pub const vec = @import("vector.zig");
const PpmImage = @import("ppm.zig").PpmImage;

const Hittable = union(enum) {
    sphere: Sphere,

    const Hit = struct {
        point: vec.Position,
        normal: vec.Direction,
        t: f32,
        frontFace: bool,
    };

    pub fn ray_hit(self: Hittable, ray: Ray, interval: Interval) ?Hit {
        switch (self) {
            inline else => |s| return s.ray_hit(ray, interval),
        }
    }
};

const Ray = struct {
    origin: vec.Position,
    direction: vec.Direction,

    pub fn at(self: Ray, t: f32) vec.Position {
        return self.origin + vec.scale(self.direction, t);
    }

    pub fn color(
        self: Ray,
        hittables: []Hittable,
    ) vec.Color {
        var closestSoFar = std.math.inf(f32);
        var maybeHit: ?Hittable.Hit = null;
        for (hittables) |hittable| {
            if (hittable.ray_hit(self, Interval.of(0.0, closestSoFar))) |h| {
                maybeHit = h;
                closestSoFar = h.t;
            }
        }

        if (maybeHit) |h| {
            return vec.scale(h.normal + @as(vec.Color, @splat(1.0)), 0.5);
        }

        const unitDirection = vec.normalize(self.direction);
        // the unit vector produced here will have components in range [-1, 1].
        // We want components to be in [0, 1] so we:
        // - +1 it so the range becomes [0, 2]
        // - /2 it so the range becomes [0, 1]
        const a = 0.5 * (unitDirection[vec.Y] + 1.0);
        return vec.lerp(vec.Color{ 1.0, 1.0, 1.0 }, vec.Color{ 0.5, 0.7, 1.0 }, a);
    }

    // Here we make an important choice. We calculate the orientation of a face
    // at geometric computation time. This is done because we will have more
    // material types than geometry types.
    //
    /// Called when we have determined that the ray has hit something. This
    /// takes care of calculating face orientation.
    pub fn hit(self: Ray, t: f32, targetOrigin: vec.Position) Hittable.Hit {
        const intersectionPoint = self.at(t);

        // This calculation always produces an outwards normal
        const normal = vec.normalize(intersectionPoint - targetOrigin);
        return hit_with_normal(self, t, intersectionPoint, normal);
    }

    /// Produces a hit and calculates the face orientation. The normal parameter MUST be normalized.
    pub fn hit_with_normal(self: Ray, t: f32, intersectionPoint: vec.Position, normal: vec.Direction) Hittable.Hit {
        std.debug.assert(@abs(vec.magnitude(normal) - 1.0) <= 0.1);

        if (vec.dot(self.direction, normal) > 0.0) {
            // The ray comes from inside the target so flip the normal and mark this as a back face
            return Hittable.Hit{ .normal = -normal, .point = intersectionPoint, .t = t, .frontFace = false };
        } else {
            // The ray comes from outside the target so keep the normal and mark this as a front face
            return Hittable.Hit{ .normal = normal, .point = intersectionPoint, .t = t, .frontFace = true };
        }
    }
};

const Sphere = struct {
    origin: vec.Position,
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
    pub fn ray_hit(self: Sphere, ray: Ray, interval: Interval) ?Hittable.Hit {
        // This actually does some further simplification to reduce the number of calculations.
        // - The dot product of a vector with itself is the squared magnitude.
        // - Setting b = -2h simplifies some disciminant calculations

        // oc = (C - Q)
        const oc = self.origin - ray.origin;
        const ocMag = vec.magnitude(oc);

        const rayDirectionMag = vec.magnitude(ray.direction);
        const a = rayDirectionMag * rayDirectionMag;

        const h = vec.dot(ray.direction, oc);
        const c = ocMag * ocMag - self.radius * self.radius;

        const discriminant = h * h - a * c;
        if (discriminant < 0) {
            return null;
        }

        const dsqrt = @sqrt(discriminant);
        var t = (h - dsqrt) / a;
        if (!interval.surrounds(t)) {
            // Maybe the other root is in bounds?
            t = (h + dsqrt) / a;
            if (!interval.surrounds(t)) {
                return null;
            }
        }

        const p = ray.at(t);
        const outwardNormal = vec.scale(p - self.origin, 1.0 / self.radius);
        return ray.hit_with_normal(t, p, outwardNormal);
    }
};

const Interval = struct {
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
};

const Camera = struct {
    origin: vec.Position,
    focalLength: f32,

    aspectRatio: f32,
    imageWidthPixels: u32,
    imageHeightPixels: u32,

    viewportHeight: f32,
    viewportWidth: f32,

    viewportRight: vec.Direction,
    viewportDown: vec.Direction,
    viewportUpperLeft: vec.Position,

    pixelDeltaRight: vec.Direction,
    pixelDeltaDown: vec.Direction,
    firstPixelPosition: vec.Position,

    const Options = struct {
        aspectRatio: f32 = 16.0 / 9.0,
        focalLength: f32 = 1.0,
        viewportHeight: f32 = 2.0,
        imageWidthPixels: u32,
    };

    pub fn init(options: Options) Camera {
        const origin: vec.Position = @splat(0.0);

        const aspectRatio = options.aspectRatio;
        const focalLength = options.focalLength;
        const viewportHeight = options.viewportHeight;
        const imageWidthPixels = options.imageWidthPixels;

        const wf = @as(f32, @floatFromInt(options.imageWidthPixels));

        const hf = height: {
            const h = wf / options.aspectRatio;
            if (h < 1) {
                break :height 1;
            } else {
                break :height h;
            }
        };
        const imageHeightPixels = @as(u32, @intFromFloat(hf));

        const viewportWidth = viewportHeight * (wf / hf);

        const viewportRight = vec.Direction{ viewportWidth, 0.0, 0.0 };
        const viewportDown = vec.Direction{ 0.0, -viewportHeight, 0.0 };

        const pixelDeltaRight = vec.scale(viewportRight, 1.0 / wf);
        const pixelDeltaDown = vec.scale(viewportDown, 1.0 / hf);

        const viewportUpperLeft = origin - vec.Direction{ 0.0, 0.0, focalLength } - vec.scale(viewportRight, 1.0 / 2.0) - vec.scale(viewportDown, 1.0 / 2.0);
        const firstPixelPosition = viewportUpperLeft + vec.scale(pixelDeltaRight + pixelDeltaDown, 0.5);

        return Camera{
            .origin = origin,
            .focalLength = focalLength,
            .aspectRatio = aspectRatio,
            .imageWidthPixels = imageWidthPixels,
            .imageHeightPixels = imageHeightPixels,
            .viewportHeight = viewportHeight,
            .viewportWidth = viewportWidth,
            .viewportRight = viewportRight,
            .viewportDown = viewportDown,
            .pixelDeltaRight = pixelDeltaRight,
            .pixelDeltaDown = pixelDeltaDown,
            .viewportUpperLeft = viewportUpperLeft,
            .firstPixelPosition = firstPixelPosition,
        };
    }

    pub fn render(self: Camera, hittables: []Hittable, framebuffer: *std.ArrayList(vec.Color)) !void {
        for (0..self.imageHeightPixels) |h| {
            for (0..self.imageWidthPixels) |w| {
                const wf = @as(f32, @floatFromInt(w));
                const hf = @as(f32, @floatFromInt(h));

                const pixelCenter = self.firstPixelPosition + vec.scale(self.pixelDeltaRight, wf) + vec.scale(self.pixelDeltaDown, hf);
                const ray = Ray{ .direction = pixelCenter - self.origin, .origin = self.origin };

                try framebuffer.append(ray.color(hittables));
            }
        }
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const camera = Camera.init(.{
        .imageWidthPixels = 860,
    });
    var ppm_image = PpmImage.init(camera.imageWidthPixels, camera.imageHeightPixels);

    var hittables = std.ArrayList(Hittable).init(arena.allocator());
    try hittables.append(Hittable{ .sphere = Sphere{ .origin = vec.Position{ 0.0, -100.5, -1.0 }, .radius = 100.0 } });
    try hittables.append(Hittable{ .sphere = Sphere{ .origin = vec.Position{ 0.0, 0.0, -1.0 }, .radius = 0.5 } });

    var framebuffer = std.ArrayList(vec.Color).init(arena.allocator());

    const renderStartMs = std.time.milliTimestamp();
    try camera.render(hittables.items, &framebuffer);
    const renderEndMs = std.time.milliTimestamp();

    const cwd = std.fs.cwd();
    var image_file = try cwd.createFile("image.ppm", .{});

    const writeStartMs = std.time.milliTimestamp();
    try ppm_image.writeTo(image_file.writer(), framebuffer.items);
    const writeEndMs = std.time.milliTimestamp();

    std.log.info("rendered in: {}ms", .{renderEndMs - renderStartMs});
    std.log.info("written in: {}ms", .{writeEndMs - writeStartMs});
    std.log.info("image dimensions: {}x{}", .{ ppm_image.width, ppm_image.height });
    std.log.debug("framebuffer size: {}", .{framebuffer.items.len});
}

test {
    std.testing.refAllDecls(@This());
}
