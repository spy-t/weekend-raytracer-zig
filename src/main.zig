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

    pub fn ray_hit(self: Hittable, ray: Ray, tmin: f32, tmax: f32) ?Hit {
        switch (self) {
            inline else => |s| return s.ray_hit(ray, tmin, tmax),
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
        for (hittables) |hittable| {
            if (hittable.ray_hit(self, 0.0, std.math.inf(f32))) |h| {
                return vec.scale(h.normal + @as(vec.Color, @splat(1.0)), 0.5);
            }
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
    pub fn ray_hit(self: Sphere, ray: Ray, tmin: f32, tmax: f32) ?Hittable.Hit {
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
        if (t <= tmin or t >= tmax) {
            // Maybe the other root is in bounds?
            t = (h + dsqrt) / a;
            if (t <= tmin or t >= tmax) {
                return null;
            }
        }

        const p = ray.at(t);
        const outwardNormal = vec.scale(p - self.origin, 1.0 / self.radius);
        return ray.hit_with_normal(t, p, outwardNormal);
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
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
    const cameraCenter: vec.Position = @splat(0.0);

    // Viewport
    const viewportRight = vec.Direction{ viewportWidth, 0.0, 0.0 };
    const viewportDown = vec.Direction{ 0.0, -viewportHeight, 0.0 };

    const pixelDeltaRight = vec.scale(viewportRight, 1.0 / width);
    const pixelDeltaDown = vec.scale(viewportDown, 1.0 / height);

    const viewportUpperLeft = cameraCenter - vec.Direction{ 0.0, 0.0, focalLength } - vec.scale(viewportRight, 1.0 / 2.0) - vec.scale(viewportDown, 1.0 / 2.0);
    const firstPixelPosition = viewportUpperLeft + vec.scale(pixelDeltaRight + pixelDeltaDown, 0.5);

    var ppm_image = PpmImage.init(arena.allocator(), @intFromFloat(width), @intFromFloat(height));

    var hittables = std.ArrayList(Hittable).init(arena.allocator());
    try hittables.append(Hittable{ .sphere = Sphere{ .origin = vec.Position{ 0.0, 0.0, -1.0 }, .radius = 0.5 } });
    try hittables.append(Hittable{ .sphere = Sphere{ .origin = vec.Position{ 0.0, -100.5, -1.0 }, .radius = 100.0 } });

    for (0..@intFromFloat(height)) |h| {
        for (0..@intFromFloat(width)) |w| {
            const wf = @as(f32, @floatFromInt(w));
            const hf = @as(f32, @floatFromInt(h));

            const pixelCenter = firstPixelPosition + vec.scale(pixelDeltaRight, wf) + vec.scale(pixelDeltaDown, hf);
            const ray = Ray{ .direction = pixelCenter - cameraCenter, .origin = cameraCenter };

            try ppm_image.pushPixel(PpmImage.RGBColor.fromVector(ray.color(hittables.items)));
        }
    }

    const cwd = std.fs.cwd();
    var image_file = try cwd.createFile("image.ppm", .{});
    try ppm_image.writeTo(image_file.writer());
}

test {
    std.testing.refAllDecls(@This());
}
