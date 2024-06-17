// A ray tracer in a weekend https://raytracing.github.io/books/RayTracingInOneWeekend.html
// Dragons observed:
// - Zig integer division produces an integer result so take extra care to use float literals when computing a literal division

const std = @import("std");
pub const vec = @import("vector.zig");
const PpmImage = @import("ppm.zig").PpmImage;
const Interval = @import("interval.zig").Interval;

// This is an enum of all the possible geometry that can be hit by rays. It reports on whether a ray hit the geometry.
const Hittable = union(enum) {
    sphere: Sphere,

    const Hit = struct {
        point: vec.Position,
        normal: vec.Direction,
        t: f32,
        frontFace: bool,
        material: Material,
    };

    pub fn ray_hit(self: Hittable, ray: Ray, interval: Interval) ?Hit {
        switch (self) {
            inline else => |s| return s.ray_hit(ray, interval),
        }
    }
};

const Material = union(enum) {
    lambertian: Lambertian,
    metal: Metal,

    const Scatter = struct {
        attenuation: vec.Color,
        ray: Ray,
    };

    pub fn scatter(self: Material, inputRay: Ray, rayHit: Hittable.Hit) ?Scatter {
        switch (self) {
            inline else => |s| return s.scatter(inputRay, rayHit),
        }
    }
};

// The ray struct is responsible for casting rays and producing hits if a
// hittable object is hit by a ray.
const Ray = struct {
    origin: vec.Position,
    direction: vec.Direction,

    pub fn at(self: Ray, t: f32) vec.Position {
        return self.origin + vec.scale(self.direction, t);
    }

    pub fn resolveHit(self: Ray, hittables: []Hittable) ?Hittable.Hit {
        var closestSoFar = std.math.inf(f32);
        var maybeHit: ?Hittable.Hit = null;
        for (hittables) |hittable| {
            // Ignore hits that are too close to the intersection point as it
            // might produce inaccurate results due to floating point error
            if (hittable.ray_hit(self, Interval.of(0.001, closestSoFar))) |h| {
                maybeHit = h;
                closestSoFar = h.t;
            }
        }

        if (maybeHit) |h| {
            return h;
        }

        return null;
    }

    // Here we make an important choice. We calculate the orientation of a face
    // at geometric computation time. This is done because we will have more
    // material types than geometry types.
    //
    /// Called when we have determined that the ray has hit something. This
    /// takes care of calculating face orientation.
    pub fn hit(self: Ray, t: f32, targetOrigin: vec.Position, targetMaterial: Material) Hittable.Hit {
        const intersectionPoint = self.at(t);

        // This calculation always produces an outwards normal
        const normal = vec.normalize(intersectionPoint - targetOrigin);
        return hit_with_normal(self, t, intersectionPoint, normal, targetMaterial);
    }

    /// Produces a hit and calculates the face orientation. The normal parameter MUST be normalized.
    pub fn hit_with_normal(self: Ray, t: f32, intersectionPoint: vec.Position, normal: vec.Direction, targetMaterial: Material) Hittable.Hit {
        std.debug.assert(@abs(vec.magnitude(normal) - 1.0) <= 0.1);

        if (vec.dot(self.direction, normal) > 0.0) {
            // The ray comes from inside the target so flip the normal and mark this as a back face
            return Hittable.Hit{ .normal = -normal, .point = intersectionPoint, .t = t, .frontFace = false, .material = targetMaterial };
        } else {
            // The ray comes from outside the target so keep the normal and mark this as a front face
            return Hittable.Hit{ .normal = normal, .point = intersectionPoint, .t = t, .frontFace = true, .material = targetMaterial };
        }
    }
};

const Sphere = struct {
    origin: vec.Position,
    radius: f32,
    material: Material,

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
        return ray.hit_with_normal(t, p, outwardNormal, self.material);
    }
};

// Always scatter Lambertian reflectance
const Lambertian = struct {
    rng: std.Random.Random,
    albedo: vec.Color,

    // The way we reflect rays is by creating a random unit vector and
    // adding it to the normal. This essentially means that we are
    // picking a point on the unit sphere that is tangent to the hit
    // point. By doing so we are approximating Lambertian reflectance.
    pub fn scatter(self: Lambertian, inputRay: Ray, rayHit: Hittable.Hit) ?Material.Scatter {
        _ = inputRay;

        var scatterDirection = rayHit.normal + vec.random_on_unit_sphere(self.rng);

        // There is a possibility that we generate a random vector that when
        // added to the normal produces a near zero result which is problematic
        // for calculations down the line so guard that.
        if (vec.near_zero(scatterDirection)) {
            scatterDirection = rayHit.normal;
        }

        const scatteredRay = Ray{ .origin = rayHit.point, .direction = scatterDirection };
        return Material.Scatter{
            .attenuation = self.albedo,
            .ray = scatteredRay,
        };
    }
};

const Metal = struct {
    rng: std.Random.Random,
    albedo: vec.Color,
    fuzz: f32,

    pub fn scatter(self: Metal, inputRay: Ray, rayHit: Hittable.Hit) ?Material.Scatter {
        var reflectionDirection = vec.reflect(inputRay.direction, rayHit.normal);

        // Add some fuzz to the metal to produce less crisp reflections.
        reflectionDirection = vec.normalize(reflectionDirection) + vec.scale(vec.random_on_unit_sphere(self.rng), self.fuzz);

        const scatteredRay = Ray{ .origin = rayHit.point, .direction = reflectionDirection };
        if (vec.dot(reflectionDirection, rayHit.normal) > 0.0) {
            return Material.Scatter{
                .ray = scatteredRay,
                .attenuation = self.albedo,
            };
        } else {
            return null;
        }
    }
};

// The camera struct is responsible for casting rays, coloring said rays based
// on whether they hit something and performing pre and post processing. (The
// name is kinda incorrect because it does not concern itself just with camera
// stuff)
const Camera = struct {
    const reflectionMaxDepth = 10;
    rng: std.Random.Random,

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

    samplesPerPixel: u32,
    pixelSamplesScale: f32,

    const Options = struct {
        aspectRatio: f32 = 16.0 / 9.0,
        focalLength: f32 = 1.0,
        viewportHeight: f32 = 2.0,
        imageWidthPixels: u32,
        samplesPerPixel: u32 = 10,
        rng: std.Random.Random,
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
            .rng = options.rng,
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
            .samplesPerPixel = options.samplesPerPixel,
            .pixelSamplesScale = 1.0 / @as(f32, @floatFromInt(options.samplesPerPixel)),
        };
    }

    fn linearToGamma(linear: f32) f32 {
        if (linear > 0.0) {
            return @sqrt(linear);
        }
        return 0.0;
    }

    pub fn render(self: Camera, hittables: []Hittable, framebuffer: *std.ArrayList(vec.Color)) !void {
        var h: u32 = 0;
        while (h < self.imageHeightPixels) : (h += 1) {
            var w: u32 = 0;
            while (w < self.imageWidthPixels) : (w += 1) {

                // Simple supersampling anti-aliasing. This blows up rendering
                // times. It works by sampling some random pixels around the
                // current pixel and blending (adding) their colors
                var pixelColor: vec.Color = @splat(0.0);
                var sample: u32 = 0;
                while (sample < self.samplesPerPixel) : (sample += 1) {
                    const r = self.getRandomRayAt(w, h);
                    pixelColor += self.rayColor(0, r, hittables);
                }

                // Do the gamma correction before appending to the framebuffer
                pixelColor = vec.scale(pixelColor, self.pixelSamplesScale);
                try framebuffer.append(vec.Color{ linearToGamma(pixelColor[0]), linearToGamma(pixelColor[1]), linearToGamma(pixelColor[2]) });

                // No antialiasing
                // ---------------
                // const wf = @as(f32, @floatFromInt(w));
                // const hf = @as(f32, @floatFromInt(h));
                // const pixelCenter = self.firstPixelPosition + vec.scale(self.pixelDeltaRight, wf) + vec.scale(self.pixelDeltaDown, hf);
                // const ray = Ray{ .direction = pixelCenter - self.origin, .origin = self.origin };
                // try framebuffer.append(ray.color(hittables));
            }
        }
    }

    fn rayColor(self: Camera, depth: u32, ray: Ray, hittables: []Hittable) vec.Color {
        if (depth >= reflectionMaxDepth) {
            return @splat(0.0);
        }

        const maybeHit = ray.resolveHit(hittables);
        if (maybeHit) |hit| {
            const maybeScatter = hit.material.scatter(ray, hit);
            if (maybeScatter) |scatter| {
                return scatter.attenuation * self.rayColor(depth + 1, scatter.ray, hittables);
            } else {
                return @splat(0.0);
            }
        }

        const unitDirection = vec.normalize(ray.direction);
        // the unit vector produced here will have components in range [-1, 1].
        // We want components to be in [0, 1] so we:
        // - +1 it so the range becomes [0, 2]
        // - /2 it so the range becomes [0, 1]
        const a = 0.5 * (unitDirection[vec.Y] + 1.0);
        return vec.lerp(vec.Color{ 1.0, 1.0, 1.0 }, vec.Color{ 0.5, 0.7, 1.0 }, a);
    }

    fn getRandomRayAt(self: Camera, i: u32, j: u32) Ray {
        const offset = self.sampleSquare();
        const fi: f32 = @floatFromInt(i);
        const fj: f32 = @floatFromInt(j);
        const pixelSample = self.firstPixelPosition + vec.scale(self.pixelDeltaRight, fi + offset[0]) + vec.scale(self.pixelDeltaDown, fj + offset[1]);
        return Ray{ .direction = pixelSample - self.origin, .origin = self.origin };
    }

    fn sampleSquare(self: Camera) vec.Position {
        const randX = self.rng.float(f32) - 0.5;
        const randY = self.rng.float(f32) - 0.5;

        return vec.Position{ randX - 0.5, randY - 0.5, 0.0 };
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const seed: u64 = @intCast(std.time.milliTimestamp());
    std.log.info("seed: {}", .{seed});
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    const camera = Camera.init(.{ .rng = rng, .imageWidthPixels = 860 });
    var ppm_image = PpmImage.init(camera.imageWidthPixels, camera.imageHeightPixels);

    const groundMaterial = Material{ .lambertian = Lambertian{ .albedo = vec.Color{ 0.8, 0.8, 0.0 }, .rng = rng } };
    const centerMaterial = Material{ .lambertian = Lambertian{ .albedo = vec.Color{ 0.1, 0.2, 0.5 }, .rng = rng } };
    const leftMaterial = Material{ .metal = Metal{ .albedo = vec.Color{ 0.8, 0.8, 0.8 }, .fuzz = 0.3, .rng = rng } };
    const rightMaterial = Material{ .metal = Metal{ .albedo = vec.Color{ 0.8, 0.6, 0.2 }, .fuzz = 1.0, .rng = rng } };

    var hittables = std.ArrayList(Hittable).init(arena.allocator());

    try hittables.append(Hittable{ .sphere = Sphere{ .origin = vec.Position{ 0.0, -100.5, -1.0 }, .radius = 100.0, .material = groundMaterial } });
    try hittables.append(Hittable{ .sphere = Sphere{ .origin = vec.Position{ 0.0, 0.0, -1.2 }, .radius = 0.5, .material = centerMaterial } });
    try hittables.append(Hittable{ .sphere = Sphere{ .origin = vec.Position{ -1.0, 0.0, -1.0 }, .radius = 0.5, .material = leftMaterial } });
    try hittables.append(Hittable{ .sphere = Sphere{ .origin = vec.Position{ 1.0, 0.0, -1.0 }, .radius = 0.5, .material = rightMaterial } });

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
