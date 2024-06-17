const std = @import("std");
const Allocator = std.mem.Allocator;
const Color = @import("vector.zig").Color;
const Interval = @import("interval.zig").Interval;

pub const PpmImage = struct {
    width: u32,
    height: u32,

    pub const RGBColor = struct {
        r: u8,
        g: u8,
        b: u8,
        pub fn fromVector(v: Color) @This() {
            return .{ .r = @intFromFloat(255.999 * v[0]), .g = @intFromFloat(255.999 * v[1]), .b = @intFromFloat(255.999 * v[2]) };
        }
    };

    pub fn init(width: u32, height: u32) PpmImage {
        return PpmImage{ .width = width, .height = height };
    }

    pub fn pushPixel(self: *PpmImage, color: RGBColor) !void {
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

    pub fn writeTo(self: PpmImage, writer: anytype, data: []Color) !void {
        std.debug.assert(data.len == self.width * self.height);

        var bf = std.io.bufferedWriter(writer);
        const bfw = bf.writer();

        // First write the header
        try std.fmt.format(bfw, "P3\n{} {}\n255\n", .{ self.width, self.height });

        const intensity = Interval{ .min = 0.0, .max = 0.999 };
        for (data) |color| {
            const r: u32 = @intFromFloat(256.0 * intensity.clamp(color[0]));
            const g: u32 = @intFromFloat(256.0 * intensity.clamp(color[1]));
            const b: u32 = @intFromFloat(256.0 * intensity.clamp(color[2]));
            try std.fmt.format(bfw, "{} {} {}\n", .{ r, g, b });
        }

        try bf.flush();
    }
};
