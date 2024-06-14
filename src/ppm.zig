const std = @import("std");
const Allocator = std.mem.Allocator;
const Color = @import("vector.zig").Color;

pub const PpmImage = struct {
    buffer: std.ArrayList(u8),
    width: u32,
    height: u32,
    current_row: u32 = 0,
    current_column: u32 = 0,

    pub const RGBColor = struct {
        r: u8,
        g: u8,
        b: u8,
        pub fn fromVector(v: Color) @This() {
            return .{ .r = @intFromFloat(255.999 * v[0]), .g = @intFromFloat(255.999 * v[1]), .b = @intFromFloat(255.999 * v[2]) };
        }
    };

    pub fn init(allocator: Allocator, width: u32, height: u32) PpmImage {
        return PpmImage{ .buffer = std.ArrayList(u8).init(allocator), .width = width, .height = height };
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

    pub fn writeTo(self: *PpmImage, writer: anytype) !void {
        // First write the header
        try std.fmt.format(writer, "P3\n{} {}\n255\n", .{ self.width, self.height });

        // Then write the pixel values
        try writer.writeAll(self.buffer.items);
    }
};
