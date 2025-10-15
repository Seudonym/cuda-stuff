import cv2
import numpy as np
import sys


class SquareSelector:
    def __init__(
        self, image_path, x_min=-2.16667, x_max=1.16667, y_min=-1.66, y_max=1.66
    ):
        # Load PGM image
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.height, self.width = self.img.shape
        self.display_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        # Complex plane bounds
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.end_point = None

    def pixel_to_complex(self, px, py):
        """Convert pixel coordinates to complex plane coordinates"""
        # Flip y-axis (image origin is top-left, complex plane origin is bottom-left)
        cx = self.x_min + (px / self.width) * (self.x_max - self.x_min)
        cy = self.y_max - (py / self.height) * (self.y_max - self.y_min)
        return cx, cy

    def draw_rectangle(self, event, x, y, flags, param):
        """Mouse callback function"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Calculate side length (use maximum of width/height difference)
                dx = x - self.start_point[0]
                dy = y - self.start_point[1]
                side = max(abs(dx), abs(dy))

                # Make it a square, maintaining the direction
                end_x = self.start_point[0] + (side if dx >= 0 else -side)
                end_y = self.start_point[1] + (side if dy >= 0 else -side)
                self.end_point = (end_x, end_y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Recalculate to ensure it's a perfect square
            dx = x - self.start_point[0]
            dy = y - self.start_point[1]
            side = max(abs(dx), abs(dy))

            end_x = self.start_point[0] + (side if dx >= 0 else -side)
            end_y = self.start_point[1] + (side if dy >= 0 else -side)
            self.end_point = (end_x, end_y)

            # Calculate complex plane coordinates
            x0, y0 = self.start_point
            x1, y1 = self.end_point

            # Ensure coordinates are in correct order
            px_min, px_max = sorted([x0, x1])
            py_min, py_max = sorted([y0, y1])

            # Convert to complex plane coordinates
            cx_min, cy_max = self.pixel_to_complex(px_min, py_min)
            cx_max, cy_min = self.pixel_to_complex(px_max, py_max)

            print("\n" + "=" * 60)
            print("SELECTED SQUARE BOUNDS (Complex Plane Coordinates)")
            print("=" * 60)
            print(f"X range: [{cx_min:.6f}, {cx_max:.6f}]")
            print(f"Y range: [{cy_min:.6f}, {cy_max:.6f}]")
            print(f"\nWidth:  {cx_max - cx_min:.6f}")
            print(f"Height: {cy_max - cy_min:.6f}")
            print("=" * 60 + "\n")

    def run(self):
        """Start the interactive selector"""
        window_name = "Draw Square (Click & Drag) - Press Q to quit"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.draw_rectangle)

        print("Instructions:")
        print("- Click and drag to draw a square")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset the image\n")

        while True:
            # Create a fresh copy of the image
            img_copy = self.display_img.copy()

            # Draw the current rectangle if dragging
            if self.start_point and self.end_point:
                cv2.rectangle(
                    img_copy, self.start_point, self.end_point, (0, 0, 255), 2
                )

            cv2.imshow(window_name, img_copy)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Reset
                self.start_point = None
                self.end_point = None
                self.drawing = False
                print("Image reset")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image.pgm>")
        sys.exit(1)

    try:
        selector = SquareSelector(sys.argv[1])
        selector.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
