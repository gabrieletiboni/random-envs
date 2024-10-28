import numpy as np

class Figure:
    def does_trajectory_hit(self, pos, vel, acc, dt):
        raise NotImplementedError

    def does_line_hit(self, pos1, pos2):
        raise NotImplementedError


class HalfPlane(Figure):
    def __init__(self, value, relation="gt", variable="x"):
        """A half plane perpendicular to the axis.
        The area of the half plane is defined as the half plane
        with coordinate `variable` `relation` `value`
        (eg: x greater than value)."""
        self.value = value
        self.relation = 1.0 if relation == "gt" else -1.0
        self.variable = 0 if variable == "x" else 1

    def does_trajectory_hit(self, pos, vel, acc, dt):
        c = (pos[self.variable] - self.value) * self.relation
        b = vel[self.variable] * self.relation
        a = 0.5 * acc[self.variable] * self.relation

        if c >= 0.0:
            return True

        eps = 1e-7
        if abs(b) < eps and abs(a) < eps:
            return False
        if abs(a) < eps:
            zero = -c / b
            return 0 < zero < dt

        delta = np.square(b) - 4 * a * c

        if delta < 0.0:
            return False

        zero_one = (-b + np.sqrt(delta)) / (2 * a)
        zero_two = (-b - np.sqrt(delta)) / (2 * a)

        return 0 < zero_one < dt or 0 < zero_two < dt

    def does_line_hit(self, pos1, pos2):
        if (pos1[self.variable] - self.value) * self.relation <= 0 and np.sign(
            pos1[self.variable] - self.value
        ) == np.sign(pos2[self.variable] - self.value):
            return False
        return True


class Rectangle(Figure):
    def __init__(self, x0, x1, y0, y1):
        """A (axis-aligned) rectangle is stored as a list of half planes.
        The rectangle is defined as the intersection of the half planes."""
        self.planes = [
            HalfPlane(x0, "gt", "x"),
            HalfPlane(x1, "lt", "x"),
            HalfPlane(y0, "gt", "y"),
            HalfPlane(y1, "lt", "y"),
        ]

        self.values = np.array([x0, x1, y0, y1], dtype=np.float32)

    def does_trajectory_hit(self, pos, vel, acc, dt):
        a = True
        for x in self.planes:
            a = a and x.does_trajectory_hit(pos, vel, acc, dt)
            if not a: break
        return a

    def does_line_hit(self, pos1, pos2):
        a = True
        for x in self.planes:
            a = a and x.does_line_hit(pos1, pos2)
        return a

    @staticmethod
    def from_line_with_epsilon(start, end, other_variable, epsilon, variable="x"):
        assert start < end, "Start position must be strictly lower than end position"
        x0 = start - epsilon
        x1 = end + epsilon
        y0 = other_variable - epsilon
        y1 = other_variable + epsilon
        if variable != "x":
            x0, x1, y0, y1 = y0, y1, x0, x1
        return Rectangle(x0, x1, y0, y1)


class Box(Figure):
    def __init__(self):
        """A box is simply a collection of rectangles,
        and its area is the union of the rectangles."""
        self.rectangles: list[Rectangle] = []

    def add_rectangle(self, rectangle: Rectangle):
        self.rectangles.append(rectangle)

    def does_trajectory_hit(self, pos, vel, acc, dt):
        a = False
        for x in self.rectangles:
            a = a or x.does_trajectory_hit(pos, vel, acc, dt)
            if a: break
        return a

    def does_line_hit(self, pos1, pos2):
        a = False
        for x in self.rectangles:
            a = a or x.does_line_hit(pos1, pos2)
        return a


class Random2DNavigationBox(Box):
    def __init__(self, hit_wall_epsilon=0.01, wall_height=0.9):
        super().__init__()

        self.hit_wall_epsilon = hit_wall_epsilon
        self.wall_height = wall_height

        rectangle = Rectangle.from_line_with_epsilon(
            -0.5, 0.5, 0.0, self.hit_wall_epsilon, variable="x"
        )
        self.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            -0.5, 0.5, 1.2, self.hit_wall_epsilon, variable="x"
        )
        self.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            0.0, 1.2, -0.5, self.hit_wall_epsilon, variable="y"
        )
        self.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            0.0, 1.2, 0.5, self.hit_wall_epsilon, variable="y"
        )
        self.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            -0.5, -0.2, self.wall_height, self.hit_wall_epsilon, variable="x"
        )
        self.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            0.2, 0.5, self.wall_height, self.hit_wall_epsilon, variable="x"
        )
        self.add_rectangle(rectangle)
    
    def get_area_before_wall(self):
        """Returns an array with flattened coordinates
        x0, x1, y0, y1 that represents the rectangular
        area before the walls and inside the perimeter.
        Optionally indicate hscale and vscale to """

        h = (0.5 - self.hit_wall_epsilon)
        #* init_pos_distr_fraction_h
        v_t = self.wall_height - self.hit_wall_epsilon
        #* init_pos_distr_fraction_v + self.initial_v_offset
        return np.array([-h, h, self.hit_wall_epsilon, v_t])
    
    @staticmethod
    def get_scaled_area_before_wall(original, vscale, hscale, vbottom):
        scaled = np.array(original)
        # scale horizontally
        scaled[:2] = (original[:2] - np.mean(original[:2]))*hscale + np.mean(original[:2])
        # set bottom
        scaled[2] = vbottom
        # scale vertically
        scaled[3] = (original[3] - vbottom)*vscale + vbottom
        return scaled