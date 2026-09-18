import numpy as np
import pygame

from .constants import LOC, WAYPOINT, Q
from .openhail_instance import OpenhailInstance

# colors
LIGHT_BLUE = (173, 216, 230)  # Light Blue
DARK_BLUE = (0, 0, 128)  # Dark Blue
LIGHT_GREEN = (144, 238, 144)
DARK_GREEN = (0, 100, 0)
LIGHT_YELLOW = (255, 255, 224)
DARK_YELLOW = (255, 215, 0)
LIGHT_RED = (255, 182, 193)
DARK_RED = (139, 0, 0)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_BG = (240, 240, 240)
GRAY_BORDER = (200, 200, 200)


class RidehailRenderer:
    def __init__(self, render_config: dict, instance: OpenhailInstance) -> None:
        self.initialized = False

        self.window = None
        self.save_frames = render_config.get("save_frames", False)
        self.frame_interval = render_config.get(
            "frame_interval", 60
        )  # Capture every N simulation minutes
        self.last_frame_time = -self.frame_interval  # Ensure first frame is captured
        self.frames = []
        self.frame_times = []  # Track simulation time of each frame

        # Dashboard configuration
        self.dashboard_width = render_config.get("dashboard_width", 320)
        self.map_height = render_config.get("y_resolution", 800)

        # Resolution will be set in set_instance based on aspect ratio
        self.resolution = (800, 600)

        self.map_color = render_config.get("map_color", LIGHT_BLUE)
        self.outline_color = render_config.get("outline_color", DARK_BLUE)
        self.charger_color = render_config.get("charger_color", DARK_RED)
        self.repo_color = render_config.get("repo_color", LIGHT_RED)

        self.full_ev_color = render_config.get("ev_color", DARK_YELLOW)
        self.empty_ev_color = render_config.get("ev_color", DARK_RED)
        self.line_color = render_config.get("ev_color", DARK_GREEN)

        self.request_color = render_config.get("request_color", DARK_RED)
        self.day_colors = [
            (0, 0, 128),
            (0, 128, 0),
            (128, 0, 0),
            (128, 128, 0),
            (128, 0, 128),
        ]

        self.repo_size = 10
        self.charger_size = 10
        self.ev_size = 5

        self.polygons = []
        self.repos = []
        self.chargers = []

        self.reward_history = []
        self.day_start_reward = {}  # Map day -> reward at start of day

        self.start_time = 0

        self.set_instance(instance)

    def set_instance(self, instance):
        self.initialized = True
        gdf = instance.zones_gdf
        self.start_time = instance.start_time

        self.midpoint = instance.midpoint

        # Calculate aspect ratio and dimensions
        x_min, x_max = instance.x_bounds
        y_min, y_max = instance.y_bounds
        width = x_max - x_min
        height = y_max - y_min

        if height == 0:
            height = 1.0  # Avoid div by zero
        aspect = width / height

        self.map_width = int(self.map_height * aspect)

        # Update resolution
        self.resolution = (self.map_width + self.dashboard_width, self.map_height)

        # Re-initialize window if it exists and size changed
        if self.window is not None:
            self.window = pygame.display.set_mode(self.resolution)

        # Scale factor to fit map_width/height with slight padding
        self.scale_factor = min(self.map_width / width, self.map_height / height) * 0.95

        raw_polygons = []
        for geom in gdf.geometry:
            if geom.geom_type == "Polygon":
                raw_polygons.append([list(point) for point in geom.exterior.coords])
            elif geom.geom_type == "MultiPolygon":
                for polygon in geom.geoms:
                    raw_polygons.append(
                        [list(point) for point in polygon.exterior.coords]
                    )

        self.polygons = []
        for polygon in raw_polygons:
            self.polygons.append([self._scale(x, y) for x, y in polygon])

        self.chargers = []
        for location in instance.cs_coords:
            coord = self._scale(*location)
            diamond = [
                (coord[0], coord[1] - self.charger_size),  # Top point
                (coord[0] - self.charger_size, coord[1]),
                (coord[0], coord[1] + self.charger_size),  # Top point
                (coord[0] + self.charger_size, coord[1]),
            ]
            self.chargers.append(diamond)

        self.repos = []
        for location in instance.repo_coords:
            is_charger = np.any(
                np.all(np.isclose(instance.cs_coords, location), axis=1)
            )
            if is_charger:
                continue
            coord = self._scale(*location)
            triangle = [
                (coord[0], coord[1] - self.repo_size),  # Top point
                (
                    coord[0] - np.sqrt(3) / 2 * self.repo_size,
                    coord[1] + self.repo_size / 2,
                ),
                (
                    coord[0] + np.sqrt(3) / 2 * self.repo_size,
                    coord[1] + self.repo_size / 2,
                ),
            ]
            self.repos.append(triangle)

    def render(self, V_render, time, total_reward):
        if not self.initialized:
            raise RuntimeError("Cannot render before an instance has been initialized.")

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.resolution)
            pygame.display.set_caption("OpenHail Simulation")

        pygame.event.pump()

        self.window.fill(WHITE)

        # --- Draw Map Area ---

        for polygon in self.polygons:
            pygame.draw.polygon(self.window, self.map_color, polygon)
            pygame.draw.polygon(
                self.window, self.outline_color, polygon, 2
            )  # Draw outline

        for charger in self.chargers:
            pygame.draw.polygon(self.window, self.charger_color, charger)

        for repo in self.repos:
            pygame.draw.polygon(self.window, self.repo_color, repo)

        for v in range(len(V_render[LOC])):
            if len(V_render[WAYPOINT][v]) > 1:
                points = list(V_render[WAYPOINT][v])
                scaled_points = [self._scale(x, y) for x, y in points]
                pygame.draw.lines(self.window, self.line_color, False, scaled_points, 2)

        avg_charge = 0.0
        num_evs = len(V_render[LOC])

        for v in range(num_evs):
            scaled_location = self._scale(V_render[LOC][v, 0], V_render[LOC][v, 1])

            charge = V_render[Q][v]
            avg_charge += charge

            ev_color = (
                int(
                    self.full_ev_color[0] * charge
                    + self.empty_ev_color[0] * (1 - charge)
                ),
                int(
                    self.full_ev_color[1] * charge
                    + self.empty_ev_color[1] * (1 - charge)
                ),
                int(
                    self.full_ev_color[2] * charge
                    + self.empty_ev_color[2] * (1 - charge)
                ),
            )

            pygame.draw.circle(self.window, ev_color, scaled_location, self.ev_size)
            pygame.draw.circle(
                self.window,
                self.line_color,
                scaled_location,
                self.ev_size + 1,
                2,
            )

        if num_evs > 0:
            avg_charge /= num_evs

        # --- Draw Dashboard ---

        # Separator Line
        pygame.draw.line(
            self.window,
            BLACK,
            (self.map_width, 0),
            (self.map_width, self.map_height),
            2,
        )

        # Dashboard Background (now WHITE as requested)
        dash_rect = pygame.Rect(
            self.map_width, 0, self.dashboard_width, self.map_height
        )
        pygame.draw.rect(self.window, WHITE, dash_rect)

        # Fonts - Changed to Monospace
        font_title = pygame.font.SysFont("Consolas", 24, bold=True)
        if not pygame.font.match_font("Consolas"):  # Fallback if consolas not available
            font_title = pygame.font.SysFont("Courier New", 24, bold=True)

        font_label = pygame.font.SysFont("Consolas", 16, bold=True)
        font_value = pygame.font.SysFont("Consolas", 20)
        font_axis = pygame.font.SysFont("Consolas", 12)

        # Text Positions
        start_x = self.map_width + 20
        current_y = 20

        # Title
        text = font_title.render("Simulation Stats", True, DARK_BLUE)
        self.window.blit(text, (start_x, current_y))
        current_y += 50

        # Time Handling
        real_time = self.start_time + time
        # Calculate day relative to start time (0 = first 24h block starting at
        # self.start_time)
        day = int((real_time - self.start_time) // 86400)

        # Time within current day (0 to 86400) relative to start_time
        day_rel_secs = (real_time - self.start_time) % 86400

        # For display, we want actual clock time
        clock_secs = real_time % 86400
        hh = int(clock_secs // 3600)
        mm = int((clock_secs % 3600) // 60)

        # Time Display
        text = font_label.render("Time:", True, BLACK)
        self.window.blit(text, (start_x, current_y))
        time_str = f"Day {day}, {hh:02d}:{mm:02d}"
        text = font_value.render(time_str, True, DARK_BLUE)
        self.window.blit(text, (start_x + 80, current_y))
        current_y += 40

        # Reward
        text = font_label.render("Profit:", True, BLACK)
        self.window.blit(text, (start_x, current_y))
        text = font_value.render(
            f"${total_reward:.2f}", True, DARK_GREEN if total_reward >= 0 else DARK_RED
        )
        self.window.blit(text, (start_x + 80, current_y))
        current_y += 40

        # Avg Battery
        text = font_label.render("Avg Battery:", True, BLACK)
        self.window.blit(text, (start_x, current_y))
        bat_pct = int(avg_charge * 100)

        bat_color = DARK_GREEN
        if bat_pct < 20:
            bat_color = DARK_RED
        elif bat_pct < 50:
            bat_color = DARK_YELLOW

        text = font_value.render(f"{bat_pct}%", True, bat_color)
        self.window.blit(text, (start_x + 120, current_y))
        current_y += 60

        # Record history with day tracking and reset logic
        if day not in self.day_start_reward:
            self.day_start_reward[day] = total_reward

        daily_reward = total_reward - self.day_start_reward[day]

        # Each entry: (day, time_of_day_relative, daily_reward)
        if (
            not self.reward_history
            or self.reward_history[-1][1] < time
            or self.reward_history[-1][0] != day
        ):
            self.reward_history.append((day, day_rel_secs, daily_reward))

        # --- Profit Graph ---
        graph_height = 200
        graph_width = self.dashboard_width - 40
        graph_x = start_x
        graph_y = current_y

        # Graph Background
        pygame.draw.rect(
            self.window, WHITE, (graph_x, graph_y, graph_width, graph_height)
        )
        # Border
        pygame.draw.rect(
            self.window, BLACK, (graph_x, graph_y, graph_width, graph_height), 1
        )

        # Fixed axes (0 to 24h relative to start time)
        min_t, max_t = 0, 86400
        t_range = max_t - min_t

        # Y-axis scaling based on max DAILY reward seen so far across all history? Or
        # just this day?
        # Let's scale based on max daily reward observed across all days to keep scale
        # consistent if possible,
        # but auto-growing is fine.
        rewards = [x[2] for x in self.reward_history]
        min_r, max_r = (
            min(rewards) if rewards else 0,
            max(rewards) if rewards else 100,
        )

        if max_r < 100:
            max_r = 100  # Minimum scale

        r_range = max_r - min_r
        if r_range == 0:
            r_range = 1

        # Ticks Helper
        window = self.window
        assert window is not None

        def draw_x_tick(val_t, label):
            norm_x = (val_t - min_t) / t_range
            px = graph_x + int(norm_x * graph_width)
            pygame.draw.line(
                window,
                BLACK,
                (px, graph_y + graph_height),
                (px, graph_y + graph_height + 5),
                1,
            )
            lx = font_axis.render(label, True, BLACK)
            window.blit(lx, (px - lx.get_width() // 2, graph_y + graph_height + 7))

        def draw_y_tick(val_r, label):
            norm_y = (val_r - min_r) / r_range
            py = graph_y + graph_height - int(norm_y * graph_height)
            pygame.draw.line(window, BLACK, (graph_x - 5, py), (graph_x, py), 1)
            ly = font_axis.render(label, True, BLACK)
            window.blit(ly, (graph_x - ly.get_width() - 7, py - ly.get_height() // 2))

        # Draw X Ticks (Relative to start time of 3AM)
        # 0 -> 03
        # 21600 (6h) -> 09
        # 43200 (12h) -> 15 (3PM)
        # 64800 (18h) -> 21 (9PM)
        # 86400 (24h) -> 03 (3AM)

        # Helper to format time given start offset
        start_hour = int(self.start_time / 3600)

        draw_x_tick(0, f"{start_hour:02d}")
        draw_x_tick(21600, f"{(start_hour + 6) % 24:02d}")
        draw_x_tick(43200, f"{(start_hour + 12) % 24:02d}")
        draw_x_tick(64800, f"{(start_hour + 18) % 24:02d}")
        draw_x_tick(86400, f"{(start_hour + 24) % 24:02d}")

        # Draw Y Ticks (Min, Max and intermediate)
        num_yticks = 5
        for i in range(num_yticks):
            val = min_r + (max_r - min_r) * i / (num_yticks - 1)
            draw_y_tick(val, f"{int(val)}")

        if len(self.reward_history) > 1:
            # Group points by day
            points_by_day = {}
            for d, t_day, r in self.reward_history:
                if d not in points_by_day:
                    points_by_day[d] = []
                # Normalize coords
                norm_x = (t_day - min_t) / t_range
                norm_y = (r - min_r) / r_range

                # Clamp to graph bounds
                norm_y = max(0, min(1, norm_y))

                px = graph_x + int(norm_x * graph_width)
                py = graph_y + graph_height - int(norm_y * graph_height)
                points_by_day[d].append((px, py))

            # Draw lines per day
            for d, points in points_by_day.items():
                if len(points) > 1:
                    color = self.day_colors[d % len(self.day_colors)]
                    pygame.draw.lines(self.window, color, False, points, 2)

            # Zero line if visible
            if min_r < 0 < max_r:
                zero_y = (
                    graph_y + graph_height - int((0 - min_r) / r_range * graph_height)
                )
                pygame.draw.line(
                    self.window,
                    GRAY_BORDER,
                    (graph_x, zero_y),
                    (graph_x + graph_width, zero_y),
                    1,
                )

        # Graph Label
        text = font_label.render("Daily Profit", True, BLACK)
        self.window.blit(text, (graph_x, graph_y - 25))

        pygame.display.flip()

        # Capture frame if saving frames for GIF at regular intervals
        if self.save_frames and (time - self.last_frame_time) >= self.frame_interval:
            # Get the surface and convert to numpy array
            frame = pygame.surfarray.array3d(self.window)
            # Transpose to get correct orientation (pygame uses (width, height,
            # channels))
            frame = np.transpose(frame, (1, 0, 2))
            self.frames.append(frame)
            self.frame_times.append(time)
            self.last_frame_time = time

    def _scale(self, x, y):
        # Scale to center on self.map_width/2, self.map_height/2
        return (
            int(self.scale_factor * (x - self.midpoint[0]) + (self.map_width // 2)),
            int(-self.scale_factor * (y - self.midpoint[1]) + (self.map_height // 2)),
        )

    def get_frames(self):
        """Return the captured frames and frame times, then clear the buffer."""
        frames = self.frames.copy()
        times = self.frame_times.copy()
        self.frames = []
        self.frame_times = []
        self.last_frame_time = -self.frame_interval
        return frames, times

    def close(self) -> None:
        """Release the Pygame display and its process-level resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
