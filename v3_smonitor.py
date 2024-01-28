import random
import numpy as np
from PIL import Image, ImageDraw
import math
import cv2

# Function to handle mouse clicks and get two points
def get_scale_from_user(image_path):
    def click_event(event, x, y, flags, points):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            if len(points) == 2:
                cv2.line(floor_plan, points[0], points[1], (255, 0, 0), 2)
                cv2.imshow("Mark 2 points on the Floor Plan", floor_plan)

    floor_plan = cv2.imread(image_path)
    cv2.imshow("Floor Plan", floor_plan)
    points = []
    cv2.setMouseCallback("Floor Plan", click_event, points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 2:
        pixel_distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        real_length = float(input("Enter the real-world length between these points in meters: "))
        return pixel_distance / real_length  # pixels per meter
    else:
        print("Two points were not selected. Exiting.")
        exit(1)

def place_iot_devices(image_path):
    def click_event(event, x, y, flags, points):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(floor_plan, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Place IoT Devices", floor_plan)

    floor_plan = cv2.imread(image_path)
    cv2.imshow("Place IoT Devices", floor_plan)
    points = []
    cv2.setMouseCallback("Place IoT Devices", click_event, points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points

# Conversion functions
def pixels_to_meters(pixels, pixels_per_meter):
    return pixels / pixels_per_meter
# Conversion functions
def meters_to_pixels(meters, pixels_per_meter):
    return meters * pixels_per_meter

def cost_function(monitor_positions, spots):
    total_cost = 0
    for spot in spots:
        x, y, priority = spot  # Unpack the values correctly
        min_distance = min(
            np.linalg.norm(np.array(monitor_position) - np.array((x, y)))
            for monitor_position in monitor_positions
        )
        total_cost += priority * min_distance
    return total_cost

def is_inside_contour(point, contour_mask):
    # Convert point coordinates to integers
    x, y = int(point[0]), int(point[1])
    return contour_mask[y, x] == 255

def optimize_monitors_within_area(floor_plan, spots, num_monitors, monitor_range, max_iterations, area):
    best_monitor_positions = None
    best_cost = float('inf')
    width, height = floor_plan.size

    for _ in range(max_iterations):
        monitor_positions = [
            (random.uniform(area[0], area[2]), random.uniform(area[1], area[3]))
            for _ in range(num_monitors)
        ]
        current_cost = cost_function(monitor_positions, spots)

        if current_cost < best_cost:
            best_cost = current_cost
            best_monitor_positions = monitor_positions

    return best_monitor_positions

def find_red_area(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the red color
    lower_red = np.array([200, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 100, 100], dtype=np.uint8)

    # Create a mask to isolate the red color
    mask = cv2.inRange(image_rgb, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour (assuming the largest red area is the floor plan)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, x + w, y + h)

    return None

def calculate_ideal_devices(floor_plan, contour_mask, monitor_radius):
    width, height = floor_plan.size
    horizontal_spacing = monitor_radius * 3 / 2
    vertical_spacing = math.sqrt(3) * monitor_radius

    ideal_monitor_positions = []

    for x in range(int(horizontal_spacing / 2), width, int(horizontal_spacing)):
        for y in range(int(vertical_spacing / 4), height, int(vertical_spacing)):
            if x % (2 * int(horizontal_spacing)) == 0:
                y += vertical_spacing / 2

            if is_inside_contour((x, y), contour_mask):
                ideal_monitor_positions.append((x, y))

    num_devices_needed = len(ideal_monitor_positions)
    print(f"Number of monitor devices needed: {num_devices_needed}")
    return ideal_monitor_positions

def find_exact_outline_area(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_red = np.array([200, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 100, 100], dtype=np.uint8)
    mask = cv2.inRange(image_rgb, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [approx], -1, (255), thickness=cv2.FILLED)
        return contour_mask

    return None

def select_outermost_devices(ideal_positions):
    if not ideal_positions:
        return []

    # Group devices by rows and columns
    rows = {}
    columns = {}
    for x, y in ideal_positions:
        rows.setdefault(y, []).append((x, y))
        columns.setdefault(x, []).append((x, y))

    outermost_devices = set()

    # Get extreme devices in each row
    for y, row_devices in rows.items():
        if row_devices:
            row_devices.sort()
            outermost_devices.add(row_devices[0])        # Leftmost
            outermost_devices.add(row_devices[-1])       # Rightmost

    # Get extreme devices in each column
    for x, column_devices in columns.items():
        if column_devices:
            column_devices.sort(key=lambda pos: pos[1])
            outermost_devices.add(column_devices[0])     # Topmost
            outermost_devices.add(column_devices[-1])    # Bottommost

    print(f"Number of outermost monitor devices selected: {len(outermost_devices)}")
    return list(outermost_devices)



def display_menu():
    print("Menu:")
    print("1. Optimize monitors for given number (user-defined)")
    print("2. Calculate ideal number of monitors for full area")
    print("3. Optimize monitors along the border for external threat protection")
    choice = input("Enter your choice (1, 2, or 3): ")
    return int(choice)

# Load the floor plan image
floor_plan_image = Image.open("test_plan.png")
floor_plan_path = "test_plan.png"

# Get the scale conversion from the user
pixels_per_meter = get_scale_from_user(floor_plan_path)

iot_device_positions = place_iot_devices(floor_plan_path)

# Define the important spots as a list of (x, y, priority) coordinates
# Create important_spots from IoT device positions
default_priority = 1  # You can set a default priority here
important_spots = [(int(x), int(y), default_priority) for x, y in iot_device_positions]

# Define the number of monitors, monitor range, and maximum iterations
num_monitors = 3
monitor_range_meters = 2 # Adjust the monitor range as needed
monitor_range_pixels = meters_to_pixels(monitor_range_meters, pixels_per_meter)

max_iterations = 450000

contour_mask = find_exact_outline_area(floor_plan_path)
optimization_area = find_red_area(floor_plan_path)
#optimization_area = (470, 290, 930, 900)

#draw = ImageDraw.Draw(floor_plan_image)
#left, upper, right, lower = optimization_area
#draw.rectangle([left, upper, right, lower], outline="blue")

# Load the IoT device icon image
icon_image = Image.open("iot_device_icon.png")
icon_width, icon_height = icon_image.size
new_icon_width = icon_width // 20
new_icon_height = icon_height // 20

icon_image = icon_image.resize((new_icon_width, new_icon_height))

# Load the monitor device icon image
monitor_image = Image.open("monitor_icon.png")
monitor_width, monitor_height = monitor_image.size
new_monitor_width = monitor_width // 10
new_monitor_height = monitor_height // 10

monitor_image = monitor_image.resize((new_monitor_width, new_monitor_height))

# Display the menu
menu_choice = display_menu()

if menu_choice == 1:
    optimized_monitor_positions = optimize_monitors_within_area(floor_plan_image, important_spots, num_monitors, monitor_range_pixels, max_iterations, optimization_area)
elif menu_choice == 2:
    optimized_monitor_positions = calculate_ideal_devices(floor_plan_image, contour_mask, monitor_range_pixels)
elif menu_choice == 3:
   # First, get the ideal positions
    ideal_monitor_positions = calculate_ideal_devices(floor_plan_image, contour_mask, monitor_range_pixels)
    # Then, select only the outermost devices
    optimized_monitor_positions = select_outermost_devices(ideal_monitor_positions)


# Create a copy of the floor plan image with monitor positions and icons marked
floor_plan_with_monitors = floor_plan_image.copy()
draw = ImageDraw.Draw(floor_plan_with_monitors)
for x, y in optimized_monitor_positions:
    draw.ellipse((x - monitor_range_pixels, y - monitor_range_pixels, x + monitor_range_pixels, y + monitor_range_pixels), outline="red")
    monitor_x = int(x - new_monitor_width // 2)
    monitor_y = int(y - new_monitor_height // 2)
    floor_plan_with_monitors.paste(monitor_image, (monitor_x, monitor_y))
    
# Paste the IoT device icons at the positions of important spots
for x, y, _ in important_spots:
    icon_x = x - new_icon_width // 2
    icon_y = y - new_icon_height // 2
    floor_plan_with_monitors.paste(icon_image, (icon_x, icon_y))

# Save or display the result
floor_plan_with_monitors.show()
