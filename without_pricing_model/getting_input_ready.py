from geopy.distance import geodesic
import numpy as np

# -----------------------------
# 1. INPUT DATA
# -----------------------------

drivers = [
    {"id": "CarA", "lat": 51.5074, "lon": -0.1278, "capacity": 2},  # London
    {"id": "CarB", "lat": 51.5090, "lon": -0.08,   "capacity": 2},  # London (slightly east)
]

passengers = [
    {"id": "P1", "pickup": (51.5200, -0.10), "dropoff": (51.5300, -0.12)},
    {"id": "P2", "pickup": (51.5000, -0.09), "dropoff": (51.4950, -0.08)},
    {"id": "P3", "pickup": (51.5150, -0.13), "dropoff": (51.5250, -0.14)},
    {"id": "P4", "pickup": (51.5050, -0.15), "dropoff": (51.5100, -0.16)},
]


# -----------------------------
# 2. BUILD POINT LIST
# -----------------------------

points = []

# Add driver start points
for d in drivers:
    points.append((d["lat"], d["lon"]))

# Add passenger pickup points
for p in passengers:
    points.append(p["pickup"])

# Add passenger dropoff points
for p in passengers:
    points.append(p["dropoff"])

print("Points list:")
for i, p in enumerate(points):
    print(i, p)

n = len(points)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j :
            dist_matrix[i,j] = geodesic(points[i], points[j]).km


print("\nDistance matrix (km):") 
print(np.round(dist_matrix, 2))



# -----------------------------
# 4. SIMPLE GREEDY ASSIGNMENT
# -----------------------------

assigned = set()  # passengers already taken
car_assignments = {d["id"]: [] for d in drivers}

for d_idx, d in enumerate(drivers):
    car_id = d["id"]
    capacity = d["capacity"]

    # Compute distance from this car to each passenger pickup
    distances = []
    for p_idx, p in enumerate(passengers):
        pickup_point_index = 2 + p_idx  # because pickups start at index 2
        dist = dist_matrix[d_idx, pickup_point_index]
        distances.append((dist, p["id"], p_idx))

    print(distances)
    # Sort passengers by distance (closest first)
    distances.sort(key=lambda x: x[0])

    # Assign until capacity is full
    for dist, pid, p_idx in distances:
        if len(car_assignments[car_id]) >= capacity:
            break
        if pid not in assigned:
            car_assignments[car_id].append(pid)
            assigned.add(pid)

print("\nAssignments:")
for car, ps in car_assignments.items():
    print(car, "→", ps)
