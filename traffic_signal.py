import numpy as np
from yolo_detector import YOLODetector
from knn_optimizer import KNNOptimizer

def simulate_cycle(knn_optimizer, road_data):
    max_road_index = np.argmax([data[0] for data in road_data])
    cycle_order = road_data[max_road_index:] + road_data[:max_road_index]

    for road_id, (vehicles_waiting, vehicles_left) in enumerate(cycle_order, start=1):
        print(f"Processing Road {road_id} with {vehicles_waiting} vehicles waiting...")

        # Predict the time allocation using KNN
        predicted_time = vehicles_waiting  # Start by allocating time equal to the number of vehicles
        if road_id > 1:
            predicted_time = knn_optimizer.predict([[vehicles_waiting]])[0]

        # Update the time allocation in the dataset based on vehicles left
        knn_optimizer.update(road_id, vehicles_waiting, vehicles_left, predicted_time)
        
        # Print the results
        print(f"Predicted Time Allocation: {predicted_time:.2f} seconds")
        print(f"Vehicles Waiting: {vehicles_waiting}")
        print(f"Vehicles Left: {vehicles_left}")

def main():
    knn_optimizer = KNNOptimizer()  # Use KNNOptimizer class instance
    
    # Example data for Cycle 1
    road_data_cycle_1 = [
        (25, 20),  # Road 1: 25 vehicles detected, 20 left
        (10, 5),   # Road 2: 10 vehicles detected, 5 left
        (15, 10),  # Road 3: 15 vehicles detected, 10 left
        (20, 15)   # Road 4: 20 vehicles detected, 15 left
    ]

    print("Cycle 1:")
    simulate_cycle(knn_optimizer, road_data_cycle_1)
    
    # Example data for Cycle 2
    road_data_cycle_2 = [
        (28, 22),  # Road 1: 28 vehicles detected, 22 left
        (12, 7),   # Road 2: 12 vehicles detected, 7 left
        (18, 12),  # Road 3: 18 vehicles detected, 12 left
        (22, 17)   # Road 4: 22 vehicles detected, 17 left
    ]

    print("\nCycle 2:")
    simulate_cycle(knn_optimizer, road_data_cycle_2)

if __name__ == "__main__":
    main()
