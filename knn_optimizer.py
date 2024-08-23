import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

class KNNOptimizer:
    def __init__(self, n_neighbors=3):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.df = pd.read_csv("KNN_DATASET_MINOR_PROJECT.csv")
        
        # Ensure feature names are consistent
        self.x = self.df[['No of Vehicles']]
        self.y = self.df['time in Sec']
        self.model.fit(self.x, self.y)

    def predict(self, X):
        # Create DataFrame with correct column names
        X_df = pd.DataFrame(X, columns=['No of Vehicles'])
        return self.model.predict(X_df)

    def update(self, road_id, no_veh_detected, vehicles_left, time_allocated):
        # Calculate the new time based on vehicles left
        if vehicles_left <= 5:
            new_time = time_allocated  # Keep the same time if very few vehicles remain
        else:
            time_per_vehicle = time_allocated / (no_veh_detected - vehicles_left)
            new_time = time_per_vehicle * vehicles_left + time_allocated
            if new_time >= 80:
                new_time = 80  # Cap the time at 80 seconds

        # Remove the old row for the road and add the updated time
        self.df = self.df[(self.df['Road ID'] != road_id) | (self.df['No of Vehicles'] != no_veh_detected)]
        new_row = pd.DataFrame({'Road ID': [road_id], 'No of Vehicles': [no_veh_detected], 'time in Sec': [new_time]})
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        self.df.to_csv('KNN_DATASET_MINOR_PROJECT.csv', index=False)
        self.df=pd.read_csv('KNN_DATASET_MINOR_PROJECT.csv')

    def predict_custom_input(self, no_of_vehicles):
        input_array = np.array([no_of_vehicles]).reshape(1, -1)
        input_df = pd.DataFrame(input_array, columns=['No of Vehicles'])
        prediction = self.model.predict(input_df)
        return prediction
