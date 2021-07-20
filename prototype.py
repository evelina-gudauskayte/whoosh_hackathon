import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm


class StyleEstimator:
    def __init__(self):
        model_path = os.path.join("./", "k-means_model.pkl")
        pca_path = os.path.join("./", "pca_model.pkl")
        scaler_path = os.path.join("./", "scaler.pkl")

        with open(model_path, "rb") as m:
            self.model = pickle.load(m)
        with open(pca_path, "rb") as p:
            self.pca = pickle.load(p)
        with open(scaler_path, "rb") as s:
            self.scaler = pickle.load(s)

        self.trips_data = pd.DataFrame()
        clusters_description = {0: "Корректное поведение",
                                1: "Опасное поведение",
                                2: "Потенциально опасное поведение"}

    def get_raw_trip_data(self, df, ride_id):
        data = df[df.ride_id == ride_id].reset_index()
        N = len(data)

        data["speed"] = data.wheel.div(3.6)

        accel = [0.0]
        a = 0.0
        for i in range(N - 1):
            dv = data.speed[i + 1] - data.speed[i]
            dt = (data.time[i + 1] - data.time[i]).total_seconds()
            if dt > 1.0:
                a = dv / dt
            accel.append(a)
        data["accel"] = accel

        data.drop(labels=["time", "wheel"], axis=1, inplace=True)
        return data

    def get_trips_data(self, csv_path):
        scooters = pd.read_csv(csv_path, parse_dates={"time": ["gps_date", "gps_t"]}, index_col=0).drop(["lat", "lon"],
                                                                                                        axis=1)

        col_names = ["average_speed", "peak_speed", "average_acceleration", "peak_acceleration"]
        self.trips_data = pd.DataFrame(columns=col_names)

        idxs = set(scooters.ride_id)
        for trip_id in tqdm(idxs):
            raw_data = self.get_raw_trip_data(scooters, trip_id)

            average_speed = raw_data.speed.mean()
            peak_speed = raw_data.speed.max()
            average_acceleration = raw_data.accel.mean()
            peak_acceleration = raw_data.accel.max()

            info = [average_speed, peak_speed, average_acceleration, peak_acceleration]
            self.trips_data.loc[trip_id] = info

    def predict_cluster(self):
        X = self.trips_data.values.tolist()
        X = self.pca.transform(X)
        X = self.scaler.transform(X)
        c_predicted = [int(x) for x in self.model.predict(X)]
        return c_predicted

    def estimate(self):
        score_dict = {0: 0.0, 1: 1.0, 2: 0.5}
        clusters = self.predict_cluster()
        scores = [score_dict[i] for i in clusters]
        res = sum(scores) / len(scores)
        return res


def main():
    estimator = StyleEstimator()
    user_data_path = "./test_user_rides.csv"
    estimator.get_trips_data(user_data_path)
    fscore = estimator.estimate()
    print(f"User score: {fscore:.2f}")


main()
