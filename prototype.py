import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm


class StyleEstimator:
    def __init__(self):
        model_path = os.path.join("/content/drive/MyDrive/samocat/", "k-means_model_new.pkl")
        pca_path = os.path.join("/content/drive/MyDrive/samocat/", "pca_model_new.pkl")
        scaler_path = os.path.join("/content/drive/MyDrive/samocat/", "scaler_new.pkl")

        with open(model_path, "rb") as m:
            self.model = pickle.load(m)
        with open(pca_path, "rb") as p:
            self.pca = pickle.load(p)
        with open(scaler_path, "rb") as s:
            self.scaler = pickle.load(s)

        self.trips_data = pd.DataFrame()
        clusters_description = {0: "Корректное поведение",
                                1: "Потенциально опасное поведение",
                                2: "Опасное поведение"}

    def get_raw_trip_data(self, df, ride_id):
        data = df[df.ride_id == ride_id].reset_index()
        N = len(data)

        data["speed"] = data.wheel.div(3.6)

        accel = [0.0]
        dt_arr = [0.0]
        a = 0.0
        for i in range(N - 1):
            dv = data.speed[i + 1] - data.speed[i]
            dt = (data.time[i + 1] - data.time[i]).total_seconds()
            dt_arr.append(dt)
            if dt > 1.0:
                a = dv / dt
            accel.append(a)
        data["accel"] = accel
        data["dt"] = dt_arr

        data.drop(labels=["time", "wheel"], axis=1, inplace=True)
        return data

    def get_trips_data(self, csv_path):
        scooters = pd.read_csv(csv_path, parse_dates={"time": ["gps_date", "gps_t"]}, index_col=0).drop(["lat", "lon"],
                                                                                                        axis=1)

        col_names = ["average_speed", "peak_speed", "average_acceleration", "peak_acceleration",
                     "rapid_overclock", "hard_braking"]
        self.trips_data = pd.DataFrame(columns=col_names)

        idxs = set(scooters.ride_id)
        for trip_id in tqdm(idxs):
            raw_data = self.get_raw_trip_data(scooters, trip_id)

            average_speed = raw_data.speed.mean()
            peak_speed = raw_data.speed.max()
            average_acceleration = raw_data.accel.mean()
            peak_acceleration = raw_data.accel.max()
            rapid_overclock = len(raw_data[(raw_data.accel > 3.43) & (raw_data.dt < 5)])
            hard_braking = len(raw_data[(raw_data.accel < -4.42) & (raw_data.dt < 5)])

            info = [average_speed, peak_speed, average_acceleration, peak_acceleration,
                    rapid_overclock, hard_braking]
            self.trips_data.loc[trip_id] = info

    def predict_cluster(self):
        X = self.trips_data.values.tolist()
        X = self.pca.transform(X)
        X = self.scaler.transform(X)
        c_predicted = [int(x) for x in self.model.predict(X)]
        return c_predicted

    def estimate(self):
        score_dict = {0: 0.0, 1: 0.5, 2: 1.0}
        clusters = self.predict_cluster()
        scores = [score_dict[i] for i in clusters]
        self.trips_data['class_predicted'] = clusters
        res = sum(scores) / len(scores)
        return res

    def make_recommendations(self):

      data = self.trips_data
      colors = {0: 'green', 1: 'yellow', 2: 'red'}

      data['color'] = [colors[i] for i in data.class_predicted]
      data['recommendation'] = ''

      # Пороги выбраны при помощи тетрадки clust_recommendations.ipynb
      data.loc[data.peak_acceleration >= 3.0, 'recommendation'] = 'Резкие ускорения и торможения на дорогах - не лучшее решение, не подвергай опасности себя и окружающих!'
      data.loc[(data.average_speed >= 5.56) & (data.recommendation == ''), 'recommendation'] = 'Сбавь скорость, так гонять небезопасно!'
      data.loc[(data.peak_speed >= 7.3) & (data.recommendation == ''), 'recommendation'] = 'Дружище, помедленнее! Побереги себя и окружающих!'
      data.loc[(data.hard_braking >= 1.0) & (data.recommendation == ''), 'recommendation'] = 'Будь осторожнее! Лучше тормозить заранее!'

      data.loc[(data.peak_acceleration >= 2.3) & (data.recommendation == ''), 'recommendation'] = 'Отличная поездка, но в следующий раз разгоняйся, пожалуйста, помедленнее! :)'
      data.loc[(data.peak_speed >= 6.9) & (data.recommendation == ''), 'recommendation'] = 'Отличная поездка! Но иногда все же лучше сбавить скорость, гонщик :)'

      data.loc[(data.color == 'green') & (data.recommendation == ''), 'recommendation'] = 'Отличная поездка, дружище! До скорой встречи! :)'
      data.loc[(data.color == 'yellow') & (data.recommendation == ''), 'recommendation'] = 'Отличная поездка, дружище! Только будь в следующий раз чуточку аккуратнее :)'
      data.loc[(data.color == 'red') & (data.recommendation == ''), 'recommendation'] = 'Отличная поездка, дружище! Только не забывай о безопасности :)'

      data.to_csv("recommendations_user.csv", index = False)
      print()
      print("Recommendations for this user are here: recommendations_user.csv")

def main():
    estimator = StyleEstimator()
    user_data_path = "/content/drive/MyDrive/samocat/test_user_rides.csv"
    estimator.get_trips_data(user_data_path)
    fscore = estimator.estimate()
    estimator.make_recommendations()
    print(f"User score: {fscore:.2f}")


main()