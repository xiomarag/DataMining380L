import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA_transform:
    def __init__(self, filename, class_num, dep_or_arr):
        self.filename = filename
        self.class_num = class_num
        self.delay_type = dep_or_arr
        
    def read_in_data(self):
        return pd.read_csv(self.filename)
    
    def select_feature_manually(self):
        self.df = self.read_in_data()
        self.df_feature = self.df[["AIRLINE_ID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "MONTH","DAY_OF_MONTH", "PLANE_AGE", "AWND", "PRCP", "SNOW", "TMAX", "TAVG",
             "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE"]]
    
    def convert_time(self, time):
        hour = np.floor(time/100)
        minutes = hour*60 + time - hour*100
        return minutes
    
    def make_dataset(self):
        airline_oh_df = pd.get_dummies(self.df_feature['AIRLINE_ID'])
        arr_airport_oh_df = pd.get_dummies(self.df_feature['DEST_AIRPORT_ID'])
        dep_airport_oh_df = pd.get_dummies(self.df_feature['ORIGIN_AIRPORT_ID'])
        
        airline_oh = airline_oh_df.to_numpy()
        arr_airport_oh = arr_airport_oh_df.to_numpy()
        dep_airport_oh = dep_airport_oh_df.to_numpy()
        
        self.df_feature = self.df_feature.drop(['AIRLINE_ID', 'DEST_AIRPORT_ID', 'ORIGIN_AIRPORT_ID'], axis=1)
        
        self.df_feature = self.df_feature.fillna(0)
        
        arr_time = self.df_feature["CRS_ARR_TIME"].to_numpy().reshape(-1, 1)
        dep_time = self.df_feature["CRS_DEP_TIME"].to_numpy().reshape(-1, 1)
        arr_time_min = self.convert_time(arr_time)
        dep_time_min = self.convert_time(dep_time)
        time_min_df = pd.DataFrame(np.hstack([arr_time_min, dep_time_min]), columns=['CRS_ARR_TIME_MIN', 'CRS_DEP_TIME_MIN'])

        self.df_feature = self.df_feature.drop(["CRS_ARR_TIME", "CRS_DEP_TIME"], axis=1)
        self.df_feature = self.df_feature.join(time_min_df)
        
        features_without_id = self.df_feature.to_numpy()
        self.features = np.hstack([airline_oh, arr_airport_oh, dep_airport_oh, features_without_id])
        self.df_truth = self.df[["CANCELLED", "DEP_DELAY_NEW", "ARR_DELAY_NEW"]]
        
    
    def LDA(self):
        if self.delay_type == "dep":
            name = "DEP_DELAY_NEW"
        if self.delay_type == "arr":
            name = "ARR_DELAY_NEW"
        
        self.select_feature_manually()
        self.make_dataset()
        
        y = np.array(self.df_truth[name].to_numpy().shape)
        for i in range(1,self.class_num):
            y_temp = self.df_truth[name].to_numpy() >= i*15
            y_temp = y_temp + [0]*y_temp.shape[0]
            y = y + y_temp
        
        lda = LinearDiscriminantAnalysis()

        X_lda = lda.fit(self.features, y).transform(self.features)
        
        return X_lda, lda, self.df_truth
    

if __name__ == '__main__':
    filename = "E:/data mining/final_project/data/train_2019.csv"
    lda = LDA_transform(filename, class_num=3, dep_or_arr="dep")
    
    X_lda, model, df_truth = lda.LDA()
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_lda[:, 0], X_lda[:, 1], df_truth["ARR_DELAY_NEW"].to_numpy())

    ax.view_init(-140, 60)