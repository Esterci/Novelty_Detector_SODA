import numpy as np
from SODA import SelfOrganisedDirectionAwareDataPartitioning


class SODA_detector:
    def __init__(self) -> None:
        pass

    def fit(self, data, granularity, distancetype="euclidean", percent=0.9):

        self.seed = data
        self.granularity = granularity
        self.distancetype = distancetype
        self.percent = percent

        # out = self.SelfOrganisedDirectionAwareDataPartitioning(
        #    data, granularity, distancetype
        # )

        # self.train_clouds = out["IDX"]

    def predict(self, data):

        import pandas as pd

        n_seed_samples = len(self.seed)
        prediction_data = np.vstack((self.seed, data))
        novelty_target = np.ones(len(prediction_data))
        novelty_target[:n_seed_samples] = 0

        out = SelfOrganisedDirectionAwareDataPartitioning(
            prediction_data, self.granularity, self.distancetype
        )

        IDX = out["IDX"]

        n_couds = np.max(IDX)
        clouds_info_number = pd.DataFrame(
            np.zeros((n_couds, 3)),
            columns=["seed_number", "novel_number", "total_number"],
        )

        for sample,label in zip(IDX,novelty_target):

            if label == 0:
                clouds_info_number.loc[int(sample-1)]["seed_number"] += 1

            else:
                clouds_info_number.loc[int(sample-1)]["novel_number"] += 1
            
            clouds_info_number.loc[int(sample-1)]["total_number"] += 1

            
        anomaly_clouds = []

        for ii in range(len(clouds_info_number)):

            clouds_info_number.loc[ii]["seed_number"] = (

                clouds_info_number.loc[ii]["seed_number"]/clouds_info_number.loc[ii]["total_number"]
            )

            clouds_info_number.loc[ii]["novel_number"] = (

                clouds_info_number.loc[ii]["novel_number"]/clouds_info_number.loc[ii]["total_number"]
            )

            if clouds_info_number.loc[ii]["novel_number"] >= self.percent:

                anomaly_clouds.append(1)

            else:

                anomaly_clouds.append(0)

        target = []

        for ii in range(n_seed_samples,len(IDX)):

            if IDX[ii] in anomaly_clouds:
                target.append(1) 

            else:
                target.append(0)

        return np.array(target)
