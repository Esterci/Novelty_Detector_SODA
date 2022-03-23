import pandas as pd

dataset = pd.read_csv("classification_base.csv").values

from anomaly_detector import SODA_detector

model = SODA_detector()

model.fit(dataset,3)

result = model.predict(dataset[0:5])
