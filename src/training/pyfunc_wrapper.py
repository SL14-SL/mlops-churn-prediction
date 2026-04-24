import mlflow.pyfunc
import pandas as pd


class ChurnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame):
        # IMMER probability zurückgeben
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(model_input)[:, 1]

        return self.model.predict(model_input)