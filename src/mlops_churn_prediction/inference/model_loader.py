import mlflow.pyfunc

def load_model_by_type(model_uri: str, model_type: str):
    """
    Universal loader using pyfunc. 
    Works for xgboost, sklearn, etc. without needing separate loaders.
    """
    # model_type is kept for logging/logic, but pyfunc handles the heavy lifting
    return mlflow.pyfunc.load_model(model_uri)