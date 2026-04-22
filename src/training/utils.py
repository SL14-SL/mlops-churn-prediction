from src.utils.logger import get_logger
from src.configs.loader import load_config


logger = get_logger(__name__)

ENV_CFG = load_config()
TRAIN_CFG = load_config("training.yaml")


def build_drop_columns(config: dict) -> list[str]:
    """
    Build feature drop list from config without duplicates.
    Ensures all names are lowercase to match the preprocessed data.
    """
   
    data_cfg = config.get("data", {})
    feature_cfg = config.get("features", {})

    target_column = data_cfg.get("target_column")
    known_targets = data_cfg.get("known_targets", [])
    time_column = data_cfg.get("time_column")
    configured_drop_columns = feature_cfg.get("drop_columns", [])

    raw_drop_list = configured_drop_columns + known_targets
    
    if target_column:
        raw_drop_list.append(target_column)
    if time_column:
        raw_drop_list.append(time_column)

   
    drop_columns = [col.lower().replace(" ", "_") for col in raw_drop_list if col]

    return list(dict.fromkeys(drop_columns))
