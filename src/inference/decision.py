from dataclasses import dataclass


@dataclass
class DecisionConfig:
    high_risk_threshold: float
    medium_risk_threshold: float
    cost_discount: float
    cost_contact: float
    customer_value: float

    @classmethod
    def from_config(cls, cfg: dict):
        decision_cfg = cfg.get("decision", {})

        return cls(
            high_risk_threshold=float(decision_cfg.get("high_risk_threshold", 0.7)),
            medium_risk_threshold=float(decision_cfg.get("medium_risk_threshold", 0.4)),
            cost_discount=float(decision_cfg.get("cost_discount", 10)),
            cost_contact=float(decision_cfg.get("cost_contact", 2.0)),
            customer_value=float(decision_cfg.get("customer_value", 100)),
        )

    def __post_init__(self):
        if not (0 <= self.medium_risk_threshold <= self.high_risk_threshold <= 1):
            raise ValueError("Invalid threshold configuration")


class DecisionEngine:
    def __init__(self, config: DecisionConfig):
        self.config = config

    def decide(self, churn_probability: float) -> dict:
        segment = self._segment(churn_probability)
        action = self._action(segment)
        expected_value = self._expected_value(churn_probability, action)

        return {
            "churn_probability": float(churn_probability),
            "segment": segment,
            "action": action,
            "expected_value": float(expected_value),
        }

    def _segment(self, p: float) -> str:
        if p >= self.config.high_risk_threshold:
            return "high_risk"
        elif p >= self.config.medium_risk_threshold:
            return "medium_risk"
        return "low_risk"

    def _action(self, segment: str) -> str:
        if segment == "high_risk":
            return "offer_discount"
        elif segment == "medium_risk":
            return "send_email"
        return "no_action"

    def _expected_value(self, p: float, action: str) -> float:
        if action == "offer_discount":
            return (p * self.config.customer_value) - self.config.cost_discount

        if action == "send_email":
            return (p * self.config.customer_value * 0.5) - self.config.cost_contact

        return 0.0