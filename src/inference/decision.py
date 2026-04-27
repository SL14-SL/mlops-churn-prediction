from dataclasses import dataclass


@dataclass
class DecisionConfig:
    # Business values
    customer_value: float
    cost_discount: float
    cost_contact: float

    # Effectiveness
    discount_uplift: float  # probability of saving a churner
    contact_uplift: float

    @classmethod
    def from_config(cls, cfg: dict):
        decision_cfg = cfg.get("decision", {})

        return cls(
            customer_value=float(decision_cfg.get("customer_value", 100)),
            cost_discount=float(decision_cfg.get("cost_discount", 10)),
            cost_contact=float(decision_cfg.get("cost_contact", 2.0)),
            discount_uplift=float(decision_cfg.get("discount_uplift", 0.3)),
            contact_uplift=float(decision_cfg.get("contact_uplift", 0.1)),
        )


class DecisionEngine:
    def __init__(self, config: DecisionConfig):
        self.config = config

    def decide(self, p: float) -> dict:
        """
        Choose best action based on expected value.
        """

        actions = {
            "offer_discount": self._value_discount(p),
            "send_email": self._value_contact(p),
            "no_action": self._value_no_action(p),
        }

        # 🔥 Pick best action
        best_action = max(actions, key=actions.get)
        best_value = actions[best_action]

        return {
            "churn_probability": float(p),
            "action": best_action,
            "expected_value": float(best_value),
            "all_actions": actions,  # optional debug 👀
        }

    # ---- VALUE FUNCTIONS ----

    def _value_discount(self, p: float) -> float:
        return (
            p * (self.config.discount_uplift * self.config.customer_value)
            - self.config.cost_discount
        )

    def _value_contact(self, p: float) -> float:
        return (
            p * (self.config.contact_uplift * self.config.customer_value)
            - self.config.cost_contact
        )

    def _value_no_action(self, p: float) -> float:
        return 0.0