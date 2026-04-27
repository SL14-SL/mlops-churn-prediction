from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DecisionConfig:
    # Business values
    customer_value: float
    cost_discount: float
    cost_contact: float

    # Effectiveness
    discount_uplift: float  # probability of saving a churner
    contact_uplift: float
    max_discount_budget: float
    max_discount_rate: float = 0.2
    

    @classmethod
    def from_config(cls, cfg: dict):
        decision_cfg = cfg.get("decision", {})

        return cls(
            customer_value=float(decision_cfg.get("customer_value", 100)),
            cost_discount=float(decision_cfg.get("cost_discount", 10)),
            cost_contact=float(decision_cfg.get("cost_contact", 2.0)),
            discount_uplift=float(decision_cfg.get("discount_uplift", 0.3)),
            contact_uplift=float(decision_cfg.get("contact_uplift", 0.1)),
            max_discount_rate=float(decision_cfg.get("max_discount_rate",0.2)),
            max_discount_budget= float(decision_cfg.get("max_discount_budget", 10))
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
    
    def decide_batch(self, probs: list[float]) -> list[dict]:
        """
        Budget-aware decision using incremental value (uplift).
        """

        candidates = []

        # 1. Compute values per customer
        for i, p in enumerate(probs):
            v_discount = self._value_discount(p)
            v_email = self._value_contact(p)
            v_none = self._value_no_action(p)

            fallback_value = max(v_email, v_none)
            fallback_action = "send_email" if v_email >= v_none else "no_action"

            uplift = v_discount - fallback_value

            candidates.append({
                "idx": i,
                "p": float(p),
                "v_discount": v_discount,
                "fallback_action": fallback_action,
                "fallback_value": fallback_value,
                "uplift": uplift,
            })

        # 2. Sort by uplift (NOT probability!)
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x["uplift"],
            reverse=True
        )

        # 3. Budget constraint 
        if self.config.max_discount_budget:
            budget = self.config.max_discount_budget
            cost_per_discount = self.config.cost_discount

            used_budget = 0
            selected = set()

            for c in candidates_sorted:
                if c["uplift"] <= 0:
                    continue

                if used_budget + cost_per_discount > budget:
                    break

                selected.add(c["idx"])
                used_budget += cost_per_discount

        elif self.config.max_discount_rate:
            max_discount = int(len(probs) * self.config.max_discount_rate)
            selected = {
                c["idx"] for c in candidates_sorted[:max_discount]
                if c["uplift"] > 0
            }

        # 4. Final decisions
        results = []

        for c in candidates:
            if c["idx"] in selected:
                action = "offer_discount"
                value = c["v_discount"]
            else:
                action = c["fallback_action"]
                value = c["fallback_value"]

            results.append({
                "churn_probability": c["p"],
                "action": action,
                "expected_value": float(value),
            })

        logger.info(f"Selected {len(selected)} discounts | budget_used={used_budget:.2f}")

        return results