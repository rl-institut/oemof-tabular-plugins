from dataclasses import dataclass
from oemof.tabular.constraint_facades import ConstraintFacade
from .constraints import renewable_share_minimum

@dataclass
class MinimumRenewableShare(ConstraintFacade):
    name: str
    type: str
    limit: float
    keyword: str = "renewable_factor"

    def build_constraint(self, model):
        # to use the constraints in oemof.solph, we need to pass the model.
        # Check if there are flows with the keyword attribute
        flows = {}
        for i, o in model.flows:
            if hasattr(model.flows[i, o], self.keyword):
                flows[(i, o)] = model.flows[i, o]

        if not flows:
            raise Warning(f"No flows with keyword {self.keyword}")
        #else:
            #print(f"These flows will contribute to the integral limit: {flows.keys()}")

        # Add constraint to the model
        renewable_share_minimum(model, flows=flows, limit=self.limit)

CONSTRAINT_TYPE_MAP = {"minimum_renewable_share": MinimumRenewableShare}