from pyomo import environ as po

def renewable_share_minimum(model, flows=None, limit=None):
    keyword = "renewable_factor"

    if flows is None:
        flows = {}
        for (i, o) in model.flows:
            if hasattr(model.flows[i, o], keyword):
                flows[(i, o)] = model.flows[i, o]
    else:
        for (i, o) in flows:
            if not hasattr(flows[i, o], keyword):
                raise AttributeError(
                    f"Flow with source: {i.label} and target: {o.label} has no attribute {keyword}."
                )
    # this structure only works if you don't use multi-period
    total_generation = sum(model.flow[i, o, 0, t] for (i, o) in flows for t in model.TIMESTEPS)
    renewable_generation = sum(
        model.flow[i, o, 0, t] * getattr(flows[i, o], keyword) for (i, o) in flows for t in model.TIMESTEPS
    )
    constraint_name = "renewable_share_minimum_constraint"
    model.add_component(constraint_name, po.Constraint(expr=(renewable_generation >= float(limit) * total_generation)))

    return model