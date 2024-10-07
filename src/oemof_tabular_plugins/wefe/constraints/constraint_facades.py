import logging
from dataclasses import dataclass
from oemof.solph.components import Source
from oemof.solph.constraints import equate_variables
from oemof.tabular.constraint_facades import ConstraintFacade


@dataclass
class EqualSolarResource(ConstraintFacade):
    name: str
    type: str

    def build_constraint(self, model):
        # to use the constraints in oemof.solph, we need to pass the model.
        # Check if there are flows with the keyword attribute

        crops = [n for n in model.nodes if n.type in ("pv-panel", "crop", "mimo-crop")]
        for crop in crops:
            if crop.type == "crop":
                invest_bus = crop.harvest_bus
                component_capacity = model.InvestmentFlowBlock.invest[
                    crop, invest_bus, 0
                ]
                solar_bus = crop.solar_bus
            elif crop.type == "pv-panel":
                invest_bus = crop.to_bus
                component_capacity = model.InvestmentFlowBlock.invest[
                    crop, invest_bus, 0
                ]
                solar_bus = crop.from_bus

            elif crop.type == "mimo-crop":
                if crop.expandable is True:
                    invest_bus = model.es.groups[crop.primary]
                    component_capacity = model.InvestmentFlowBlock.invest[
                        invest_bus, crop, 0
                    ]
                else:
                    component_capacity = None
                solar_bus = crop.solar_energy_bus

            solar_nodes = [n for n in solar_bus.inputs if n.tech == "source"]

            solar_radiation = solar_nodes[0]
            if crop.expandable is True and solar_radiation.expandable is True:
                try:
                    # Add constraint to the model
                    equate_variables(
                        model,
                        model.InvestmentFlowBlock.invest[solar_radiation, solar_bus, 0],
                        component_capacity,
                    )
                except KeyError:
                    logging.error(
                        f"The equate_variable constraints cannot be set between crop '{crop.label}' and {solar_radiation.label}, please double check your input csv file"
                    )
            else:
                logging.error(
                    f"You wish to set an equality constraints between area of crop '{crop.label}' and {solar_radiation.label} resources. "
                    f"However both need to have 'expandable' set to 'True' in their respective csv input file.\n"
                    f"Currently '{crop.label}' has expandable set to {str(crop.expandable)} and '{solar_radiation.label}' has expandable set to {str(solar_radiation.expandable)}"
                )


CONSTRAINT_TYPE_MAP = {"equal_solar_resource": EqualSolarResource}
