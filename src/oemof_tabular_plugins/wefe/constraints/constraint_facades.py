import logging
from dataclasses import dataclass
from oemof.solph.constraints import equate_variables
from oemof.tabular.constraint_facades import ConstraintFacade


@dataclass
class EqualSolarResource(ConstraintFacade):
    name: str
    type: str

    def build_constraint(self, model):
        # to use the constraints in oemof.solph, we need to pass the model.
        # Check if there are flows with the keyword attribute

        crops = [n for n in model.nodes if n.type == "crop"]
        for crop in crops:
            harvest_bus = crop.harvest_bus
            solar_bus = crop.solar_bus
            solar_nodes = [n for n in solar_bus.inputs if n.tech == "source"]
            solar_radiation = solar_nodes[0]
            if crop.expandable is True and solar_radiation.expandable is True:
                try:
                    # Add constraint to the model
                    equate_variables(
                        model,
                        model.InvestmentFlowBlock.invest[crop, harvest_bus, 0],
                        model.InvestmentFlowBlock.invest[solar_radiation, solar_bus, 0],
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
