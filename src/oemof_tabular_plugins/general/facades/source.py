from oemof.solph.buses import Bus
from oemof.solph.components import Source
from oemof.solph.flows import Flow

from oemof.tabular._facade import Facade, dataclass_facade


@dataclass_facade
class SimpleSource(Source, Facade):
    r"""Bare Source element with one output. This class can be used to model
    Sun or rain when connected to a multi-input, multi-output converter (MIMO)
    with adequate conversion factors


    Parameters
    ----------
    bus: oemof.solph.Bus
        An oemof bus instance where the source is connected to
    """

    bus: Bus

    carrier: str = ""

    tech: str = ""

    def build_solph_components(self):
        """ """

        self.outputs.update({self.bus: Flow()})
