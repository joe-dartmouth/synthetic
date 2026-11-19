"""The BKB vocabulary: the ontology, I-nodes and S-nodes, and the cell's data fragment."""

from video.layout import SectionScene


class Ontology(SectionScene):
    """The eight HPO classes as three is_a chains."""

    name = "Ontology"


class INodesAndSNodes(SectionScene):
    """Each class becomes an I-node and each is_a edge an S-node."""

    name = "INodesAndSNodes"


class DatasetFragment(SectionScene):
    """The cell's records become priors on the concepts they show."""

    name = "DatasetFragment"
