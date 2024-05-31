from oemof.tabular.config import config

SPECIAL_FIELD_NAMES = {}
for fk, descriptor in config.FOREIGN_KEY_DESCRIPTORS.items():
    for el in descriptor:
        SPECIAL_FIELD_NAMES[el["fields"]] = fk
