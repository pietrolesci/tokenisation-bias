from lightning.pytorch.callbacks import ModelCheckpoint as _ModelCheckpoint


class ModelCheckpoint(_ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = ""
