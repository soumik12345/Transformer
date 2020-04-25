import tensorflow as tf
from src.blocks.encoding import visualize_pe
from src.models import Encoder

encoder = Encoder(units=512, d_model=128, heads=4, dropout=0.3, name="Encoder")
encoder.summary()
encoder.save('encoder.h5')