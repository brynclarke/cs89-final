from keras.models import Model
from keras import layers, activations, regularizers

class BaseModel(Model):
    def __init__(self, config):
        self.config = self._check_config(config)
        self.tensors = []
        self.model = None
        self.inputs = []
        self.outputs = []
        self.topology = self.config["topology"]
    
    def _init_model(self):
        super().__init__(self.tensors[0], self.tensors[-1])

    def _parse_activation(self, activation_char):
        if activation_char == "r":
            return activations.relu
        elif activation_char == "l":
            return activations.linear
        elif activation_char == "s":
            return activations.sigmoid

    def _check_config(self, config):
        if isinstance(config, str):
            return self._read_config(config)
        else:
            assert isinstance(config, dict)
            return config

    def _read_config(self, config_path):
        import json

        with open(config_path, "r") as fp:
            return json.load(fp)

    def save_config(self, config_path):
        import json

        with open(config_path, "w") as fp:
            json.dump(self.config, fp, sort_keys=True, indent=4)

class AutoEncoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        tensor = layers.Input(shape=(int(self.topology[0]),))
        self.tensors.append(tensor)

        for topo in self.topology[1:]:
            self._add(topo)

        super()._init_model()

    def _add(self, topo):
        n_nodes = int(topo[:-1])
        act = self._parse_activation(topo[-1])
        prev_tensor = self.tensors[-1]
        layer = layers.Dense(n_nodes, activation=act)
        self.tensors.append(layer(prev_tensor))

class MultiLayerPerceptron(BaseModel):
    def __init__(self, mlp_config, enc_config, enc_weights):
        super().__init__(mlp_config)

        self.dropout = self.config["dropout"]
        self.dropout_input = self.config["dropout_input"]
        self.L2_reg = self.config["L2_reg"]

        self.encoder = AutoEncoder(enc_config)
        self.encoder.load_weights(enc_weights)
        self.tensors.append(self.encoder.tensors[0])

        for layer in self.encoder.layers[1:-1]:
            layer.trainable = False
        concat = layers.concatenate(self.encoder.tensors[1:-1])
        #assert concat.shape == self.topology[0]
        self.tensors.append(concat)

        if self.dropout_input is not None:
            prev_tensor = self.tensors[-1]
            layer = layers.Dropout(self.dropout_input)
            self.tensors.append(layer(prev_tensor))

        for topo in self.topology[1:-1]:
            self._add(topo)
        self._add(self.topology[-1], output_layer=True)

        super()._init_model()

    def _add(self, topo, output_layer=False):
        n_nodes = int(topo[:-1])
        activation_char = topo[-1]
        act = self._parse_activation(activation_char)
        if self.L2_reg is not None and not output_layer:
            kernel_reg = regularizers.l2(self.L2_reg)
        else:
            kernel_reg = None
        prev_tensor = self.tensors[-1]
        layer = layers.Dense(n_nodes, activation=act, kernel_regularizer=kernel_reg)
        self.tensors.append(layer(prev_tensor))
        
        if self.dropout is not None and not output_layer:
            prev_tensor = self.tensors[-1]
            layer = layers.Dropout(self.dropout)
            self.tensors.append(layer(prev_tensor))