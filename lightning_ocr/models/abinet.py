from .encoder_decoder_recognizer import EncoderDecoderRecognizer
from .backbones.resnet_abi import ResNetABI
from .encoders.abi_encoder import ABIEncoder
from .decoders.abi_vision_decoder import ABIVisionDecoder

class ABINetVision(EncoderDecoderRecognizer):
    def __init__(
        self,
        config : dict
    ):
        super().__init__(
            ResNetABI(**config.get("backbone", {})), 
            ABIEncoder(**config.get("encoder", {})), 
            ABIVisionDecoder(**config.get("decod    er", {})), 
        )
    