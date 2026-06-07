from src.stega import encode_image, decode_image

class DecodeTensor:
    def __init__(self, bbox=None):
        self.bbox = bbox

    def __call__(self, tensor):
        # Decode the image using the provided bounding box
        decoded_image = decode_image.decode(tensor, bbox=self.bbox)
        return decoded_image

