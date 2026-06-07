import numpy as np
from PIL import Image
from typing import IO, Iterator, Union
from functools import reduce
from typing import IO, List, Union

from PIL import Image

ENCODINGS = {"UTF-8": 8, "UTF-32LE": 32}

def a2bits_list(chars: str, encoding: str = "UTF-8") -> List[str]:
    """Convert a string to its bits representation as a list of 0's and 1's.

    >>>  a2bits_list("Hello World!")
    ['01001000',
    '01100101',
    '01101100',
    '01101100',
    '01101111',
    '00100000',
    '01010111',
    '01101111',
    '01110010',
    '01101100',
    '01100100',
    '00100001']
    >>> "".join(a2bits_list("Hello World!"))
    '010010000110010101101100011011000110111100100000010101110110111101110010011011000110010000100001'
    """
    return [bin(ord(x))[2:].rjust(ENCODINGS[encoding], "0") for x in chars]



def setlsb(component: int, bit: str) -> int:
    """Set Least Significant Bit of a colour component."""
    return component & ~1 | int(bit)



def open_image(fname_or_instance: Union[str, IO[bytes]]):
    """Opens a Image and returns it.

    :param fname_or_instance: Can either be the location of the image as a
                              string or the Image.Image instance itself.
    """
    if isinstance(fname_or_instance, Image.Image):
        return fname_or_instance

    return Image.open(fname_or_instance)


class Hider:
    def __init__(
        self,
        input_image: Union[str, IO[bytes]],
        message: str,
        encoding: str = "UTF-8",
        auto_convert_rgb: bool = False,
        bbox: Union[list, tuple] = None
    ):
        self._index = 0

        message_length = len(message)
        assert message_length != 0, "message length is zero"

        # image = open_image(input_image)
        image = input_image

        if image.mode not in ["RGB", "RGBA", "L"]:
            if not auto_convert_rgb:
                print(f"The mode of the image is not RGB. Mode is {image.mode}")
                answer = input("Convert the image to RGB ? [Y / n]\n") or "Y"
                if answer.lower() == "n":
                    raise Exception("Not a RGB image.")

            image = image.convert("RGB")
        if image.mode == "L":
            self.n_channels = 1
        elif image.mode == "RGB":
            self.n_channels = 3

        self.encoded_image = image.copy()
        # image.close()

        message = str(message_length) + ":" + str(message)
        self._message_bits = "".join(a2bits_list(message, encoding))
        self._message_bits += "0" * ((3 - (len(self._message_bits) % 3)) % 3)

        if bbox:
            ulx, uly, lrx, lry = bbox
            width = lrx - ulx + 1
            height = lry - uly + 1
        else:
            width, height = self.encoded_image.size
        npixels = width * height
        self._len_message_bits = len(self._message_bits)

        if self._len_message_bits > npixels * self.n_channels:
            raise Exception(
                f"The message you want to hide is too long: {message_length}"
            )

    def encode_another_pixel(self):
        return True if self._index + self.n_channels <= self._len_message_bits else False

    def encode_pixel(self, coordinate: tuple):
        # Get the colour component.
        if self.n_channels == 1:
            r = self.encoded_image.getpixel(coordinate)
            r = setlsb(r, self._message_bits[self._index])
        else:
            r, g, b, *a = self.encoded_image.getpixel(coordinate)
            # Change the Least Significant Bit of each colour component.
            r = setlsb(r, self._message_bits[self._index])
            g = setlsb(g, self._message_bits[self._index + 1])
            b = setlsb(b, self._message_bits[self._index + 2])


        # Save the new pixel
        if self.encoded_image.mode == "RGBA":
            self.encoded_image.putpixel(coordinate, (r, g, b, *a))
        elif self.encoded_image.mode == "RGB":
            self.encoded_image.putpixel(coordinate, (r, g, b))
        else:
            self.encoded_image.putpixel(coordinate, r)

        self._index += self.n_channels

class NumpyImage:
    def __init__(self, arr):
        self.im = arr
        self.height = arr.shape[1]
        self.width = arr.shape[2]
        self.mode = "RGB"

    def getpixel(self, coordinate):
        return self.im[:, coordinate[1], coordinate[0]]

    def close(self):
        return

class Revealer:
    def __init__(self, encoded_image: Union[str, IO[bytes], np.array], encoding: str = "UTF-8"):
        if type(encoded_image) == str:
            self.encoded_image = open_image(encoded_image)
        elif isinstance(encoded_image, Image.Image):
            self.encoded_image = encoded_image
        else:
            self.encoded_image = NumpyImage(encoded_image)
        self._encoding_length = ENCODINGS[encoding]
        self._buff, self._count = 0, 0
        self._bitab: List[str] = []
        self._limit: Union[None, int] = None
        self.secret_message = "",

    def decode_pixel(self, coordinate: tuple):
        # pixel = [r, g, b] or [r,g,b,a]
        pixel = self.encoded_image.getpixel(coordinate)

        if self.encoded_image.mode == "RGBA":
            pixel = pixel[:3]  # ignore the alpha
        elif self.encoded_image.mode == "L":
            pixel = [pixel]
            

        for color in pixel:
            self._buff += (color & 1) << (self._encoding_length - 1 - self._count)
            self._count += 1

            if self._count == self._encoding_length:
                self._bitab.append(chr(self._buff))
                self._buff, self._count = 0, 0

                if self._bitab[-1] == ":" and self._limit is None:
                    if "".join(self._bitab[:-1]).isdigit():
                        self._limit = int("".join(self._bitab[:-1]))
                    else:
                        return None
                        # raise IndexError("Impossible to detect message.")

        if len(self._bitab) - len(str(self._limit)) - 1 == self._limit:
            self.secret_message = "".join(self._bitab)[
                len(str(self._limit)) + 1 :  # noqa: E203
            ]
            # self.encoded_image.close()

            return True

        else:
            return False


def identity() -> Iterator[int]:
    """f(x) = x"""
    n = 0
    while True:
        yield n
        n += 1

def hide(
    image: Union[str, IO[bytes]],
    message: str,
    generator: Union[None, Iterator[int]] = None,
    shift: int = 0,
    encoding: str = "UTF-8",
    auto_convert_rgb: bool = False,
    bbox: Union[str, tuple] = None
):
    """Hide a message (string) in an image with the
    LSB (Least Significant Bit) technique.
    """
    hider = Hider(image, message, encoding, auto_convert_rgb, bbox)
    if bbox:
        ulx, uly, lrx, lry = bbox
        width = lrx - ulx + 1
    else:
        ulx, uly = 0, 0
        width = hider.encoded_image.width

    if not generator:
        generator = identity()

    while shift != 0:
        next(generator)
        shift -= 1

    while hider.encode_another_pixel():
        generated_number = next(generator)

        col = generated_number % width + ulx
        row = int(generated_number / width) + uly

        hider.encode_pixel((col, row))

    return hider.encoded_image

def reveal(
    encoded_image: Union[str, IO[bytes], np.array],
    generator: Union[None, Iterator[int]] = None,
    shift: int = 0,
    encoding: str = "UTF-8",
    bbox: Union[list, tuple] = None
):
    """Find a message in an image (with the LSB technique)."""
    revealer = Revealer(encoded_image, encoding)
    if bbox:
        ulx, uly, lrx, lry = bbox
        width = lrx - ulx + 1
    else:
        ulx, uly, lrx, lry = 0, 0, revealer.encoded_image.width-1, revealer.encoded_image.height-1
        width = revealer.encoded_image.width

    if not generator:
        generator = identity()

    while shift != 0:
        next(generator)
        shift -= 1

    col, row = ulx, uly
    while row < lry:
        generated_number = next(generator)
        col = generated_number % width + ulx
        row = int(generated_number / width) + uly
        if revealer.decode_pixel((col, row)):
            return revealer.secret_message
