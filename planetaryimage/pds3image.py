# -*- coding: utf-8 -*-
import numpy
import six
import pvl
import collections

from .image import PlanetaryImage
from .decoders import BandSequentialDecoder


class Pointer(collections.namedtuple('Pointer', ['filename', 'bytes'])):
    @staticmethod
    def _parse_bytes(value, record_bytes):
        if isinstance(value, six.integer_types):
            return (value - 1) * record_bytes

        if isinstance(value, pvl.Units) and value.units == 'BYTES':
            return value.value

        raise ValueError('Unsupported pointer type')

    @classmethod
    def parse(cls, value, record_bytes):
        """Parses the pointer label.

        Parameters
        ----------
        pointer_data
            Supported values for `pointer_data` are::

                ^PTR = nnn
                ^PTR = nnn <BYTES>
                ^PTR = "filename"
                ^PTR = ("filename")
                ^PTR = ("filename", nnn)
                ^PTR = ("filename", nnn <BYTES>)

        record_bytes
            Record multiplier value

        Returns
        -------
        Pointer object
        """
        if isinstance(value, six.string_types):
            return cls(value, 0)

        if isinstance(value, list):
            if len(value) == 1:
                return cls(value[0], 0)

            if len(value) == 2:
                return cls(value[0], cls._parse_bytes(value[1], record_bytes))

            raise ValueError('Unsupported pointer type')

        return cls(None, cls._parse_bytes(value, record_bytes))


class PDS3Image(PlanetaryImage):
    """A PDS3 image reader.

    Examples
    --------

    >>> from planetaryimage import PDS3Image
    >>> image = PDS3Image.open('tests/mission_data/2p129641989eth0361p2600r8m1.img')
    >>> image
    tests/mission_data/2p129641989eth0361p2600r8m1.img
    >>> image.label['IMAGE']['LINES']
    64

    """

    SAMPLE_TYPES = {
        'MSB_INTEGER': '>i',
        'INTEGER': '>i',
        'MAC_INTEGER': '>i',
        'SUN_INTEGER': '>i',

        'MSB_UNSIGNED_INTEGER': '>u',
        'UNSIGNED_INTEGER': '>u',
        'MAC_UNSIGNED_INTEGER': '>u',
        'SUN_UNSIGNED_INTEGER': '>u',

        'LSB_INTEGER': '<i',
        'PC_INTEGER': '<i',
        'VAX_INTEGER': '<i',

        'LSB_UNSIGNED_INTEGER': '<u',
        'PC_UNSIGNED_INTEGER': '<u',
        'VAX_UNSIGNED_INTEGER': '<u',

        'IEEE_REAL': '>f',
        'FLOAT': '>f',
        'REAL': '>f',
        'MAC_REAL': '>f',
        'SUN_REAL': '>f',

        'IEEE_COMPLEX': '>c',
        'COMPLEX': '>c',
        'MAC_COMPLEX': '>c',
        'SUN_COMPLEX': '>c',

        'PC_REAL': '<f',
        'PC_COMPLEX': '<c',

        'MSB_BIT_STRING': '>S',
        'LSB_BIT_STRING': '<S',
        'VAX_BIT_STRING': '<S',
    }

    @property
    def _bands(self):
        return self.label['IMAGE'].get('BANDS', 1)

    @property
    def _lines(self):
        return self.label['IMAGE']['LINES']

    @property
    def _samples(self):
        return self.label['IMAGE']['LINE_SAMPLES']

    @property
    def _format(self):
        return self.label['IMAGE'].get('format', 'BAND_SEQUENTIAL')

    @property
    def _start_byte(self):
        return self._image_pointer.bytes

    @property
    def _data_filename(self):
        return self._image_pointer.filename

    @property
    def _dtype(self):
        return self._pixel_type.newbyteorder(self._byte_order)

    @property
    def record_bytes(self):
        """Number of bytes for fixed length records."""
        return self.label.get('RECORD_BYTES', 0)

    @property
    def _image_pointer(self):
        return Pointer.parse(self.label['^IMAGE'], self.record_bytes)

    @property
    def _sample_type(self):
        sample_type = self.label['IMAGE']['SAMPLE_TYPE']
        try:
            return self.SAMPLE_TYPES[sample_type]
        except KeyError:
            raise ValueError('Unsupported sample type: %r' % sample_type)

    @property
    def _sample_bytes(self):
        # get bytes to match NumPy dtype expressions
        return int(self.label['IMAGE']['SAMPLE_BITS'] / 8)

    # FIXME:  This dtype overrides the Image.dtype right?  Then whats the point
    # of _dtype above here ^^, should we just rename this one _dtype and remove
    # the other one?
    @property
    def dtype(self):
        """Pixel data type."""
        return numpy.dtype('%s%d' % (self._sample_type, self._sample_bytes))

    @property
    def _decoder(self):
        if self.format == 'BAND_SEQUENTIAL':
            return BandSequentialDecoder(
                self.dtype, self.shape, self.compression
            )
        raise ValueError('Unkown format (%s)' % self.format)
