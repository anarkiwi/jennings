"""Observable memory: a 64 KiB store with per-address read/write callbacks.

py65-compatible ``ObservableMemory``: wrap a subject (list/bytearray/MPU memory)
and register callbacks fired on access to specific addresses, e.g. to trap SID
register writes or synthesise VIC/CIA reads.
"""

from collections import defaultdict


class ObservableMemory:
    def __init__(self, subject=None, addrWidth=16):
        self.physMask = 0x3FFFF if addrWidth > 16 else 0xFFFF
        if subject is None:
            subject = (self.physMask + 1) * [0x00]
        self._subject = subject
        self._read_subscribers = defaultdict(list)
        self._write_subscribers = defaultdict(list)

    def __setitem__(self, address, value):
        if isinstance(address, slice):
            for n, v in zip(range(*address.indices(self.physMask + 1)), value):
                self[n] = v
            return
        address &= self.physMask
        for callback in self._write_subscribers[address]:
            result = callback(address, value)
            if result is not None:
                value = result
        self._subject[address] = value

    def __getitem__(self, address):
        if isinstance(address, slice):
            return [self[n] for n in range(*address.indices(self.physMask + 1))]
        address &= self.physMask
        final = None
        for callback in self._read_subscribers[address]:
            result = callback(address)
            if result is not None:
                final = result
        return self._subject[address] if final is None else final

    def __getattr__(self, attribute):
        return getattr(self._subject, attribute)

    def subscribe_to_write(self, address_range, callback):
        for address in address_range:
            callbacks = self._write_subscribers[address & self.physMask]
            if callback not in callbacks:
                callbacks.append(callback)

    def subscribe_to_read(self, address_range, callback):
        for address in address_range:
            callbacks = self._read_subscribers[address & self.physMask]
            if callback not in callbacks:
                callbacks.append(callback)

    def write(self, start_address, bytes):
        start_address &= self.physMask
        self._subject[start_address : start_address + len(bytes)] = bytes
