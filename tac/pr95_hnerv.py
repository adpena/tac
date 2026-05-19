"""PR95-family HNeRV single-member archive wire helpers.

Public PR95/PR98-style HNeRV archives use one stored ZIP member named
``0.bin``. The member payload is three length-prefixed brotli blobs:
metadata, decoder weights, and uint8 latent rows. This module keeps that
wire grammar in ``tac`` so profilers, residual-atom planners, and replay tools
do not each carry their own parser.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import struct
import zipfile
from collections.abc import Sequence
from pathlib import Path

import brotli

FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


@dataclasses.dataclass(frozen=True)
class LatentPayload:
    n_pairs: int
    latent_dim: int
    mins_f16: bytes
    scales_f16: bytes
    quantized: tuple[tuple[int, ...], ...]

    def to_bytes(self) -> bytes:
        if len(self.quantized) != self.n_pairs:
            raise ValueError(f"expected {self.n_pairs} latent rows, got {len(self.quantized)}")
        if len(self.mins_f16) != self.latent_dim * 2:
            raise ValueError("mins_f16 length does not match latent_dim")
        if len(self.scales_f16) != self.latent_dim * 2:
            raise ValueError("scales_f16 length does not match latent_dim")
        previous = [0] * self.latent_dim
        lo = bytearray()
        hi = bytearray()
        for pair_index, row in enumerate(self.quantized):
            if len(row) != self.latent_dim:
                raise ValueError(f"row {pair_index} has {len(row)} dims, expected {self.latent_dim}")
            for dim_index, value in enumerate(row):
                ivalue = int(value)
                if not 0 <= ivalue <= 255:
                    raise ValueError(
                        f"latent quantized value out of uint8 range at pair {pair_index}, "
                        f"dim {dim_index}: {ivalue}"
                    )
                delta = ivalue if pair_index == 0 else ivalue - previous[dim_index]
                zz = delta * 2 if delta >= 0 else -2 * delta - 1
                lo.append(zz & 0xFF)
                hi.append((zz >> 8) & 0xFF)
                previous[dim_index] = ivalue
        return (
            struct.pack("<II", self.n_pairs, self.latent_dim)
            + self.mins_f16
            + self.scales_f16
            + bytes(lo)
            + bytes(hi)
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact(buf: io.BytesIO, n: int, label: str) -> bytes:
    data = buf.read(n)
    if len(data) != n:
        raise ValueError(f"truncated {label}: wanted {n}, got {len(data)}")
    return data


def read_single_member_zip(path: Path) -> tuple[str, bytes, dict[str, int | list[int]]]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected exactly one archive member, got {len(infos)}")
        info = infos[0]
        if info.filename != "0.bin":
            raise ValueError(f"PR95-family archive must contain exactly 0.bin, got {info.filename!r}")
        data = zf.read(info.filename)
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC validation failed for member {bad!r}")
        return (
            info.filename,
            data,
            {
                "compress_type": int(info.compress_type),
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "crc": int(info.CRC),
                "date_time": list(info.date_time),
            },
        )


def write_stored_zip(path: Path, member_name: str, payload: bytes) -> None:
    if member_name != "0.bin":
        raise ValueError(f"PR95-family archive member must be 0.bin, got {member_name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member_name, date_time=FIXED_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def parse_top_blob(blob: bytes) -> dict[str, bytes]:
    buf = io.BytesIO(blob)
    out: dict[str, bytes] = {}
    for label in ("meta", "decoder", "latents"):
        (size,) = struct.unpack("<I", _read_exact(buf, 4, f"{label}_brotli_len"))
        compressed = _read_exact(buf, size, f"{label}_brotli")
        out[f"{label}_brotli"] = compressed
        out[f"{label}_raw"] = brotli.decompress(compressed)
    rest = buf.read()
    if rest:
        raise ValueError(f"trailing bytes after PR95 blob: {len(rest)}")
    return out


def encode_top_blob(meta_brotli: bytes, decoder_brotli: bytes, latents_brotli: bytes) -> bytes:
    out = io.BytesIO()
    for payload in (meta_brotli, decoder_brotli, latents_brotli):
        out.write(struct.pack("<I", len(payload)))
        out.write(payload)
    return out.getvalue()


def parse_latents_raw(latents_raw: bytes) -> LatentPayload:
    buf = io.BytesIO(latents_raw)
    n_pairs, latent_dim = struct.unpack("<II", _read_exact(buf, 8, "latent header"))
    mins_f16 = _read_exact(buf, latent_dim * 2, "latent mins_f16")
    scales_f16 = _read_exact(buf, latent_dim * 2, "latent scales_f16")
    total = n_pairs * latent_dim
    lo = _read_exact(buf, total, "latent lo delta stream")
    hi = _read_exact(buf, total, "latent hi delta stream")
    rest = buf.read()
    if rest:
        raise ValueError(f"latent raw has trailing bytes: {len(rest)}")
    previous = [0] * latent_dim
    rows: list[tuple[int, ...]] = []
    for pair_index in range(n_pairs):
        row: list[int] = []
        for dim_index in range(latent_dim):
            offset = pair_index * latent_dim + dim_index
            zz = lo[offset] | (hi[offset] << 8)
            delta = zz // 2 if zz % 2 == 0 else -(zz // 2) - 1
            value = delta if pair_index == 0 else previous[dim_index] + delta
            if not 0 <= value <= 255:
                raise ValueError(
                    f"latent quantized value out of uint8 range at pair {pair_index}, "
                    f"dim {dim_index}: {value}"
                )
            row.append(value)
            previous[dim_index] = value
        rows.append(tuple(row))
    return LatentPayload(
        n_pairs=n_pairs,
        latent_dim=latent_dim,
        mins_f16=mins_f16,
        scales_f16=scales_f16,
        quantized=tuple(rows),
    )


def latent_rows(payload: LatentPayload) -> list[list[int]]:
    return [list(row) for row in payload.quantized]


def latent_payload_from_rows(source: LatentPayload, rows: Sequence[Sequence[int]]) -> LatentPayload:
    if len(rows) != source.n_pairs:
        raise ValueError(f"expected {source.n_pairs} latent rows, got {len(rows)}")
    checked: list[tuple[int, ...]] = []
    for pair_index, row in enumerate(rows):
        if len(row) != source.latent_dim:
            raise ValueError(f"pair {pair_index} expected latent_dim={source.latent_dim}, got {len(row)}")
        out_row: list[int] = []
        for dim_index, value in enumerate(row):
            ivalue = int(value)
            if not 0 <= ivalue <= 255:
                raise ValueError(
                    f"latent value out of uint8 range at pair {pair_index}, dim {dim_index}: {ivalue}"
                )
            out_row.append(ivalue)
        checked.append(tuple(out_row))
    return dataclasses.replace(source, quantized=tuple(checked))
