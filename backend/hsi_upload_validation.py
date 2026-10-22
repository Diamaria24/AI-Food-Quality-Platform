
from pathlib import Path
import re

ENVI_DATA_TYPES = {
    1: 1,   # uint8
    2: 2,   # int16
    3: 4,   # int32
    4: 4,   # float32
    5: 8,   # float64
    12: 2,  # uint16
    13: 4,  # uint32
    14: 8,  # int64
    15: 8,  # uint64
}

MAX_PIXELS = 10_000_000


def validate_envi_upload(header_path, binary_path):
    header_path = Path(header_path)
    binary_path = Path(binary_path)

    text = header_path.read_text(encoding="utf-8-sig")

    if not re.match(r"^\s*ENVI\s*(?:\r?\n|$)", text, re.I):
        raise ValueError("Invalid ENVI header.")
    
    # Reject duplicate critical ENVI metadata fields.
    # Different parsers may interpret duplicates differently.
    critical_fields = (
        "samples",
        "lines",
        "bands",
        "data type",
        "interleave",
        "byte order",
        "header offset",
        "data file",
    )

    for field in critical_fields:
        matches = re.findall(
            rf"(?im)^\s*{re.escape(field)}\s*=",
            text,
        )

        if len(matches) > 1:
            raise ValueError(
                f"Duplicate ENVI metadata field: {field}"
            )


    def get_integer_field(name):
        pattern = rf"(?im)^\s*{re.escape(name)}\s*=\s*([+-]?\d+)\s*$"
        match = re.search(pattern, text)

        if not match:
            raise ValueError(
                f"Missing or invalid ENVI field: {name}"
            )

        return int(match.group(1))

    samples = get_integer_field("samples")
    lines = get_integer_field("lines")
    bands = get_integer_field("bands")
    data_type = get_integer_field("data type")
    header_offset = get_integer_field("header offset")
    byte_order = get_integer_field("byte order")

    interleave_match = re.search(
        r"(?im)^\s*interleave\s*=\s*(bil|bip|bsq)\s*$",
        text,
    )

    if not interleave_match:
        raise ValueError("Unsupported ENVI interleave.")

    if samples <= 0 or lines <= 0:
        raise ValueError("Invalid ENVI cube dimensions.")

    if bands != 277:
        raise ValueError(
            f"Expected 277 spectral bands, received {bands}."
        )

    if samples * lines > MAX_PIXELS:
        raise ValueError(
            "ENVI cube exceeds the supported pixel limit."
        )

    if data_type not in ENVI_DATA_TYPES:
        raise ValueError("Unsupported ENVI data type.")
    
    # Restrict uploads to the currently validated
    # Eggplant/Cabbage model input format.
    if data_type != 4:
        raise ValueError(
            "Only float32 ENVI data (type 4) is supported."
        )

    if interleave_match.group(1).lower() != "bil":
        raise ValueError(
            "Only BIL-interleaved ENVI data is supported."
        )

    if byte_order != 0:
        raise ValueError(
            "Only little-endian ENVI data is supported."
        )


    if header_offset != 0:
        raise ValueError(
            "Nonzero ENVI header offsets are not supported."
        )

    if byte_order not in (0, 1):
        raise ValueError("Invalid ENVI byte order.")

    bytes_per_value = ENVI_DATA_TYPES[data_type]

    expected_size = (
        samples * lines * bands * bytes_per_value
    )

    actual_size = binary_path.stat().st_size

    if actual_size != expected_size:
        raise ValueError(
            "ENVI binary size mismatch: "
            f"expected {expected_size} bytes, "
            f"received {actual_size} bytes."
        )

    return {
        "samples": samples,
        "lines": lines,
        "bands": bands,
        "data_type": data_type,
        "interleave": interleave_match.group(1).lower(),
        "byte_order": byte_order,
        "binary_bytes": actual_size,
    }
