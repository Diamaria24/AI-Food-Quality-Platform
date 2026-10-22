import os
import struct


# ======================================================================
# TOMATO RAW HYPERSPECTRAL MAT FILE INSPECTION
# ======================================================================

print("=" * 70)
print("TOMATO RAW HYPERSPECTRAL MAT FILE INSPECTION")
print("=" * 70)


PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"


INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "tomato_hsi"
)


# ======================================================================
# CHECK DIRECTORY
# ======================================================================

print("\n" + "=" * 70)
print("DATASET DIRECTORY")
print("=" * 70)


if not os.path.isdir(INPUT_DIR):

    raise FileNotFoundError(
        f"Tomato HSI directory not found:\n{INPUT_DIR}"
    )


print("\nDirectory:")
print(INPUT_DIR)


# ======================================================================
# FIND MAT FILES
# ======================================================================

mat_files = sorted(
    [
        os.path.join(INPUT_DIR, filename)
        for filename in os.listdir(INPUT_DIR)
        if filename.lower().endswith(".mat")
    ]
)


print("\nMAT files found:", len(mat_files))


if len(mat_files) == 0:

    raise FileNotFoundError(
        "No .mat files found."
    )


for filepath in mat_files:

    size_mb = (
        os.path.getsize(filepath)
        /
        (1024 ** 2)
    )

    print(
        f"{os.path.basename(filepath):40s}"
        f"{size_mb:12.2f} MB"
    )


# ======================================================================
# DETERMINE MAT FILE FORMAT
# ======================================================================

def detect_mat_format(filepath):

    with open(
        filepath,
        "rb"
    ) as file:

        header = file.read(128)


    # MATLAB v7.3 files use HDF5.

    if header[:8] == b"\x89HDF\r\n\x1a\n":

        return "MATLAB v7.3 / HDF5"


    # Older MATLAB MAT files normally contain
    # a MATLAB 5/7 header.

    if b"MATLAB" in header:

        return "MATLAB 5/7 MAT"


    return "Unknown"


# ======================================================================
# INSPECT FILE HEADERS
# ======================================================================

print("\n" + "=" * 70)
print("MAT FILE FORMAT")
print("=" * 70)


file_formats = {}


for filepath in mat_files:

    fmt = detect_mat_format(
        filepath
    )


    file_formats[
        os.path.basename(filepath)
    ] = fmt


    print(
        f"{os.path.basename(filepath):40s} : {fmt}"
    )


# ======================================================================
# TRY SCIPY STRUCTURE INSPECTION
# ======================================================================

print("\n" + "=" * 70)
print("MATLAB VARIABLE INSPECTION")
print("=" * 70)


try:

    from scipy.io import whosmat

    scipy_available = True

except ImportError:

    scipy_available = False


if scipy_available:

    for filepath in mat_files:

        filename = os.path.basename(
            filepath
        )


        print("\n" + "-" * 70)

        print(
            "FILE:",
            filename
        )


        if file_formats[
            filename
        ] == "MATLAB 5/7 MAT":

            try:

                variables = whosmat(
                    filepath
                )


                print(
                    "\nVariables:"
                )


                for variable in variables:

                    name = variable[0]

                    shape = variable[1]

                    matlab_type = variable[2]


                    print(
                        f"  Name : {name}"
                    )

                    print(
                        f"  Shape: {shape}"
                    )

                    print(
                        f"  Type : {matlab_type}"
                    )

                    print()


            except Exception as error:

                print(
                    "Unable to inspect with scipy:"
                )

                print(
                    error
                )

        else:

            print(
                "Skipping scipy whosmat because "
                "this appears to be MATLAB v7.3/HDF5."
            )

else:

    print(
        "SciPy is not installed."
    )


# ======================================================================
# TRY HDF5 STRUCTURE INSPECTION
# ======================================================================

print("\n" + "=" * 70)
print("HDF5 STRUCTURE INSPECTION")
print("=" * 70)


try:

    import h5py

    h5py_available = True

except ImportError:

    h5py_available = False


if h5py_available:

    for filepath in mat_files:

        filename = os.path.basename(
            filepath
        )


        if file_formats[
            filename
        ] != "MATLAB v7.3 / HDF5":

            continue


        print("\n" + "-" * 70)

        print(
            "FILE:",
            filename
        )


        try:

            with h5py.File(
                filepath,
                "r"
            ) as h5file:


                def show_item(
                    name,
                    obj
                ):

                    if isinstance(
                        obj,
                        h5py.Dataset
                    ):

                        print(
                            f"DATASET : {name}"
                        )

                        print(
                            f"SHAPE   : {obj.shape}"
                        )

                        print(
                            f"DTYPE   : {obj.dtype}"
                        )


                    elif isinstance(
                        obj,
                        h5py.Group
                    ):

                        print(
                            f"GROUP   : {name}"
                        )


                h5file.visititems(
                    show_item
                )


        except Exception as error:

            print(
                "Unable to inspect HDF5 structure:"
            )

            print(
                error
            )

else:

    print(
        "h5py is not installed."
    )


# ======================================================================
# OPTIONAL DEPENDENCY CHECK
# ======================================================================

print("\n" + "=" * 70)
print("DEPENDENCY CHECK")
print("=" * 70)


print(
    "\nSciPy available:",
    scipy_available
)


print(
    "h5py available:",
    h5py_available
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("TOMATO MAT INSPECTION COMPLETED")
print("=" * 70)


print("""
No hyperspectral cube was modified.

No PCA was performed.

No classifier was trained.

No train/test split was created.

This was only a structural inspection of
the original Tomato MATLAB hyperspectral files.

NEXT STEP:

Use the variable names and cube dimensions
reported above to build the correct spectral
signature extraction pipeline.

The final pipeline will extract:

    wavelength / spectral-band axis
            ↓
    hyperspectral pixel spectra
            ↓
    Tomato class
            ↓
    class-wise mean spectral signature
            ↓
    spectral variability
            ↓
    spectral plots

IMPORTANT:

Do not manually guess the MATLAB variable name
or cube dimensions.
""")


print("=" * 70)