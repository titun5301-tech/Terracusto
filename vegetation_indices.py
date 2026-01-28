# ============================================================
# Vegetation indices + LAI from multiband GeoTIFF (simple version)
# - No local helper functions
# - Same indices + exports
# - White background + colormap where 0 = white (by using caxis min)
# ============================================================

import os
import numpy as np
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# If you use cmocean in Python:
# pip install cmocean
import cmocean


def main():
    # ============================================================
    # STEP 2) Add the TIFF file & quicklook
    # ============================================================
    tifPath = r"C:\Users\ushre\Downloads\Composite_GoodALC_only.tif"

    if not os.path.isfile(tifPath):
        raise FileNotFoundError(f"TIFF file not found: {tifPath}")

    with rasterio.open(tifPath) as ds:
        # Read all bands: shape -> (bands, rows, cols)
        A = ds.read().astype(np.float64)
        R_transform: Affine = ds.transform
        R_crs = ds.crs
        R_profile = ds.profile.copy()

        # Similar to MATLAB "info"
        print("Raster size:", A.shape)
        print("Coordinate system code (PCS if available): ")
        if R_crs is not None:
            try:
                # Often prints "EPSG:xxxx" if available
                print(R_crs)
            except Exception:
                print("CRS available but could not be printed.")
        else:
            print("PCS/CRS not available (varies by file).")

        # Try to detect NoData value (varies by GeoTIFF)
        noDataVal = None
        try:
            # Prefer dataset nodata
            if ds.nodata is not None:
                noDataVal = float(ds.nodata)
            else:
                # Try GDAL_NODATA tag (like MATLAB GeoTIFFTags.GDAL_NODATA)
                tags = ds.tags()
                if "GDAL_NODATA" in tags:
                    noDataVal = float(tags["GDAL_NODATA"])
        except Exception:
            noDataVal = None

    # Quicklook (band 1)
    plt.figure(num="GeoTIFF quicklook")
    if A.ndim == 3:
        # A is (bands, rows, cols)
        plt.imshow(A[0, :, :], interpolation="nearest")
        plt.axis("equal")
        plt.axis("off")
        plt.colorbar()
        plt.title("Quicklook (Band 1)")
    else:
        plt.imshow(A, interpolation="nearest")
        plt.axis("equal")
        plt.axis("off")
        plt.colorbar()
        plt.title("Quicklook (Single band)")
    plt.show()

    # ============================================================
    # STEP 3) Display bands (define band mapping)
    # ============================================================
    # ----------- BAND MAPPING (EDIT THIS TO MATCH YOUR TIFF) ----------
    # Example assumption: [1=Blue, 2=Green, 3=Red, 4=RedEdge, 5=NIR]
    # Python is 1-based in the "meaning" here, but array indices are 0-based.
    bBlue = 1
    bGreen = 2
    bRed = 3
    bRedEdge = 4  # if not available: set None and NDRE will be NaN
    bNIR = 5
    # ------------------------------------------------------------------

    if A.ndim != 3:
        raise ValueError(f"Your TIFF appears not to be multiband. A.ndim = {A.ndim}")

    nBands, nRows, nCols = A.shape
    print("Number of bands detected:", nBands)

    reqBands = [bBlue, bGreen, bRed, bNIR]
    if any(b > nBands for b in reqBands):
        raise ValueError("Band mapping points to a band index that doesn't exist. "
                         "Check bBlue/bGreen/bRed/bNIR.")

    # Extract bands (convert 1-based band numbers to 0-based indices)
    BLUE = A[bBlue - 1, :, :]
    GREEN = A[bGreen - 1, :, :]
    RED = A[bRed - 1, :, :]
    NIR = A[bNIR - 1, :, :]

    if bRedEdge is not None:
        if bRedEdge > nBands:
            raise ValueError("bRedEdge is out of range. Set bRedEdge=None if your TIFF has no red-edge band.")
        RE = A[bRedEdge - 1, :, :]
    else:
        RE = None

    # Apply NoData masking (if detected)
    if noDataVal is not None and not np.isnan(noDataVal):
        maskNoData = (BLUE == noDataVal) | (GREEN == noDataVal) | (RED == noDataVal) | (NIR == noDataVal)
        if RE is not None:
            maskNoData = maskNoData | (RE == noDataVal)

        BLUE = BLUE.copy()
        GREEN = GREEN.copy()
        RED = RED.copy()
        NIR = NIR.copy()
        BLUE[maskNoData] = np.nan
        GREEN[maskNoData] = np.nan
        RED[maskNoData] = np.nan
        NIR[maskNoData] = np.nan
        if RE is not None:
            RE = RE.copy()
            RE[maskNoData] = np.nan

    # Optional scaling (if your reflectance is 0..10000)
    # scaleFactor = 10000.0
    # BLUE  = BLUE  / scaleFactor
    # GREEN = GREEN / scaleFactor
    # RED   = RED   / scaleFactor
    # NIR   = NIR   / scaleFactor
    # if RE is not None:
    #     RE = RE / scaleFactor

    # Band diagnostics
    print("---- Band diagnostics (finite pixels) ----")
    for k in range(nBands):
        band = A[k, :, :]
        finite_mask = np.isfinite(band)
        finite_count = int(np.count_nonzero(finite_mask))
        total_count = band.size
        band_min = np.nanmin(band) if finite_count > 0 else np.nan
        band_max = np.nanmax(band) if finite_count > 0 else np.nan
        print(f"Band {k + 1}: min={band_min:g} max={band_max:g} finite={finite_count}/{total_count}")

    # Display bands
    fig = plt.figure(num="Band display")
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(BLUE, interpolation="nearest")
    ax.set_title("Blue")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(GREEN, interpolation="nearest")
    ax.set_title("Green")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(RED, interpolation="nearest")
    ax.set_title("Red")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(NIR, interpolation="nearest")
    ax.set_title("NIR")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 1])
    if RE is not None:
        im = ax.imshow(RE, interpolation="nearest")
        ax.set_title("RedEdge")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.set_title("RedEdge not provided")
        ax.set_axis_off()

    fig.add_subplot(gs[1, 2]).set_axis_off()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # STEP 4) Calculate vegetation indices (per pixel)
    # ============================================================
    epsVal = 1e-10

    NDVI = (NIR - RED) / (NIR + RED + epsVal)
    EVI = 2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0 + epsVal)
    GNDVI = (NIR - GREEN) / (NIR + GREEN + epsVal)

    if RE is not None:
        NDRE = (NIR - RE) / (NIR + RE + epsVal)
    else:
        NDRE = np.full_like(NIR, np.nan, dtype=np.float64)

    L = 0.5
    SAVI = (1.0 + L) * (NIR - RED) / (NIR + RED + L + epsVal)

    msaviArg = (2.0 * NIR + 1.0) ** 2 - 8.0 * (NIR - RED)
    msaviArg = np.maximum(msaviArg, 0.0)
    MSAVI2 = (2.0 * NIR + 1.0 - np.sqrt(msaviArg)) / 2.0

    # LAI (NDVI-based empirical model; you can tune a,b,c)
    a = 0.69
    b = 0.59
    c = 0.91
    laiArg = (a - NDVI) / b
    laiArg = np.maximum(laiArg, 1e-6)
    LAI = -np.log(laiArg) / c

    # Clip common ranges
    NDVI = np.clip(NDVI, -1, 1)
    EVI = np.clip(EVI, -1, 1)
    GNDVI = np.clip(GNDVI, -1, 1)
    NDRE = np.clip(NDRE, -1, 1)

    LAI[~np.isfinite(LAI)] = np.nan
    LAI[LAI < 0] = 0
    LAI[LAI > 10] = 10

    # ============================================================
    # STEP 5) Visualize + stats + pixel counts + export GeoTIFFs
    # ============================================================

    # --- Colormap where 0 is white (lowest color = white) ---
    # NOTE: This works as expected when you set caxis so that 0 is the minimum
    nC = 255
    cm = cmocean.cm.curl(np.linspace(0, 1, nC))  # sample cmocean "curl" into an (nC,4) RGBA array
    cm = cm[:, :3]  # RGB only
    cm[0, :] = [1, 1, 1]  # force lowest color to white
    cm_mpl = mcolors.ListedColormap(cm)

    # Figure "Vegetation indices" (2x4 tiles)
    fig = plt.figure(num="Vegetation indices")
    gs = fig.add_gridspec(2, 4)

    def show_tile(ax, data, title, vmin, vmax, cmap):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show_tile(fig.add_subplot(gs[0, 0]), NDVI, "NDVI", -1, 1, cm_mpl)
    show_tile(fig.add_subplot(gs[0, 1]), EVI, "EVI", -1, 1, cm_mpl)
    show_tile(fig.add_subplot(gs[0, 2]), GNDVI, "GNDVI", -1, 1, cm_mpl)
    show_tile(fig.add_subplot(gs[0, 3]), NDRE, "NDRE", -1, 1, cm_mpl)

    show_tile(fig.add_subplot(gs[1, 0]), SAVI, "SAVI", -1, 1, cm_mpl)
    show_tile(fig.add_subplot(gs[1, 1]), MSAVI2, "MSAVI2", -1, 1, cm_mpl)

    # LAI: caxis([0 10]) -> vmin=0 vmax=10
    show_tile(fig.add_subplot(gs[1, 2]), LAI, "LAI", 0, 10, cm_mpl)

    # last tile off
    ax_last = fig.add_subplot(gs[1, 3])
    ax_last.set_axis_off()

    plt.tight_layout()
    plt.show()

    print("\n--- Index statistics (finite pixels only) ---")

    def stats(name, arr):
        x = arr[np.isfinite(arr)]
        if x.size == 0:
            print(f"{name}: no finite pixels")
        else:
            print(f"{name}: min={x.min():g} max={x.max():g} mean={x.mean():g} std={x.std():g}")

    stats("NDVI", NDVI)
    stats("EVI", EVI)
    stats("GNDVI", GNDVI)
    stats("NDRE", NDRE)
    stats("SAVI", SAVI)
    stats("MSAVI2", MSAVI2)
    stats("LAI", LAI)

    totalPixels = NIR.size

    print("\n--- Pixel counts ---")
    print("Total pixels:", int(totalPixels))
    print("Valid NDVI pixels: ", int(np.count_nonzero(np.isfinite(NDVI))))
    print("Valid EVI pixels:  ", int(np.count_nonzero(np.isfinite(EVI))))
    print("Valid GNDVI pixels:", int(np.count_nonzero(np.isfinite(GNDVI))))
    print("Valid NDRE pixels: ", int(np.count_nonzero(np.isfinite(NDRE))))
    print("Valid SAVI pixels: ", int(np.count_nonzero(np.isfinite(SAVI))))
    print("Valid MSAVI2 pixels:", int(np.count_nonzero(np.isfinite(MSAVI2))))
    print("Valid LAI pixels:  ", int(np.count_nonzero(np.isfinite(LAI))))

    outDir = os.path.join(os.path.dirname(tifPath), "indices_output")
    os.makedirs(outDir, exist_ok=True)

    # Write GeoTIFFs (float32), same transform + crs
    # Keep nodata as NaN (float). If your GIS dislikes NaN nodata, set nodata=-9999 and replace NaNs before writing.
    def write_tif(out_name, arr2d):
        out_path = os.path.join(outDir, out_name)
        profile = R_profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=np.nan,
            compress="deflate"
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr2d.astype(np.float32), 1)

    write_tif("NDVI.tif", NDVI)
    write_tif("EVI.tif", EVI)
    write_tif("GNDVI.tif", GNDVI)
    write_tif("NDRE.tif", NDRE)
    write_tif("SAVI.tif", SAVI)
    write_tif("MSAVI2.tif", MSAVI2)
    write_tif("LAI.tif", LAI)

    print("GeoTIFF outputs written to:", outDir)


if __name__ == "__main__":
    main()
