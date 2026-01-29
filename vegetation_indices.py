# ============================================================
# Vegetation indices + LAI from multiband GeoTIFF 
# - Same indices + exports
# ============================================================

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ============================================================
# STEP 2) Add the TIFF file & quicklook
# ============================================================
tifPath = r"C:\Users\ushre\Downloads\06_10_2025_GoodALC_only.tif"

if not os.path.isfile(tifPath):
    raise FileNotFoundError(f"TIFF file not found: {tifPath}")

with rasterio.open(tifPath) as ds:
    # Read all bands: shape -> (bands, rows, cols)
    A = ds.read().astype(np.float64)
    R_transform = ds.transform
    R_crs = ds.crs
    R_profile = ds.profile.copy()

    # Similar to MATLAB "info"
    print("Raster size:", A.shape)
    print("Coordinate system code (PCS if available): ")
    if R_crs is not None:
        try:
            print(R_crs)
        except Exception:
            print("CRS available but could not be printed.")
    else:
        print("PCS/CRS not available (varies by file).")

    # Try to detect NoData value (varies by GeoTIFF)
    noDataVal = None
    try:
        if ds.nodata is not None:
            noDataVal = float(ds.nodata)
        else:
            tags = ds.tags()
            if "GDAL_NODATA" in tags:
                noDataVal = float(tags["GDAL_NODATA"])
    except Exception:
        noDataVal = None

# Quicklook (band 1)
plt.figure(num="GeoTIFF quicklook")
if A.ndim == 3:
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
# Example assumption: [1=Blue, 2=Green, 3=Red, 4=RedEdge, 5=NIR]
bBlue = 1
bGreen = 2
bRed = 3
bRedEdge = 4 
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

# Extract bands (convert 1-based to 0-based)
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

# --- Colormap: BrBG (ColorBrewer) ---
# Create N colors and force the LOWEST color to white (for ranges where 0 is minimum)
nC = 255
brbg_base = cm.get_cmap("BrBG", nC)                 # sampled BrBG
cm_arr = brbg_base(np.linspace(0, 1, nC))[:, :3]    # RGB
cm_arr[0, :] = [1, 1, 1]                            # force lowest color to white
cm_mpl = mcolors.ListedColormap(cm_arr)

# Vegetation indices figure (2x4 tiles)
fig = plt.figure(num="Vegetation indices")
gs = fig.add_gridspec(2, 4)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(NDVI, vmin=-1, vmax=1, cmap=cm_mpl, interpolation="nearest")
ax.set_title("NDVI")
ax.set_axis_off()
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(EVI, vmin=-1, vmax=1, cmap=cm_mpl, interpolation="nearest")
ax.set_title("EVI")
ax.set_axis_off()
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(GNDVI, vmin=-1, vmax=1, cmap=cm_mpl, interpolation="nearest")
ax.set_title("GNDVI")
ax.set_axis_off()
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[0, 3])
im = ax.imshow(NDRE, vmin=-1, vmax=1, cmap=cm_mpl, interpolation="nearest")
ax.set_title("NDRE")
ax.set_axis_off()
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(SAVI, vmin=-1, vmax=1, cmap=cm_mpl, interpolation="nearest")
ax.set_title("SAVI")
ax.set_axis_off()
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(MSAVI2, vmin=-1, vmax=1, cmap=cm_mpl, interpolation="nearest")
ax.set_title("MSAVI2")
ax.set_axis_off()
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[1, 2])
im = ax.imshow(LAI, vmin=0, vmax=10, cmap=cm_mpl, interpolation="nearest")
ax.set_title("LAI")
ax.set_axis_off()
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[1, 3])
ax.set_axis_off()

plt.tight_layout()
plt.show()

print("\n--- Index statistics (finite pixels only) ---")

x = NDVI[np.isfinite(NDVI)]
if x.size == 0:
    print("NDVI: no finite pixels")
else:
    print(f"NDVI: min={x.min():g} max={x.max():g} mean={x.mean():g} std={x.std():g}")

x = EVI[np.isfinite(EVI)]
if x.size == 0:
    print("EVI: no finite pixels")
else:
    print(f"EVI: min={x.min():g} max={x.max():g} mean={x.mean():g} std={x.std():g}")

x = GNDVI[np.isfinite(GNDVI)]
if x.size == 0:
    print("GNDVI: no finite pixels")
else:
    print(f"GNDVI: min={x.min():g} max={x.max():g} mean={x.mean():g} std={x.std():g}")

x = NDRE[np.isfinite(NDRE)]
if x.size == 0:
    print("NDRE: no finite pixels")
else:
    print(f"NDRE: min={x.min():g} max={x.max():g} mean={x.mean():g} std={x.std():g}")

x = SAVI[np.isfinite(SAVI)]
if x.size == 0:
    print("SAVI: no finite pixels")
else:
    print(f"SAVI: min={x.min():g} max={x.max():g} mean={x.mean():g} std={x.std():g}")

x = MSAVI2[np.isfinite(MSAVI2)]
if x.size == 0:
    print("MSAVI2: no finite pixels")
else:
    print(f"MSAVI2: min={x.min():g} max={x.max():g} mean={x.mean():g} std={x.std():g}")

x = LAI[np.isfinite(LAI)]
if x.size == 0:
    print("LAI: no finite pixels")
else:
    print(f"LAI: min={x.min():g} max={x.max():g} mean={x.mean():g} std={x.std():g}")

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
profile = R_profile.copy()
profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress="deflate")

out_path = os.path.join(outDir, "NDVI.tif")
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(NDVI.astype(np.float32), 1)

out_path = os.path.join(outDir, "EVI.tif")
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(EVI.astype(np.float32), 1)

out_path = os.path.join(outDir, "GNDVI.tif")
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(GNDVI.astype(np.float32), 1)

out_path = os.path.join(outDir, "NDRE.tif")
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(NDRE.astype(np.float32), 1)

out_path = os.path.join(outDir, "SAVI.tif")
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(SAVI.astype(np.float32), 1)

out_path = os.path.join(outDir, "MSAVI2.tif")
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(MSAVI2.astype(np.float32), 1)

out_path = os.path.join(outDir, "LAI.tif")
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(LAI.astype(np.float32), 1)

print("GeoTIFF outputs written to:", outDir)
