import itertools
import os
from pathlib import Path
from typing import Literal

import cdsapi
import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from ochanticipy import (
    CodAB,
    GeoBoundingBox,
    IriForecastProb,
    create_country_config,
    create_custom_country_config,
)
from rasterio.enums import Resampling
from shapely import box
from tqdm import tqdm

DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
PROC_CODAB_DIR = DATA_DIR / "public/processed/sah/cod_ab"
PROC_ASAP_DIR = DATA_DIR / "public/processed/sah/asap"
PROC_IRI_DIR = DATA_DIR / "private/processed/sah/iri"
PROC_IRI_SEASON_TIF_DIR = PROC_IRI_DIR / "aoi_inseason_tif"
PROC_GLB_SEASON_DIR = DATA_DIR / "public/processed/glb/asap/season"
PROC_SAH_SEASON_DIR = DATA_DIR / "public/processed/sah/asap/season"
PROC_GLB_SEASON_DEKAD_DIR = PROC_GLB_SEASON_DIR / "dekad_sen"
PROC_SAH_SEASON_DEKAD_DIR = PROC_SAH_SEASON_DIR / "dekad_sen"
PROC_GLB_SEASON_MONTH_DIR = PROC_GLB_SEASON_DIR / "month_sen"
PROC_SAH_SEASON_MONTH_DIR = PROC_SAH_SEASON_DIR / "month_any_sen"
PROC_GLB_SEASON_TRI_DIR = PROC_GLB_SEASON_DIR / "trimester_any_sen"
PROC_SAH_SEASON_TRI_DIR = PROC_SAH_SEASON_DIR / "trimester_any_sen"
RAW_ASAP_REF_DIR = DATA_DIR / "public/raw/glb/asap/reference_data"
RAW_ECMWF_DIR = DATA_DIR / "public" / "raw" / "sah" / "ecmwf"
PROC_ECMWF_DIR = DATA_DIR / "public" / "processed" / "sah" / "ecmwf"


def process_ecmwf_rank() -> xr.Dataset:
    pass


def load_ecmwf() -> xr.DataArray:
    filename = "ecmwf_total-precipitation_sah.nc"
    ds = xr.open_dataset(PROC_ECMWF_DIR / filename)
    da = ds["tprate"]
    da.rio.set_crs(4326, inplace=True)
    da = da.drop_vars("surface")
    return da


def process_ecmwf():
    leadtimes = range(1, 7)
    aoi = load_codab(aoi_only=True)
    das = []
    system = "51"
    fileformat = "grib"
    for leadtime in tqdm(leadtimes):
        filename = f"ecmwf-total-leadtime-{leadtime}_sys{system}.{fileformat}"
        ds = xr.load_dataset(RAW_ECMWF_DIR / filename, engine="cfgrib")
        da = ds["tprate"].mean(dim=["number", "step"])
        da.rio.write_crs(4326, inplace=True)
        da = da.rio.clip(aoi.geometry, all_touched=True)
        da["leadtime"] = leadtime
        das.append(da)
    em = xr.concat(das, dim="leadtime")

    filename = "ecmwf_total-precipitation_sah.nc"
    filepath = PROC_ECMWF_DIR / filename
    if filepath.exists():
        filepath.unlink()
    em.to_netcdf(filepath)


def download_ecmwf():
    c = cdsapi.Client()
    aoi = load_codab(aoi_only=True)
    bounds = aoi.total_bounds
    area = [bounds[3], bounds[0], bounds[1], bounds[2]]
    start_year = 1981
    end_year = 2022
    leadtimes = range(1, 7)
    system = "51"
    fileformat = "grib"
    for leadtime_month in leadtimes:
        print(leadtime_month)
        data_request_netcdf = {
            "format": fileformat,
            "originating_centre": "ecmwf",
            "system": system,
            "variable": "total_precipitation",
            "product_type": "monthly_mean",
            "year": [f"{d}" for d in range(start_year, end_year + 1)],
            "month": [f"{d:02d}" for d in range(1, 13)],
            "leadtime_month": leadtime_month,
            "area": area,
        }
        filename = (
            f"ecmwf-total-leadtime-{leadtime_month}_sys{system}.{fileformat}"
        )
        c.retrieve(
            "seasonal-monthly-single-levels",
            data_request_netcdf,
            RAW_ECMWF_DIR / filename,
        )


def load_codab(aoi_only: bool = False) -> gpd.GeoDataFrame:
    """Loads admin2 CODAB for BFA, NER, TCD

    Parameters
    ----------
    aoi_only: bool = False
        If True, returns CODAB clipped to area of interest for frameworks.
        In Burkina and Chad, based on specific admin1s.
        In Niger, is everything south of 17N.

    Returns
    -------
    gpd.GeoDataFrame
    """
    filestem = "bfa-ner-tcd_adm2_codab"
    if aoi_only:
        filestem += "_aoi"
    filename = f"{filestem}.shp.zip"
    return gpd.read_file(f"zip://{PROC_CODAB_DIR / filename}")


def process_clip_cod_to_aoi():
    """Clip CODAB to area of interest for frameworks. In Burkina and Chad,
    based on specific admin1s. In Niger, is everything south of 17N.
    Returns
    -------

    """
    cod = load_codab()
    bfa_adm1_codes = ["BF46", "BF49", "BF54", "BF56"]
    bfa_clip = cod[cod["ADM1_CODE"].isin(bfa_adm1_codes)]
    tcd_adm1_codes = ["TD01", "TD06", "TD07", "TD17", "TD19"]
    tcd_clip = cod[cod["ADM1_CODE"].isin(tcd_adm1_codes)]
    ner_clip = cod[cod["ADM0_CODE"] == "NER"].clip(box(-180, -90, 180, 17))
    cod_clip = pd.concat([bfa_clip, tcd_clip, ner_clip], ignore_index=True)
    filename = "bfa-ner-tcd_adm2_codab_aoi.shp.zip"
    cod_clip.to_file(PROC_CODAB_DIR / filename)


def load_asap_sos_eos() -> xr.DataArray:
    """Loads ASAP start/end/senescence of season for countries of interest.

    Returns
    -------
    xr.DataArray
    """
    da_ss = []
    for season in [1, 2]:
        da_ses = []
        for s_e in ["s", "e", "sen"]:
            filename = f"pheno{s_e}{season}_v03_sah.tif"
            da_se = rxr.open_rasterio(PROC_ASAP_DIR / filename).astype("uint8")
            da_se["s_e"] = s_e
            da_ses.append(da_se)
        da_s = xr.concat(da_ses, dim="s_e")
        da_s["season"] = season
        da_ss.append(da_s)
    da = xr.concat(da_ss, dim="season")
    da = da.squeeze(drop=True)
    return da


def process_iri_aoi_lowtercile():
    """Processes IRI low tercile probability for only AOI"""
    cod_all = load_codab()
    aoi = load_codab(aoi_only=True)
    country_config = create_custom_country_config("../sah.yaml")
    geobb = GeoBoundingBox.from_shape(cod_all)

    iri_prob = IriForecastProb(
        country_config=country_config, geo_bounding_box=geobb
    )
    # download and process with AnticiPy
    # Note: iri_prob.download() requires Python 3.9
    iri_prob.download()
    iri_prob.process()
    da = iri_prob.load()["prob"]
    # keep only low tercile (C=0)
    da = da.sel(C=0).squeeze(drop=True)
    da_aoi = da.rio.clip(aoi.geometry, all_touched=True)
    filename = "sah_iri_lowtercileprob_aoi.nc"
    filepath = PROC_IRI_DIR / filename
    # below lines may be needed due to Xarray bug
    if filepath.exists():
        filepath.unlink()
    da_aoi.to_netcdf(filepath)


def load_iri() -> xr.DataArray:
    """Loads IRI low tercile probability over AOI

    Returns
    -------
    xr.DataArray
    """
    filename = "sah_iri_lowtercileprob_aoi.nc"
    da = xr.load_dataset(PROC_IRI_DIR / filename)["prob"]
    da.rio.write_crs(4326, inplace=True)
    return da


def load_iri_inseason_stats() -> pd.DataFrame:
    """Loads raster stats for IRI-inseason intersections."""
    filename = "iri_stats_adm0_any_inseason.csv"
    return pd.read_csv(PROC_IRI_DIR / filename)


def calculate_iri_inseason_stats():
    """Calculates raster stats for IRI-inseason intersections."""
    iri = load_iri()
    aoi = load_codab(aoi_only=True)
    df_ins = []
    percentiles = range(10, 100, 20)

    for F in tqdm(iri.F.values):
        da = load_iri_inseason(forecast_date=F)
        df_in = da.oap.compute_raster_stats(
            gdf=aoi, feature_col="ADM0_CODE", percentile_list=percentiles
        )
        df_in["F"] = F
        df_ins.append(df_in)

    stats = pd.concat(df_ins, ignore_index=True)
    stats["F"] = pd.to_datetime(stats["F"])
    stats["rel_month1"] = stats["F"].apply(lambda x: x.month) + stats[
        "L"
    ].astype(int)
    stats["rel_month1"] = stats["rel_month1"].apply(
        lambda x: x if x < 13 else x - 12
    )
    stats["F_year"] = stats["F"].apply(lambda x: x.year)

    filename = "iri_stats_adm0_any_inseason.csv"
    stats.to_csv(PROC_IRI_DIR / filename, index=False)


def load_iri_inseason(
    forecast_date: str | cftime.Datetime360Day,
) -> xr.DataArray:
    """Loads IRI forecast masked by trimestrial inseason from ASAP

    Parameters
    ----------
    forecast_date: str | cftime.Datetime360Day
        IRI forecast publication date

    Returns
    -------

    """
    if isinstance(forecast_date, cftime.Datetime360Day):
        forecast_date = forecast_date.isoformat().split("T")[0]
    filename = f"sah_iri_lowtercileprob_aoi_inseason_{forecast_date}.tif"
    da = rxr.open_rasterio(PROC_IRI_SEASON_TIF_DIR / filename)
    da = da.rename({"band": "L"})
    return da


def load_asap_inseason(
    interval: Literal["dekad", "month", "trimester"],
    number: int,
    agg: Literal["any", "sum"] = "any",
) -> xr.DataArray:
    """Loads ASAP inseason rasters.

    Parameters
    ----------
    interval: Literal["dekad", "month", "trimester"]
    number: int
        The number of the interval to load.
        For trimester, the first month of the trimester.
    agg: Literal["any", "sum"] = "any"


    Returns
    -------
    xr.DataArray
    """
    if interval == "trimester":
        file_interval = f"{agg}_dekad_"
        dir_agg = f"trimester_{agg}"
        number = "-".join(
            [
                str(x) if x < 13 else str(x - 12)
                for x in range(number, number + 3)
            ]
        )
        interval = "months-"
    elif interval == "month":
        file_interval = ""
        dir_agg = f"month_{agg}"
    else:
        file_interval, dir_agg = "", "dekad"
    load_dir = PROC_SAH_SEASON_DIR / f"{dir_agg}_sen"
    filename = f"{file_interval}inseason_{interval}{number}_sen_aoi.tif"
    return rxr.open_rasterio(load_dir / filename).squeeze(drop=True)


def process_iri_inseason():
    """Processes IRI and ASAP inseason intersection.
    Interpolates IRI forecast (1deg resolution) at coordinates of ASAP inseason
     raster (0.01deg resolution).
    Keeps only pixels that are in season.

    Returns
    -------

    """
    # Note: not very computationally efficient,
    # but works at this scale of data.

    # load iri
    iri = load_iri()

    # load inseason
    da_ins = []
    for month in range(1, 13):
        da_in = load_asap_inseason("trimester", month, agg="any")
        da_in["start_month"] = month
        da_ins.append(da_in)
    tri = xr.concat(da_ins, dim="start_month")
    tri = tri.where(tri < 251)

    # get intersection of inseason and IRI
    for F in tqdm(iri.F.values):
        start_months = [
            x if x < 13 else x - 12 for x in range(F.month + 1, F.month + 5)
        ]
        tri_l = tri.sel(start_month=start_months)
        tri_l = tri_l.rename({"start_month": "L", "x": "X", "y": "Y"})
        tri_l["L"] = iri.L
        da_iri = tri_l * iri.sel(F=F).interp_like(tri_l, method="nearest")
        da_iri = da_iri.where(da_iri > 0)
        da_iri = da_iri.transpose("L", "Y", "X")
        da_iri = da_iri.rio.set_spatial_dims(x_dim="X", y_dim="Y")
        filename = (
            f"sah_iri_lowtercileprob_aoi_inseason_"
            f"{F.isoformat().split('T')[0]}.tif"
        )
        da_iri.rio.to_raster(PROC_IRI_SEASON_TIF_DIR / filename, driver="COG")


def clip_asap_inseason_dekad(start_dekad: int = 1):
    """Clips existing global inseason dekad files to AOI"""
    # Note: might crash. Adjust start_dekad to pick up where you left off.
    aoi = load_codab(aoi_only=True)
    for dekad in tqdm(range(start_dekad, 37)):
        filestem = f"inseason_dekad{dekad}_sen"
        ext = ".tif"
        da = (
            rxr.open_rasterio(PROC_GLB_SEASON_DEKAD_DIR / f"{filestem}{ext}")
            .astype(float)
            .squeeze(drop=True)
        )
        da.rio.write_crs(4326, inplace=True)
        da_clip = da.rio.clip(aoi.geometry, all_touched=True)
        da_clip = da_clip.fillna(254)
        da_clip = da_clip.astype("uint8")
        da_clip.rio.to_raster(
            PROC_SAH_SEASON_DEKAD_DIR / f"{filestem}_aoi{ext}", driver="COG"
        )


def clip_asap_inseason_month(start_month: int = 1):
    """Clips existing global inseason month files to AOI"""
    # Note: might crash. Adjust start_month to pick up where you left off.
    aoi = load_codab(aoi_only=True)
    for month in tqdm(range(start_month, 13)):
        filestem = f"inseason_month{month}_sen"
        ext = ".tif"
        da = (
            rxr.open_rasterio(PROC_GLB_SEASON_MONTH_DIR / f"{filestem}{ext}")
            .astype(float)
            .squeeze(drop=True)
        )
        da.rio.write_crs(4326, inplace=True)
        da_clip = da.rio.clip(aoi.geometry, all_touched=True)
        da_clip = da_clip.fillna(254)
        da_clip = da_clip.astype("uint8")
        da_clip.rio.to_raster(
            PROC_SAH_SEASON_MONTH_DIR / f"{filestem}_aoi{ext}", driver="COG"
        )


def clip_asap_inseason_trimester(start_month: int = 1):
    """Clips existing global inseason trimester files to AOI"""
    # Note: might crash. Adjust start_month to pick up where you left off.
    aoi = load_codab(aoi_only=True)
    for month in tqdm(range(start_month, 13)):
        rel_months_str = "-".join(
            [str(((x - 1) % 12) + 1) for x in range(month, month + 3)]
        )
        filestem = f"any_dekad_inseason_months-{rel_months_str}_sen"
        ext = ".tif"
        da = (
            rxr.open_rasterio(PROC_GLB_SEASON_TRI_DIR / f"{filestem}{ext}")
            .astype(float)
            .squeeze(drop=True)
        )
        da.rio.write_crs(4326, inplace=True)
        da_clip = da.rio.clip(aoi.geometry, all_touched=True)
        da_clip = da_clip.fillna(254)
        da_clip = da_clip.astype("uint8")
        da_clip.rio.to_raster(
            PROC_SAH_SEASON_TRI_DIR / f"{filestem}_aoi{ext}", driver="COG"
        )


def clip_asap_raw():
    """Clips ASAP start/end/senescence to AOI"""
    codab = load_codab(aoi_only=True)
    for season in [1, 2]:
        for s_e in ["s", "e", "sen"]:
            filestem = f"pheno{s_e}{season}_v03"
            ext = ".tif"
            da = rxr.open_rasterio(
                RAW_ASAP_REF_DIR / f"{filestem}{ext}"
            ).astype("uint8")
            da = da.assign_attrs({"_FillValue": 254})
            da = da.rio.clip(codab["geometry"], all_touched=True)
            da = da.squeeze(drop=True)
            da.rio.to_raster(
                PROC_ASAP_DIR / f"{filestem}_sah{ext}", driver="COG"
            )


def process_codab_all():
    """
    Combine CODABs for Burkina, Niger, and Chad.
    Standardize column naming for admin names and codes.
    Returns
    -------

    """
    adm0s = [
        {
            "iso3": "bfa",
            "name": "Burkina Faso",
            "adm2": {"name_col": "ADM2_FR", "code_col": "ADM2_PCODE"},
            "adm1": {"name_col": "ADM1_FR", "code_col": "ADM1_PCODE"},
        },
        {
            "iso3": "ner",
            "name": "Niger",
            "adm2": {"name_col": "adm_02", "code_col": "rowcacode2"},
            "adm1": {"name_col": "adm_01", "code_col": "rowcacode1"},
        },
        {
            "iso3": "tcd",
            "name": "Chad",
            "adm2": {"name_col": "admin2Name", "code_col": "admin2Pcod"},
            "adm1": {"name_col": "admin1Name", "code_col": "admin1Pcod"},
        },
    ]
    gdfs = []
    for adm0 in adm0s:
        codab = CodAB(create_country_config(adm0.get("iso3")))
        codab.download()
        codab.process()
        gdf = codab.load(admin_level=2)
        gdf["ADM0_CODE"] = adm0.get("iso3").upper()
        gdf["ADM0_NAME"] = adm0.get("name")
        for level in [1, 2]:
            gdf[f"ADM{level}_NAME"] = gdf[
                adm0.get(f"adm{level}").get("name_col")
            ]
            gdf[f"ADM{level}_CODE"] = gdf[
                adm0.get(f"adm{level}").get("code_col")
            ]
        gdfs.append(gdf)
    cols = [
        *[f"ADM{x}_{y}" for x in [0, 1, 2] for y in ["CODE", "NAME"]],
        "geometry",
    ]
    codab_all = pd.concat(gdfs, ignore_index=True)[cols]
    filename = "bfa-ner-tcd_adm2_codab.shp.zip"
    codab_all.to_file(PROC_CODAB_DIR / filename)


def approx_mask_raster(
    ds: xr.Dataset | xr.DataArray,
    x_dim: str,
    y_dim: str,
    resolution: float = 0.05,
) -> xr.Dataset:
    """
    Resample raster data to given resolution.

    Uses as resample method nearest neighbour, i.e. aims to keep the values
    the same as the original data. Mainly used to create an approximate mask
    over an area

    Taken from pa-aa-bfa-drought

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to resample.
    resolution: float, default = 0.05
        Resolution in degrees to resample to

    Returns
    -------
        Upsampled dataset
    """
    upsample_list = []
    # can only do reproject on 3D array so
    # loop over all +3D dimensions
    list_dim = [d for d in ds.dims if (d != x_dim) & (d != y_dim)]
    # select from second element of list_dim since can loop over 3D
    # loop over all combs of dims
    dim_names = list_dim[1:]
    for dim_values in itertools.product(*[ds[d].values for d in dim_names]):
        ds_sel = ds.sel(
            {name: value for name, value in zip(dim_names, dim_values)}
        )

        ds_sel_upsample = ds_sel.rio.reproject(
            ds_sel.rio.crs,
            resolution=resolution,
            resampling=Resampling.nearest,
            nodata=np.nan,
        )
        upsample_list.append(
            ds_sel_upsample.expand_dims(
                {name: [value] for name, value in zip(dim_names, dim_values)}
            )
        )
    ds_upsample = xr.combine_by_coords(upsample_list)
    # reproject changes spatial dims names to x and y
    # so change back here
    ds_upsample = ds_upsample.rename({"x": x_dim, "y": y_dim})
    if isinstance(ds, xr.DataArray):
        ds_upsample = ds_upsample[ds.name]
    return ds_upsample
