import itertools
import os
from pathlib import Path
from typing import Literal

import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from ochanticipy import CodAB, create_country_config
from rasterio.enums import Resampling
from shapely import box
from tqdm import tqdm

DATA_DIR = Path(os.getenv("AA_DATA_DIR"))


def load_codab_aoi() -> gpd.GeoDataFrame:
    load_dir = DATA_DIR / "public/processed/sah/cod_ab"
    filename = "bfa-ner-tcd_adm2_codab_aoi.shp.zip"
    return gpd.read_file(f"zip://{load_dir / filename}")


def process_clip_cod_to_aoi():
    """
    Clip CODAB to area of interest for frameworks. In Burkina and Chad, based
    on specific admin1s. In Niger, is everything south of 17N.
    Returns
    -------

    """
    cod = load_codab_all()
    bfa_adm1_codes = ["BF46", "BF49", "BF54", "BF56"]
    bfa_clip = cod[cod["ADM1_CODE"].isin(bfa_adm1_codes)]
    tcd_adm1_codes = ["TD01", "TD06", "TD07", "TD17", "TD19"]
    tcd_clip = cod[cod["ADM1_CODE"].isin(tcd_adm1_codes)]
    ner_clip = cod[cod["ADM0_CODE"] == "NER"].clip(box(-180, -90, 180, 17))
    cod_clip = pd.concat([bfa_clip, tcd_clip, ner_clip], ignore_index=True)
    save_dir = DATA_DIR / "public/processed/sah/cod_ab"
    filename = "bfa-ner-tcd_adm2_codab_aoi.shp.zip"
    cod_clip.to_file(save_dir / filename)


def load_asap_sos_eos():
    load_dir = DATA_DIR / "public/processed/sah/asap"
    da_ss = []
    for season in [1, 2]:
        da_ses = []
        for s_e in ["s", "e", "sen"]:
            filename = f"pheno{s_e}{season}_v03_sah.tif"
            da_se = rxr.open_rasterio(load_dir / filename).astype("uint8")
            da_se["s_e"] = s_e
            da_ses.append(da_se)
        da_s = xr.concat(da_ses, dim="s_e")
        da_s["season"] = season
        da_ss.append(da_s)
    da = xr.concat(da_ss, dim="season")
    da = da.squeeze(drop=True)
    return da


def load_iri():
    load_dir = DATA_DIR / "private/processed/sah/iri"
    filename = "sah_iri_lowtercileprob_aoi.nc"
    da = xr.load_dataset(load_dir / filename)["prob"]
    return da


def calculate_iri_inseason_stats():
    iri = load_iri()
    aoi = load_codab_aoi()
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

    save_dir = DATA_DIR / "private/processed/sah/iri"
    filename = "iri_stats_adm0_any_inseason.csv"
    stats.to_csv(save_dir / filename, index=False)


def load_iri_inseason(forecast_date: str | cftime.Datetime360Day):
    if isinstance(forecast_date, cftime.Datetime360Day):
        forecast_date = forecast_date.isoformat().split("T")[0]
    load_dir = DATA_DIR / "private/processed/sah/iri/aoi_inseason_tif"
    filename = f"sah_iri_lowtercileprob_aoi_inseason_{forecast_date}.tif"
    da = rxr.open_rasterio(load_dir / filename)
    da = da.rename({"band": "L"})
    return da


def load_asap_inseason(interval, number, agg: Literal["any", "sum"] = "any"):
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
    else:
        file_interval, dir_agg = "", "dekad"
    load_dir = DATA_DIR / f"public/processed/sah/asap/season/{dir_agg}_sen"
    filename = f"{file_interval}inseason_{interval}{number}_sen_aoi.tif"
    return rxr.open_rasterio(load_dir / filename).squeeze(drop=True)


def process_asap_inseason():
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
        save_dir = DATA_DIR / "private/processed/sah/iri/aoi_inseason_tif"
        filename = (
            f"sah_iri_lowtercileprob_aoi_inseason_"
            f"{F.isoformat().split('T')[0]}.tif"
        )
        da_iri.rio.to_raster(save_dir / filename, driver="COG")


def clip_asap_inseason_dekad(start_dekad: int = 1):
    # Note: might crash. Adjust start_dekad to pick up where you left off.
    aoi = load_codab_aoi()
    load_dir = DATA_DIR / "public/processed/glb/asap/season/dekad_sen"
    save_dir = DATA_DIR / "public/processed/sah/asap/season/dekad_sen"
    for dekad in tqdm(range(start_dekad, 37)):
        filestem = f"inseason_dekad{dekad}_sen"
        ext = ".tif"
        da = (
            rxr.open_rasterio(load_dir / f"{filestem}{ext}")
            .astype(float)
            .squeeze(drop=True)
        )
        da.rio.write_crs(4326, inplace=True)
        da_clip = da.rio.clip(aoi.geometry, all_touched=True)
        da_clip = da_clip.fillna(254)
        da_clip = da_clip.astype("uint8")
        da_clip.rio.to_raster(save_dir / f"{filestem}_aoi{ext}", driver="COG")


def clip_asap_inseason_trimester(start_month: int = 1):
    # Note: might crash. Adjust start_month to pick up where you left off.
    aoi = load_codab_aoi()
    load_dir = DATA_DIR / "public/processed/glb/asap/season/trimester_any_sen"
    save_dir = DATA_DIR / "public/processed/sah/asap/season/trimester_any_sen"
    for month in tqdm(range(start_month, 13)):
        rel_months_str = "-".join(
            [
                str(x) if x < 13 else str(x - 12)
                for x in range(month, month + 3)
            ]
        )
        filestem = f"any_dekad_inseason_months-{rel_months_str}_sen"
        ext = ".tif"
        da = (
            rxr.open_rasterio(load_dir / f"{filestem}{ext}")
            .astype(float)
            .squeeze(drop=True)
        )
        da.rio.write_crs(4326, inplace=True)
        da_clip = da.rio.clip(aoi.geometry, all_touched=True)
        da_clip = da_clip.fillna(254)
        da_clip = da_clip.astype("uint8")
        da_clip.rio.to_raster(save_dir / f"{filestem}_aoi{ext}", driver="COG")


def process_asap_raw():
    codab = load_codab_all()
    load_dir = DATA_DIR / "public/raw/glb/asap/reference_data"
    save_dir = DATA_DIR / "public/processed/sah/asap"
    for season in [1, 2]:
        for s_e in ["s", "e", "sen"]:
            filestem = f"pheno{s_e}{season}_v03"
            ext = ".tif"
            da = rxr.open_rasterio(load_dir / f"{filestem}{ext}").astype(
                "uint8"
            )
            da = da.assign_attrs({"_FillValue": 254})
            da = da.rio.clip(codab["geometry"], all_touched=True)
            da = da.squeeze(drop=True)
            da.rio.to_raster(save_dir / f"{filestem}_sah{ext}", driver="COG")


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
    save_dir = DATA_DIR / "public/processed/sah/cod_ab"
    filename = "bfa-ner-tcd_adm2_codab.shp.zip"
    codab_all.to_file(save_dir / filename)


def load_codab_all() -> gpd.GeoDataFrame:
    load_dir = DATA_DIR / "public/processed/sah/cod_ab"
    filename = "bfa-ner-tcd_adm2_codab.shp.zip"
    return gpd.read_file(f"zip://{load_dir / filename}")


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
