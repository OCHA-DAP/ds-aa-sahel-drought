import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from ochanticipy import CodAB, create_country_config

DATA_DIR = Path(os.getenv("AA_DATA_DIR"))


def process_asap_raw():
    codab = load_codab_all()
    load_dir = DATA_DIR / "public/raw/glb/asap/reference_data"
    save_dir = DATA_DIR / "public/processed/sah/asap"
    for season in [1, 2]:
        for s_e in ["s", "e"]:
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