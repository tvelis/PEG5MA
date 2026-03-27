# %%
# --- CODE 1: Ask / set current round (interactive, INDICATIVE ONLY) ---
def ask_current_round(default: int = 1, min_round: int = 1) -> int:
    try:
        r = int(input(f"Current round? (default={default}): ").strip() or default)
    except ValueError:
        r = default
    return max(min_round, r)

CURRENT_ROUND = ask_current_round(default=1)
CURRENT_ROUND

# %%
# Imports y configuración general
import os
import re
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from typing import Dict, List, Optional, Tuple

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

# %%
# Definir ruta base del proyecto
BASE_PATH = r"c:\\PEG5MA"
DATA_PATH = os.path.join(BASE_PATH, "data")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No existe la carpeta: {DATA_PATH}")

print("Ruta base:", BASE_PATH)
print("Ruta datos:", DATA_PATH)
print(f"Ronda actual (solo indicativa para títulos): {CURRENT_ROUND}")

# %%
def load_csv(filename: str, skiprows: int = 0) -> pd.DataFrame:
    path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {filename}")
    df = pd.read_csv(path, skiprows=skiprows, sep=",")
    print(f"{filename} cargado | filas={len(df)}")
    return df

# %%
df_period_tender = load_csv("Period_tender.csv")
df_period_mechanism = load_csv("Period_Mechanism.csv")
df_requirement_power = load_csv("Requirement_Power.csv")
df_requirement_energy_hr = load_csv("Requirement_Energy_Hr.csv")
df_reffuelprice = load_csv("Referencefuelprices.csv", skiprows=1)
df_otherfuelvar = load_csv("OtherFuelValue.csv", skiprows=1)
df_bidders = load_csv("List_Bidders.csv")
df_period_calendar_month = load_csv("Period_Calendar_Month.csv")

# %%
df_bidders

# %%
# =========================
# CODE 2 (SCRIPT) — UPDATED
# - Uses df_bidders already loaded
# - CURRENT_ROUND is INDICATIVE ONLY (not used to filter bidders)
# - List_Bidders.csv now includes: cod, bidder, tipo, preciomonomico, 1,2,3,...
# - The monomic price used downstream comes from df_bidders['preciomonomico']
# =========================

MAX_BIDDERS = 50
SUFFIXES = ["OE", "OT_T1", "OT_T2", "OT_T3", "OT_T4"]

if len(df_bidders) > MAX_BIDDERS:
    raise ValueError(f"Too many bidders: {len(df_bidders)} (max {MAX_BIDDERS})")

df_bidders.columns = [str(c).strip() for c in df_bidders.columns]

# If df_bidders was loaded as a single column like "cod,bidder,tipo,preciomonomico,1", split it
if len(df_bidders.columns) == 1 and "," in df_bidders.columns[0]:
    tmp = df_bidders[df_bidders.columns[0]].astype(str).str.split(",", expand=True)
    tmp.columns = tmp.iloc[0].tolist()
    df_bidders = tmp.iloc[1:].reset_index(drop=True)
    df_bidders.columns = [str(c).strip() for c in df_bidders.columns]

required_cols = {"cod", "bidder", "tipo", "preciomonomico"}
missing_cols = required_cols - set(c.lower() for c in df_bidders.columns)
if missing_cols:
    raise ValueError(
        f"Columns found: {list(df_bidders.columns)} | Expected at least: cod, bidder, tipo, preciomonomico"
    )

def normalize_bidder_tipo(tipo_raw: str) -> str:
    s = str(tipo_raw).strip().upper()
    s = re.sub(r"\s+", " ", s)

    if s == "FUTURA":
        return "FUTURA"
    if s == "EXISTENTE CON INVERSIONES ADICIONALES":
        return "EXISTENTE_CON_INV_ADIC"
    if s == "EXISTENTE SIN INVERSIONES ADICIONALES":
        return "EXISTENTE_SIN_INV_ADIC"

    raise ValueError(
        f"Tipo de oferente no reconocido: '{tipo_raw}'. "
        f"Valores válidos: 'Futura', "
        f"'Existente con inversiones adicionales', "
        f"'Existente sin inversiones adicionales'."
    )


# normalize key column names preserving original labels if needed
col_cod = [c for c in df_bidders.columns if c.lower() == "cod"][0]
col_bidder = [c for c in df_bidders.columns if c.lower() == "bidder"][0]
col_tipo = [c for c in df_bidders.columns if c.lower() == "tipo"][0]
col_pmon = [c for c in df_bidders.columns if c.lower() == "preciomonomico"][0]

df_bidders[col_cod] = pd.to_numeric(df_bidders[col_cod], errors="raise").astype(int)
df_bidders[col_bidder] = df_bidders[col_bidder].astype(str).str.strip()
#df_bidders[col_tipo] = df_bidders[col_tipo].astype(str).str.strip().str.upper()
df_bidders[col_tipo] = df_bidders[col_tipo].astype(str).apply(normalize_bidder_tipo)
df_bidders[col_pmon] = pd.to_numeric(df_bidders[col_pmon], errors="raise").astype(float)

df_bidders = df_bidders.rename(
    columns={
        col_cod: "cod",
        col_bidder: "bidder",
        col_tipo: "tipo",
        col_pmon: "preciomonomico",
    }
)


def max_contract_years_from_tipo(tipo_norm: str) -> int:
    if tipo_norm in {"FUTURA", "EXISTENTE_CON_INV_ADIC"}:
        return 15
    if tipo_norm == "EXISTENTE_SIN_INV_ADIC":
        return 5
    raise ValueError(f"Tipo normalizado sin regla de plazo: '{tipo_norm}'")


# CURRENT_ROUND is NOT used for filtering anymore
df_enabled = df_bidders.loc[:, ["cod", "bidder", "tipo", "preciomonomico"]].copy().reset_index(drop=True)

print(f"CURRENT_ROUND={CURRENT_ROUND} (solo indicativo, sin filtrar oferentes)")
print(f"Bidders cargados: {len(df_enabled)}")
print(df_enabled)

# Mapping monomic price from List_Bidders.csv (source of truth)
PMON_INPUT_MAP = df_enabled.set_index("cod")["preciomonomico"].astype(float).to_dict()
TIPO_INPUT_MAP = df_enabled.set_index("cod")["tipo"].astype(str).to_dict()

# 4) Load offer tables for all bidders
offers_by_bidder = {}  # cod -> {"bidder": str, "tables": {suffix: DataFrame|None}}

for cod, bidder, tipo, preciomonomico in df_enabled.itertuples(index=False):
    cod = int(cod)
    bidder = str(bidder).strip()

    offers_by_bidder[cod] = {
        "bidder": bidder,
        "tipo": str(tipo).strip().upper(),
        "preciomonomico": float(preciomonomico),
        "tables": {}
    }

    for sfx in SUFFIXES:
        fname = f"{cod}_{sfx}.csv"
        fpath = os.path.join(DATA_PATH, fname)

        if not os.path.exists(fpath):
            offers_by_bidder[cod]["tables"][sfx] = None
            print(f"{fname} no presentado -> None")
            continue

        df_raw = pd.read_csv(fpath, sep=",", skiprows=1)
        print(f"{fname} cargado (offers) | filas={len(df_raw)}")
        offers_by_bidder[cod]["tables"][sfx] = df_raw

# %%
offers_by_bidder[6]["tables"]["OT_T4"] if 6 in offers_by_bidder and offers_by_bidder[6]["tables"]["OT_T4"] is not None else None

# %%
# --- TYPE CASTS ---
df_period_calendar_month["year"] = df_period_calendar_month["year"].astype(int)
df_period_calendar_month["month"] = df_period_calendar_month["month"].astype(int)

df_period_tender["tender_period"] = df_period_tender["tender_period"].astype(str).str.strip()
df_period_mechanism["mechanism_period"] = df_period_mechanism["mechanism_period"].astype(str).str.strip()

df_requirement_energy_hr["mechanism_period"] = df_requirement_energy_hr["mechanism_period"].astype(str).str.strip()
df_requirement_energy_hr["month"] = df_requirement_energy_hr["month"].astype(int)
df_requirement_energy_hr["hour"] = df_requirement_energy_hr["hour"].astype(int)

df_requirement_power["mechanism_period"] = df_requirement_power["mechanism_period"].astype(str).str.strip()
df_requirement_power["Type"] = df_requirement_power["Type"].astype(str).str.strip()

# %%
# --- BUILD 4320 STAGES + TOTAL REQUIREMENT VECTOR (Base + Complementary) ---
def build_stage_calendar_4320(
    df_period_calendar_month: pd.DataFrame,
    df_period_mechanism: pd.DataFrame
) -> pd.DataFrame:
    cal = (
        df_period_calendar_month[["year", "month"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    hrs = pd.DataFrame({"hour": range(1, 25)})
    stages = (
        cal.merge(hrs, how="cross")
        .sort_values(["year", "month", "hour"])
        .reset_index(drop=True)
    )

    stages["stage"] = range(1, len(stages) + 1)
    stages["date"] = pd.to_datetime(dict(year=stages["year"], month=stages["month"], day=1))

    mech = df_period_mechanism.copy()
    mech["start_date"] = pd.to_datetime(mech["start_date"], dayfirst=True, errors="coerce")
    mech["end_date"] = pd.to_datetime(mech["end_date"], dayfirst=True, errors="coerce")

    tmp = stages[["stage", "date"]].merge(
        mech[["mechanism_period", "start_date", "end_date"]],
        how="cross"
    )
    tmp = tmp.loc[
        (tmp["date"] >= tmp["start_date"]) & (tmp["date"] <= tmp["end_date"]),
        ["stage", "mechanism_period"]
    ].drop_duplicates()

    stages = stages.merge(tmp, on="stage", how="left")

    if stages["mechanism_period"].isna().any():
        miss = stages.loc[stages["mechanism_period"].isna(), ["year", "month"]].drop_duplicates()
        raise ValueError(f"Meses sin mechanism_period asignado:\n{miss}")

    if len(stages) != 4320:
        raise ValueError(f"Esperaba 4320 etapas, obtuve {len(stages)}")

    return stages[["stage", "year", "month", "hour", "mechanism_period"]]

def build_requirement_total_vector_4320(
    df_period_calendar_month: pd.DataFrame,
    df_period_mechanism: pd.DataFrame,
    df_requirement_power: pd.DataFrame,
    df_requirement_energy_hr: pd.DataFrame
) -> pd.DataFrame:
    stages = build_stage_calendar_4320(df_period_calendar_month, df_period_mechanism)

    pw = df_requirement_power.copy()
    pw["Type"] = pw["Type"].astype(str).str.strip().str.lower()
    pw["mechanism_period"] = pw["mechanism_period"].astype(str).str.strip()

    pw_base = pw.loc[pw["Type"].eq("base"), ["mechanism_period", "power_requirement"]].copy()
    stages = stages.merge(pw_base, on="mechanism_period", how="left")

    if stages["power_requirement"].isna().any():
        raise ValueError("Faltan power_requirement para algún mechanism_period (Type=Base).")

    er = df_requirement_energy_hr.copy()
    er["mechanism_period"] = er["mechanism_period"].astype(str).str.strip()
    er["month"] = er["month"].astype(int)
    er["hour"] = er["hour"].astype(int)

    stages = stages.merge(
        er[["mechanism_period", "month", "hour", "energy_requirement"]],
        on=["mechanism_period", "month", "hour"],
        how="left"
    )

    if stages["energy_requirement"].isna().any():
        miss = stages.loc[stages["energy_requirement"].isna(), ["mechanism_period", "month", "hour"]].head(50)
        raise ValueError(f"Faltan energy_requirement para algunas combinaciones (ejemplos):\n{miss}")

    stages["requirement_total"] = stages["power_requirement"] + stages["energy_requirement"]

    return stages[[
        "stage", "year", "month", "hour", "mechanism_period",
        "power_requirement", "energy_requirement", "requirement_total"
    ]]

df_req_4320 = build_requirement_total_vector_4320(
    df_period_calendar_month=df_period_calendar_month,
    df_period_mechanism=df_period_mechanism,
    df_requirement_power=df_requirement_power,
    df_requirement_energy_hr=df_requirement_energy_hr
)

req_vector_4320 = df_req_4320.set_index("stage")["requirement_total"]

# %%
# Gráfico de requerimiento total
labels = (
    df_req_4320["year"].astype(str) + "-" +
    df_req_4320["month"].astype(str).str.zfill(2) + " h" +
    df_req_4320["hour"].astype(str).str.zfill(2)
)

plt.figure(figsize=(14, 5))
plt.plot(req_vector_4320.index, req_vector_4320.values)
step = 100
plt.xticks(
    ticks=req_vector_4320.index[::step],
    labels=labels.iloc[::step],
    rotation=90
)
plt.yticks(range(0, int(req_vector_4320.max()) + 100, 100))
plt.xlabel("Etapa (año-mes-hora)")
plt.ylabel("Requerimiento total")
plt.title(f"Vector de requerimiento total (Base + Complementario) | Ronda {CURRENT_ROUND}")
plt.tight_layout()
plt.ylim(bottom=0)
plt.show()

# %%
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    return d

def _req_cols(df: pd.DataFrame, cols: list, where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {where}: {missing}. Disponibles: {list(df.columns)}")

def _ot_t3_wide_to_long(t3: pd.DataFrame) -> pd.DataFrame:
    t3s = _lower_cols(t3)
    _req_cols(t3s, ["hour"], "OT_T3")
    month_cols = [c for c in t3s.columns if c != "hour" and str(c).isdigit()]
    if not month_cols:
        raise ValueError(f"OT_T3 no tiene columnas de mes '1'..'12'. cols={list(t3s.columns)}")

    out = t3s.melt(id_vars=["hour"], value_vars=month_cols, var_name="month", value_name="mwgaran")
    out["month"] = out["month"].astype(int)
    out["hour"] = out["hour"].astype(int)
    out["mwgaran"] = pd.to_numeric(out["mwgaran"], errors="coerce")
    return out

def build_offer_profiles_4320_strict(
    offers_by_bidder: dict,
    df_period_tender: pd.DataFrame,
    df_req_4320: pd.DataFrame,
):
    stages = df_req_4320.copy()
    _req_cols(stages, ["stage", "year", "month", "hour", "mechanism_period"], "df_req_4320")
    stages["date"] = pd.to_datetime(dict(year=stages["year"], month=stages["month"], day=1))

    tender = _lower_cols(df_period_tender)
    _req_cols(tender, ["tender_period", "start_date"], "df_period_tender")
    tender["tender_period"] = tender["tender_period"].astype(str).str.strip()
    tender["start_date"] = pd.to_datetime(tender["start_date"], dayfirst=True, errors="coerce")
    tender_map = dict(zip(tender["tender_period"], tender["start_date"]))

    meta_rows, prof_long, cap_month_dcc, cap_month_eg = [], [], [], []

    for cod, pack in offers_by_bidder.items():
        tables = pack.get("tables", {})
        oe = tables.get("OE")
        t1 = tables.get("OT_T1")
        t2 = tables.get("OT_T2")
        t3 = tables.get("OT_T3")
        t4 = tables.get("OT_T4")

        if oe is None or t1 is None:
            continue

        oe = _lower_cols(oe)
        t1 = _lower_cols(t1)

        _req_cols(oe, ["contractype"], f"OE (cod={cod})")
        _req_cols(t1, ["period", "potgaranmax", "potgaranmin"], f"OT_T1 (cod={cod})")

        contractype = str(oe["contractype"].iloc[0]).strip().upper()
        if contractype not in {"OCE", "DCC", "EG"}:
            raise ValueError(f"Tipo de contrato inválido (solo OCE/DCC/EG): '{contractype}' (cod={cod})")

        # period = str(t1["period"].iloc[0]).strip()
        # start_date = tender_map.get(period, pd.NaT)
        # if pd.isna(start_date):
        #     raise ValueError(f"No se pudo mapear start_date para period={period} (cod={cod})")
        period = str(t1["period"].iloc[0]).strip()
        start_date = tender_map.get(period, pd.NaT)
        if pd.isna(start_date):
            raise ValueError(f"No se pudo mapear start_date para period={period} (cod={cod})")

        tipo_input = str(pack.get("tipo", "")).strip().upper()
        max_years = int(max_contract_years_from_tipo(tipo_input))
        end_date_exclusive = pd.Timestamp(start_date) + pd.DateOffset(years=max_years)

        blockbidded = None
        if "blockbidded" in t1.columns:
            blockbidded = str(t1["blockbidded"].iloc[0]).strip().upper()

        pgmax = float(t1["potgaranmax"].iloc[0])
        pgmin = float(t1["potgaranmin"].iloc[0])

        installed = np.nan
        if contractype == "EG":
            if t4 is None:
                raise ValueError(f"EG requiere OT_T4 (cod={cod})")
            t4s = _lower_cols(t4)
            _req_cols(t4s, ["installedcapacity"], f"OT_T4 (cod={cod})")
            installed = float(t4s["installedcapacity"].iloc[0])
            if not np.isfinite(installed) or installed <= 0:
                raise ValueError(f"installedcapacity inválida (cod={cod}): {installed}")

        # meta_rows.append({
        #     "cod": int(cod),
        #     "bidder": pack.get("bidder"),
        #     "tipo_input": pack.get("tipo"),
        #     "preciomonomico_input": float(pack.get("preciomonomico", np.nan)),
        #     "contractype": contractype,
        #     "period": period,
        #     "start_date": start_date,
        #     "blockbidded": blockbidded,
        #     "potgaranmin": pgmin,
        #     "potgaranmax": pgmax,
        #     "installedcapacity": installed
        # })
        
        meta_rows.append({
            "cod": int(cod),
            "bidder": pack.get("bidder"),
            "tipo_input": tipo_input,
            "preciomonomico_input": float(pack.get("preciomonomico", np.nan)),
            "contractype": contractype,
            "period": period,
            "start_date": start_date,
            "max_contract_years": max_years,
            "end_date_exclusive": end_date_exclusive,
            "blockbidded": blockbidded,
            "potgaranmin": pgmin,
            "potgaranmax": pgmax,
            "installedcapacity": installed
        })

        #avail = (stages["date"] >= start_date).astype(float)
        avail = (
            (stages["date"] >= pd.Timestamp(start_date)) &
            (stages["date"] < pd.Timestamp(end_date_exclusive))
        ).astype(float)
        
        if contractype == "OCE":
            a = avail.copy()
            prof_long.append(pd.DataFrame({"stage": stages["stage"].values, "cod": int(cod), "a_per_mw": a.values}))

        elif contractype == "DCC":
            if t3 is None:
                raise ValueError(f"DCC requiere OT_T3 (cod={cod})")
            if not np.isfinite(pgmax) or pgmax <= 0:
                raise ValueError(f"potgaranmax inválida (DCC cod={cod}): {pgmax}")

            t3_long = _ot_t3_wide_to_long(t3)
            tmp = stages.merge(t3_long, on=["month", "hour"], how="left")
            if tmp["mwgaran"].isna().any():
                raise ValueError(f"OT_T3 no calza con stages (DCC cod={cod}).")

            a = (tmp["mwgaran"].astype(float) / float(pgmax)) * avail
            prof_long.append(pd.DataFrame({"stage": tmp["stage"].values, "cod": int(cod), "a_per_mw": a.values}))

            if t2 is not None:
                t2s = _lower_cols(t2)
                _req_cols(t2s, ["month", "mwhgaran"], f"OT_T2 (cod={cod})")
                t2s["month"] = t2s["month"].astype(int)
                t2s["mwhgaran"] = pd.to_numeric(t2s["mwhgaran"], errors="coerce")

                sm = stages[["year", "month"]].drop_duplicates()
                sm = sm.merge(
                    t2s[["month", "mwhgaran"]].rename(columns={"mwhgaran": "cap_mwh"}),
                    on=["month"],
                    how="left"
                )
                sm = sm.loc[sm["cap_mwh"].notna(), ["year", "month", "cap_mwh"]].copy()
                if len(sm):
                    sm["cod"] = int(cod)
                    cap_month_dcc.append(sm[["cod", "year", "month", "cap_mwh"]])

        elif contractype == "EG":
            if t3 is None or t4 is None:
                raise ValueError(f"EG requiere OT_T3 y OT_T4 (cod={cod})")

            t3_long = _ot_t3_wide_to_long(t3)
            tmp = stages.merge(t3_long.rename(columns={"mwgaran": "gen_mw"}), on=["month", "hour"], how="left")
            if tmp["gen_mw"].isna().any():
                raise ValueError(f"OT_T3 no calza con stages (EG cod={cod}).")

            a = (tmp["gen_mw"].astype(float) / float(installed)) * avail
            prof_long.append(pd.DataFrame({"stage": tmp["stage"].values, "cod": int(cod), "a_per_mw": a.values}))

            t4s = _lower_cols(t4)
            _req_cols(t4s, ["month", "mwhgaran", "installedcapacity"], f"OT_T4 (EG cod={cod})")
            t4s["month"] = t4s["month"].astype(int)
            t4s["mwhgaran"] = pd.to_numeric(t4s["mwhgaran"], errors="coerce")

            sm = stages[["year", "month"]].drop_duplicates()
            sm = sm.merge(
                t4s[["month", "mwhgaran"]].rename(columns={"mwhgaran": "cap_mwh"}),
                on=["month"],
                how="left"
            )
            sm = sm.loc[sm["cap_mwh"].notna(), ["year", "month", "cap_mwh"]].copy()
            if len(sm):
                sm["cod"] = int(cod)
                cap_month_eg.append(sm[["cod", "year", "month", "cap_mwh"]])

    df_offer_meta = pd.DataFrame(meta_rows).sort_values("cod").reset_index(drop=True)

    df_profiles_long = (
        pd.concat(prof_long, ignore_index=True)
        if len(prof_long)
        else pd.DataFrame(columns=["stage", "cod", "a_per_mw"])
    )

    df_profiles_wide = (
        df_profiles_long.pivot(index="stage", columns="cod", values="a_per_mw")
        .fillna(0.0)
        .sort_index()
    )

    eg_cods = df_offer_meta.loc[df_offer_meta["contractype"].str.upper().eq("EG"), "cod"].astype(int).tolist()
    print("EG cods:", eg_cods)

    for k in eg_cods:
        a = df_profiles_long.loc[df_profiles_long["cod"].astype(int).eq(int(k)), "a_per_mw"].astype(float)
        print(f"cod={k} | A count={len(a)} | min={a.min():.6f} | max={a.max():.6f} | sum={a.sum():.6f}")

    df_cap_month_dcc = (
        pd.concat(cap_month_dcc, ignore_index=True)
        if len(cap_month_dcc)
        else pd.DataFrame(columns=["cod", "year", "month", "cap_mwh"])
    )

    df_cap_month_eg = (
        pd.concat(cap_month_eg, ignore_index=True)
        if len(cap_month_eg)
        else pd.DataFrame(columns=["cod", "year", "month", "cap_mwh"])
    )

    return df_offer_meta, df_profiles_long, df_profiles_wide, df_cap_month_dcc, df_cap_month_eg

# ---- RUN ----
df_offer_meta, df_profiles_long, df_profiles_wide, df_cap_month_dcc, df_cap_month_eg = build_offer_profiles_4320_strict(
    offers_by_bidder=offers_by_bidder,
    df_period_tender=df_period_tender,
    df_req_4320=df_req_4320
)

# %%
for k in [3, 6]:
    if k in offers_by_bidder and offers_by_bidder[k]["tables"]["OT_T3"] is not None:
        t3 = offers_by_bidder[k]["tables"]["OT_T3"].copy()
        t3.columns = [str(c).strip().lower() for c in t3.columns]
        month_cols = [c for c in t3.columns if c != "hour" and str(c).isdigit()]
        mx_mw = pd.to_numeric(t3[month_cols].stack(), errors="coerce").max()
        pi = float(df_offer_meta.set_index("cod").loc[k, "installedcapacity"])
        print(k, "max OT_T3 (MW) =", mx_mw, "| PI =", pi, "| ratio =", mx_mw / pi)

# %%
# ============================================================
# MODULO PEO TIEMPO-DEPENDIENTE (PARA PYOMO)
# ============================================================
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    return d

def build_peo_stage(
    df_offer_meta: pd.DataFrame,
    offers_by_bidder: dict,
    df_req_4320: pd.DataFrame,
    df_period_tender: pd.DataFrame,
    df_reffuelprice: pd.DataFrame,
    df_otherfuelvar: pd.DataFrame,
    season_start_month: int = 5,
    citt_km: float = 0.09,
    citt_ref_km: float = 25.0,
) -> Tuple[dict, dict]:
    K_list = sorted(df_offer_meta["cod"].astype(int).unique().tolist())
    S_list = sorted(df_req_4320["stage"].astype(int).unique().tolist())

    ofv = _norm_cols(df_otherfuelvar.copy())
    if "crecimientoanualppi" not in ofv.columns:
        raise KeyError(f"OtherFuelValue: falta crecimientoanualppi. cols={list(ofv.columns)}")
    PPI_GROWTH = float(pd.to_numeric(ofv["crecimientoanualppi"], errors="coerce").dropna().iloc[0]) / 100.0
    CITT_CARBON_25 = float(pd.to_numeric(ofv.get("cittcarbon", pd.Series([0.0])), errors="coerce").fillna(0.0).iloc[0])
    CITT_COQUE_25 = float(pd.to_numeric(ofv.get("cittcoque", pd.Series([0.0])), errors="coerce").fillna(0.0).iloc[0])

    def _season_anchor(y: int, m: int) -> int:
        return int(y) if int(m) >= season_start_month else int(y) - 1

    def season_year_index(y: int, m: int, y0: int, m0: int) -> int:
        s0 = _season_anchor(y0, m0)
        s = _season_anchor(y, m)
        return int(s - s0 + 1)

    def ppi_ratio(j: int) -> float:
        return 1.0 if j <= 1 else (1.0 + PPI_GROWTH) ** (j - 1)

    def citt_carbon(distance_km: float) -> float:
        return CITT_CARBON_25 + (float(distance_km) - float(citt_ref_km)) * float(citt_km)

    def citt_coque(distance_km: float) -> float:
        return CITT_COQUE_25 + (float(distance_km) - float(citt_ref_km)) * float(citt_km)

    tp = _norm_cols(df_period_tender.copy())
    if "tender_period" not in tp.columns or "start_date" not in tp.columns:
        raise KeyError(f"Period_tender: faltan tender_period/start_date. cols={list(tp.columns)}")
    tp["tender_period"] = tp["tender_period"].astype(str).str.strip().str.upper()
    tp["start_date"] = pd.to_datetime(tp["start_date"], dayfirst=True, errors="raise")
    TP_START = {r["tender_period"]: (int(r["start_date"].year), int(r["start_date"].month)) for r in tp.to_dict("records")}

    fp = _norm_cols(df_reffuelprice.copy())
    if "year" not in fp.columns:
        raise KeyError(f"Referencefuelprices: falta year. cols={list(fp.columns)}")
    fp["year"] = pd.to_numeric(fp["year"], errors="raise").astype(int)

    def fuel_price(y: int, col: str) -> float:
        if col not in fp.columns:
            raise KeyError(f"Referencefuelprices: falta columna '{col}'. cols={list(fp.columns)}")
        row = fp.loc[fp["year"].eq(int(y))]
        if row.empty:
            raise KeyError(f"Referencefuelprices: falta year={y}")
        return float(row.iloc[0][col])

    FUEL_CASE = {
        "BUNKER": "BUNKER",
        "GAS NATURAL": "GAS",
        "GAS": "GAS",
        "CARBÓN": "CARBON",
        "CARBON": "CARBON",
        "COQUE DE PETRÓLEO": "COQUE",
        "COQUE DE PETROLEO": "COQUE",
        "CARBÓN-COQUE DE PETRÓLEO": "MIX",
        "CARBON-COQUE DE PETROLEO": "MIX",
        "MIX": "MIX",
    }

    def norm_fuel(s: str) -> str:
        return str(s).strip().upper()

    def norm_typetech(s: str) -> str:
        return str(s).strip().upper()

    YM_h = sorted(set(zip(df_req_4320["year"].astype(int), df_req_4320["month"].astype(int))))
    PEO_KYM = {}

    for k in K_list:
        oe_df = offers_by_bidder.get(int(k), {}).get("tables", {}).get("OE")
        if oe_df is None or len(oe_df) == 0:
            raise KeyError(f"Falta OE para cod={k}")
        oe0 = _norm_cols(oe_df).iloc[0]
        peo_base = float(pd.to_numeric(oe0.get("peo", 0.0), errors="coerce") or 0.0)
        oym_base = float(pd.to_numeric(oe0.get("o&m", 0.0), errors="coerce") or 0.0)
        ctung = float(pd.to_numeric(oe0.get("ctung", 0.0), errors="coerce") or 0.0)
        ci = float(pd.to_numeric(oe0.get("ci", 0.0), errors="coerce") or 0.0)
        fagn = float(pd.to_numeric(oe0.get("fagn", 0.0), errors="coerce") or 0.0)
        fgsc = float(pd.to_numeric(oe0.get("fgsc", 0.0), errors="coerce") or 0.0)
        pcp_pct = float(pd.to_numeric(oe0.get("pcp", oe0.get("coque", 0.0)), errors="coerce") or 0.0)
        pcp = max(0.0, min(1.0, pcp_pct / 100.0))

        t1_df = offers_by_bidder.get(int(k), {}).get("tables", {}).get("OT_T1")
        if t1_df is None or len(t1_df) == 0:
            raise KeyError(f"Falta OT_T1 para cod={k}")
        t10 = _norm_cols(t1_df).iloc[0]
        tender_period = str(t10.get("period", "")).strip().upper()
        if tender_period == "" or tender_period not in TP_START:
            raise KeyError(f"cod={k}: period inválido/no mapeable en Period_tender: '{tender_period}'")
        y0, m0 = TP_START[tender_period]
        typetech = norm_typetech(t10.get("typetech", ""))
        fuel_raw = norm_fuel(t10.get("fuel", ""))
        dist = float(pd.to_numeric(t10.get("distancefuel", 25.0), errors="coerce") or 25.0)
        is_no_ren = ("NO RENOVABLE" in typetech) or (typetech == "RECURSO NO RENOVABLE")

        for (y, m) in YM_h:
            j = season_year_index(int(y), int(m), int(y0), int(m0))
            idx = ppi_ratio(j)
            oym_j = oym_base * idx

            if is_no_ren:
                if fuel_raw not in FUEL_CASE:
                    raise KeyError(f"cod={k}: OT_T1.fuel no reconocido: '{fuel_raw}'")
                case = FUEL_CASE[fuel_raw]
                if case == "BUNKER":
                    fb = fuel_price(y, "bunker")
                    peo = ctung * fb + ci + oym_j
                elif case == "CARBON":
                    fc = fuel_price(y, "carbon")
                    peo = ctung * fc + citt_carbon(dist) + oym_j
                elif case == "GAS":
                    fg = fuel_price(y, "gasnatural")
                    peo = ctung * (fg + fagn + fgsc) + ci + oym_j
                elif case in ("COQUE", "MIX"):
                    fc = fuel_price(y, "carbon")
                    fcp = fuel_price(y, "coque")
                    peo = (
                        ctung * ((1.0 - pcp) * fc + pcp * fcp)
                        + (1.0 - pcp) * citt_carbon(dist)
                        + pcp * citt_coque(dist)
                        + oym_j
                    )
                else:
                    peo = peo_base + oym_j
            else:
                peo = peo_base + oym_j

            PEO_KYM[(int(k), int(y), int(m))] = float(peo)

    PEO_stage = {}
    req_ym = df_req_4320[["stage", "year", "month"]].copy()
    req_ym["stage"] = req_ym["stage"].astype(int)
    req_ym["year"] = req_ym["year"].astype(int)
    req_ym["month"] = req_ym["month"].astype(int)

    for r in req_ym.itertuples(index=False):
        s = int(r.stage)
        y = int(r.year)
        mo = int(r.month)
        for k in K_list:
            PEO_stage[(int(k), int(s))] = float(PEO_KYM.get((int(k), y, mo), 0.0))

    print(f"PEO_stage construido | keys={len(PEO_stage)} (esperado {len(K_list)}*{len(S_list)}={len(K_list)*len(S_list)})")
    return PEO_KYM, PEO_stage

# %%
PEO_KYM, PEO_stage = build_peo_stage(
    df_offer_meta=df_offer_meta,
    offers_by_bidder=offers_by_bidder,
    df_req_4320=df_req_4320,
    df_period_tender=df_period_tender,
    df_reffuelprice=df_reffuelprice,
    df_otherfuelvar=df_otherfuelvar,
    season_start_month=5
)

CE_stage = PEO_stage

# %%
# =========================
# STAGE 1: EG CAPACITY AWARD
# =========================
meta = df_offer_meta.copy()
meta["cod"] = meta["cod"].astype(int)
meta["contractype"] = meta["contractype"].astype(str).str.upper()

K_EG = sorted(meta.loc[meta["contractype"].eq("EG"), "cod"].astype(int).unique().tolist())
if len(K_EG) == 0:
    raise ValueError("No hay oferentes EG en df_offer_meta.")

S = sorted(df_req_4320["stage"].astype(int).unique().tolist())
R = df_req_4320.set_index("stage")["requirement_total"].astype(float).to_dict()

df_w = df_req_4320[["stage", "year", "month"]].drop_duplicates().copy()
df_w["stage"] = df_w["stage"].astype(int)
df_w["w_days"] = df_w.apply(lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1], axis=1).astype(float)
W = df_w.set_index("stage")["w_days"].to_dict()

installed = meta.set_index("cod")["installedcapacity"].fillna(0).astype(float).to_dict()
EG_CAP = 150.0
eg_ub = {int(k): float(min(EG_CAP, float(installed.get(int(k), 0.0)))) for k in K_EG}
EG_AVAIL = float(sum(eg_ub.values()))
EG_TARGET = float(min(EG_CAP, EG_AVAIL))

print("K_EG =", K_EG)
print("installed(EG) =", {k: float(installed.get(k, 0.0)) for k in K_EG})
print("eg_ub =", eg_ub)
print(f"EG_AVAIL={EG_AVAIL:.6f} | EG_TARGET={EG_TARGET:.6f}")

if EG_TARGET <= 0:
    raise ValueError("EG_TARGET<=0: no hay capacidad instalada EG disponible para adjudicar.")

if "CE_stage" in globals():
    PEO_stage = CE_stage
elif "PEO_stage" in globals():
    PEO_stage = PEO_stage
else:
    raise ValueError("No existe CE_stage ni PEO_stage en memoria. Necesito PEO_stage[(k,s)] USD/MWh.")

if "df_profiles_long" not in globals() or df_profiles_long is None or len(df_profiles_long) == 0:
    raise ValueError("No existe df_profiles_long (perfiles). Ejecutá build_offer_profiles_4320_strict primero.")

tmpA = df_profiles_long.copy()
tmpA["cod"] = tmpA["cod"].astype(int)
tmpA["stage"] = tmpA["stage"].astype(int)
tmpA["a_per_mw"] = tmpA["a_per_mw"].astype(float)
tmpA = tmpA[tmpA["cod"].isin(K_EG)]
A_dict_local = tmpA.set_index(["cod", "stage"])["a_per_mw"].to_dict()

if len(A_dict_local) == 0:
    raise ValueError("A_dict_local vacío para EG. Revisar df_profiles_long.")

m1 = pyo.ConcreteModel("STAGE1_EG_ENERGY_COST")
m1.K = pyo.Set(initialize=K_EG, ordered=True)
m1.S = pyo.Set(initialize=S, ordered=True)

m1.A = pyo.Param(
    m1.K, m1.S,
    initialize=lambda m, k, s: float(A_dict_local.get((int(k), int(s)), 0.0)),
    within=pyo.NonNegativeReals
)
m1.PEO = pyo.Param(
    m1.K, m1.S,
    initialize=lambda m, k, s: float(PEO_stage.get((int(k), int(s)), 0.0)),
    within=pyo.NonNegativeReals
)
m1.W = pyo.Param(
    m1.S,
    initialize=lambda m, s: float(W.get(int(s), 30.0)),
    within=pyo.PositiveReals
)

m1.z = pyo.Var(m1.K, within=pyo.Binary)
m1.g = pyo.Var(m1.K, within=pyo.NonNegativeReals)

def _g_ub(m, k):
    return m.g[k] <= float(eg_ub.get(int(k), 0.0)) * m.z[k]
m1.g_ub = pyo.Constraint(m1.K, rule=_g_ub)

def _eg_target(m):
    return sum(m.g[k] for k in m.K) == float(EG_TARGET)
m1.eg_target = pyo.Constraint(rule=_eg_target)

def _obj(m):
    return sum(m.PEO[k, s] * (m.A[k, s] * m.g[k]) * m.W[s] for k in m.K for s in m.S)
m1.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

solver = pyo.SolverFactory("highs")
res = solver.solve(m1, tee=True)

g_sol_EG = {int(k): float(pyo.value(m1.g[k]) or 0.0) for k in m1.K}
z_sol_EG = {int(k): int(round(pyo.value(m1.z[k]) or 0.0)) for k in m1.K}

print("STAGE1 | sum g =", sum(g_sol_EG.values()), "target=", EG_TARGET)
print("STAGE1 | g =", g_sol_EG)
print("STAGE1 | z =", z_sol_EG)
print("STAGE1 | Obj (energy cost) =", float(pyo.value(m1.obj)))

EG_supply = {
    int(s): float(sum(float(A_dict_local.get((int(k), int(s)), 0.0)) * float(g_sol_EG.get(int(k), 0.0)) for k in g_sol_EG.keys()))
    for s in S
}
R_net = {int(s): max(0.0, float(R[int(s)]) - float(EG_supply[int(s)])) for s in S}
R_NET = R_net.copy()

E_EG_total_MWh = float(sum(EG_supply[int(s)] * float(W[int(s)]) for s in S))

print("POST1 | max EG_supply =", max(EG_supply.values()) if EG_supply else 0.0)
print("POST1 | min R_net     =", min(R_net.values()) if R_net else 0.0)
print("POST1 | max R_net     =", max(R_net.values()) if R_net else 0.0)
print("POST1 | E_EG_total_MWh (ponderado por días) =", E_EG_total_MWh)

df_stage_out = df_req_4320[["stage", "year", "month", "hour", "requirement_total"]].copy()
df_stage_out["EG_supply"] = df_stage_out["stage"].astype(int).map(EG_supply).fillna(0.0)
df_stage_out["R_net"] = df_stage_out["stage"].astype(int).map(R_net).fillna(df_stage_out["requirement_total"])

# %%
# =========================
# SALIDA: ADJUDICACIÓN EG
# =========================
df_awarded_eg = pd.DataFrame([
    {
        "cod": int(k),
        "bidder": df_offer_meta.set_index("cod").loc[int(k), "bidder"],
        "g_awarded_MW": float(g_sol_EG.get(int(k), 0.0)),
        "installed_MW": float(installed.get(int(k), 0.0)),
        "share_%": 100.0 * float(g_sol_EG.get(int(k), 0.0)) / float(EG_TARGET) if EG_TARGET > 0 else 0.0
    }
    for k in K_EG
])

df_awarded_eg = df_awarded_eg.sort_values("g_awarded_MW", ascending=False)

print("\n=== ADJUDICACIÓN BLOQUE EG (Stage 1) ===")
print(df_awarded_eg.to_string(index=False))
print("\nResumen:")
print("Total adjudicado EG (MW) =", df_awarded_eg["g_awarded_MW"].sum())
print("Target EG (MW) =", EG_TARGET)

# %%
# ============================================================
# STAGE 2 — PYOMO MILP (OCE + DCC)
# ============================================================
S = sorted(df_req_4320["stage"].astype(int).unique().tolist())
K = sorted(df_offer_meta["cod"].astype(int).unique().tolist())

R_NET = globals().get("R_NET", None)
if R_NET is None:
    R_NET = df_req_4320.set_index("stage")["requirement_total"].astype(float).to_dict()

df_w = df_req_4320[["stage", "year", "month"]].drop_duplicates().copy()
df_w["stage"] = df_w["stage"].astype(int)
df_w["w_days"] = df_w.apply(lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1], axis=1).astype(float)
W = df_w.set_index("stage")["w_days"].to_dict()

stage_to_pm = df_req_4320.set_index("stage")["mechanism_period"].astype(str).to_dict()
P = sorted(df_req_4320["mechanism_period"].astype(str).unique().tolist())

tmpA = df_profiles_long.copy()
tmpA["cod"] = tmpA["cod"].astype(int)
tmpA["stage"] = tmpA["stage"].astype(int)
tmpA["a_per_mw"] = tmpA["a_per_mw"].astype(float)
A_dict = tmpA.set_index(["cod", "stage"])["a_per_mw"].to_dict()

meta = df_offer_meta.copy()
meta["cod"] = meta["cod"].astype(int)
meta["contractype"] = meta["contractype"].astype(str).str.upper()
meta["blockbidded"] = meta.get("blockbidded", "").fillna("").astype(str).str.upper()

ctype = meta.set_index("cod")["contractype"].to_dict()
block = meta.set_index("cod")["blockbidded"].to_dict()

K_OCE = [k for k in K if ctype.get(k) == "OCE"]
K_DCC = [k for k in K if ctype.get(k) == "DCC"]
K2 = sorted(set(K_OCE + K_DCC))

K_BASE = [k for k in K2 if block.get(k, "") == "BASE"]
K_COMP = [k for k in K2 if block.get(k, "").startswith("COMP")]

pgmin = meta.set_index("cod")["potgaranmin"].fillna(0).astype(float).to_dict()
pgmax = meta.set_index("cod")["potgaranmax"].fillna(0).astype(float).to_dict()

oe_rows = []
for cod, pack in offers_by_bidder.items():
    df = pack.get("tables", {}).get("OE")
    if df is None or len(df) == 0:
        continue
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "ppg" in d.columns:
        oe_rows.append({"cod": int(cod), "ppg": float(d["ppg"].iloc[0])})
df_oe = pd.DataFrame(oe_rows).set_index("cod") if len(oe_rows) else pd.DataFrame(columns=["ppg"])
PPG = df_oe["ppg"].astype(float).to_dict() if len(df_oe) else {}

PEO_stage = CE_stage
VIRTUAL_COST = float(globals().get("VIRTUAL_COST", 250.0))

pw = df_requirement_power.copy()
pw["mechanism_period"] = pw["mechanism_period"].astype(str).str.strip()
pw["Type"] = pw["Type"].astype(str).str.strip().str.upper()

cap_base_pm = (
    pw.loc[pw["Type"].eq("BASE"), ["mechanism_period", "power_requirement"]]
    .set_index("mechanism_period")["power_requirement"].astype(float).to_dict()
)
cap_comp_pm = (
    pw.loc[pw["Type"].isin(["COMPLEMENTARY", "COMPLEMENTARIO", "COMPLEMENTARIA"]), ["mechanism_period", "power_requirement"]]
    .set_index("mechanism_period")["power_requirement"].astype(float).to_dict()
)

# pm_dates = df_period_mechanism.copy()
# pm_dates["mechanism_period"] = pm_dates["mechanism_period"].astype(str).str.strip()
# pm_dates["start_date"] = pd.to_datetime(pm_dates["start_date"], dayfirst=True, errors="coerce")
# pm_dates["end_date"] = pd.to_datetime(pm_dates["end_date"], dayfirst=True, errors="coerce")
# pm_end = pm_dates.set_index("mechanism_period")["end_date"].to_dict()

# start_date_k = meta.set_index("cod")["start_date"].to_dict()
# ACTIVE = {}
# for k in K2:
#     sd = pd.to_datetime(start_date_k.get(int(k), pd.NaT))
#     for p in P:
#         ed = pm_end.get(str(p), pd.NaT)
#         ACTIVE[(int(k), str(p))] = 1.0 if (pd.notna(sd) and pd.notna(ed) and sd <= ed) else 0.0

pm_dates = df_period_mechanism.copy()
pm_dates["mechanism_period"] = pm_dates["mechanism_period"].astype(str).str.strip()
pm_dates["start_date"] = pd.to_datetime(pm_dates["start_date"], dayfirst=True, errors="coerce")
pm_dates["end_date"] = pd.to_datetime(pm_dates["end_date"], dayfirst=True, errors="coerce")

pm_start = pm_dates.set_index("mechanism_period")["start_date"].to_dict()
pm_end = pm_dates.set_index("mechanism_period")["end_date"].to_dict()

start_date_k = meta.set_index("cod")["start_date"].to_dict()
end_date_k = meta.set_index("cod")["end_date_exclusive"].to_dict()

ACTIVE = {}
for k in K2:
    sd = pd.to_datetime(start_date_k.get(int(k), pd.NaT))
    edx = pd.to_datetime(end_date_k.get(int(k), pd.NaT))  # exclusive
    for p in P:
        ps = pd.to_datetime(pm_start.get(str(p), pd.NaT))
        pe = pd.to_datetime(pm_end.get(str(p), pd.NaT))
        ACTIVE[(int(k), str(p))] = 1.0 if (
            pd.notna(sd) and pd.notna(edx) and pd.notna(ps) and pd.notna(pe)
            and (sd <= pe)
            and (edx > ps)
        ) else 0.0

bd = df_bidders.copy()
bd.columns = [str(c).strip() for c in bd.columns]
bd["cod"] = pd.to_numeric(bd["cod"], errors="raise").astype(int)

if "tipo" not in [c.lower() for c in bd.columns]:
    raise KeyError(f"df_bidders debe incluir columna 'tipo'. Columnas: {list(bd.columns)}")

tipo_col = [c for c in bd.columns if c.lower() == "tipo"][0]
bd[tipo_col] = bd[tipo_col].astype(str).str.strip().str.upper()
tipo_map = bd.set_index("cod")[tipo_col].to_dict()

#K_EXIST = [k for k in K2 if str(tipo_map.get(int(k), "")).upper() == "EXISTENTE"]
K_EXIST = [
    k for k in K2
    if str(tipo_map.get(int(k), "")).upper() in {
        "EXISTENTE_CON_INV_ADIC",
        "EXISTENTE_SIN_INV_ADIC"
    }
]

K_EXIST_CON_INV = [
    k for k in K2
    if str(tipo_map.get(int(k), "")).upper() == "EXISTENTE_CON_INV_ADIC"
]

K_EXIST_SIN_INV = [
    k for k in K2
    if str(tipo_map.get(int(k), "")).upper() == "EXISTENTE_SIN_INV_ADIC"
]

K_FUTURA = [
    k for k in K2
    if str(tipo_map.get(int(k), "")).upper() == "FUTURA"
]
EXIST_CAP = 500.0

# --- Crear modelo primero ---
m = pyo.ConcreteModel()

m.S = pyo.Set(initialize=S, ordered=True)
m.K = pyo.Set(initialize=K2, ordered=True)
m.P = pyo.Set(initialize=P, ordered=True)

m.K_OCE = pyo.Set(initialize=[k for k in K2 if k in K_OCE])
m.K_DCC = pyo.Set(initialize=[k for k in K2 if k in K_DCC])
m.K_BASE = pyo.Set(initialize=K_BASE)
m.K_COMP = pyo.Set(initialize=K_COMP)
m.K_EXIST = pyo.Set(initialize=K_EXIST)

# --- nuevos sets ---
m.K_EXIST_CON_INV = pyo.Set(initialize=K_EXIST_CON_INV)
m.K_EXIST_SIN_INV = pyo.Set(initialize=K_EXIST_SIN_INV)
m.K_FUTURA = pyo.Set(initialize=K_FUTURA)

# m.K_OCE = pyo.Set(initialize=[k for k in K2 if k in K_OCE])
# m.K_DCC = pyo.Set(initialize=[k for k in K2 if k in K_DCC])
# m.K_BASE = pyo.Set(initialize=K_BASE)
# m.K_COMP = pyo.Set(initialize=K_COMP)
# m.K_EXIST = pyo.Set(initialize=K_EXIST)

m.R = pyo.Param(m.S, initialize=lambda m, s: float(R_NET.get(int(s), 0.0)), within=pyo.NonNegativeReals)
m.W = pyo.Param(m.S, initialize=lambda m, s: float(W.get(int(s), 30.0)), within=pyo.PositiveReals)

m.A = pyo.Param(
    m.K, m.S,
    initialize=lambda m, k, s: float(A_dict.get((int(k), int(s)), 0.0)),
    within=pyo.NonNegativeReals
)

m.ACTIVE = pyo.Param(
    m.K, m.P,
    initialize=lambda m, k, p: float(ACTIVE.get((int(k), str(p)), 0.0)),
    within=pyo.NonNegativeReals
)

m.CAP_BASE = pyo.Param(m.P, initialize=lambda m, p: float(cap_base_pm.get(str(p), 0.0)), within=pyo.NonNegativeReals)
m.CAP_COMP = pyo.Param(m.P, initialize=lambda m, p: float(cap_comp_pm.get(str(p), 0.0)), within=pyo.NonNegativeReals)

m.PEO = pyo.Param(
    m.K, m.S,
    initialize=lambda m, k, s: float(PEO_stage.get((int(k), int(s)), 0.0)),
    within=pyo.NonNegativeReals
)

m.z = pyo.Var(m.K, within=pyo.Binary)
m.x = pyo.Var(m.K, within=pyo.NonNegativeReals)
m.y = pyo.Var(m.K, m.S, within=pyo.NonNegativeReals)
m.v = pyo.Var(m.S, within=pyo.NonNegativeReals)

# ============================================================
# Balance por tramos:
# - hasta abril 2035: se permite sobreasignación (>=)
# - desde mayo 2035: cumplimiento exacto (==)
# ============================================================
CUT_OFF_DATE = pd.Timestamp("2035-05-01")

stage_dates_df = df_req_4320[["stage", "year", "month"]].drop_duplicates().copy()
stage_dates_df["stage"] = stage_dates_df["stage"].astype(int)
stage_dates_df["date"] = pd.to_datetime(
    dict(year=stage_dates_df["year"], month=stage_dates_df["month"], day=1)
)

stage_date_map = stage_dates_df.set_index("stage")["date"].to_dict()

S_RELAX = [
    int(s) for s in S
    if pd.notna(stage_date_map.get(int(s), pd.NaT))
    and stage_date_map[int(s)] < CUT_OFF_DATE
]

S_STRICT = [
    int(s) for s in S
    if pd.notna(stage_date_map.get(int(s), pd.NaT))
    and stage_date_map[int(s)] >= CUT_OFF_DATE
]

m.S_RELAX = pyo.Set(initialize=S_RELAX, ordered=True)
m.S_STRICT = pyo.Set(initialize=S_STRICT, ordered=True)


def link_x_lb(m, k):
    return m.x[k] >= float(pgmin.get(int(k), 0.0)) * m.z[k]
m.link_x_lb = pyo.Constraint(m.K, rule=link_x_lb)

def link_x_ub(m, k):
    return m.x[k] <= float(pgmax.get(int(k), 0.0)) * m.z[k]
m.link_x_ub = pyo.Constraint(m.K, rule=link_x_ub)

def dcc_rule(m, k, s):
    if int(k) not in K_DCC:
        return pyo.Constraint.Skip
    return m.y[k, s] == m.A[k, s] * m.x[k]
m.dcc = pyo.Constraint(m.K, m.S, rule=dcc_rule)

# def oce_rule(m, k, s):
#     if int(k) not in K_OCE:
#         return pyo.Constraint.Skip
#     return m.y[k, s] <= m.x[k]
# m.oce = pyo.Constraint(m.K, m.S, rule=oce_rule)

# def oce_rule(m, k, s):
#     if int(k) not in K_OCE:
#         return pyo.Constraint.Skip
#     return m.y[k, s] <= m.A[k, s] * m.x[k]
# m.oce = pyo.Constraint(m.K, m.S, rule=oce_rule)
def oce_rule(m, k, s):
    if int(k) not in K_OCE:
        return pyo.Constraint.Skip
    if str(block.get(int(k), "")).upper() == "BASE":
        return m.y[k, s] == m.A[k, s] * m.x[k]
    else:
        return m.y[k, s] <= m.A[k, s] * m.x[k]
m.oce = pyo.Constraint(m.K, m.S, rule=oce_rule)


# def balance_rule(m, s):
#     return sum(m.y[k, s] for k in m.K) + m.v[s] == m.R[s]
# m.balance = pyo.Constraint(m.S, rule=balance_rule)

def balance_relax_rule(m, s):
    return sum(m.y[k, s] for k in m.K) + m.v[s] >= m.R[s]
m.balance_relax = pyo.Constraint(m.S_RELAX, rule=balance_relax_rule)

def balance_strict_rule(m, s):
    return sum(m.y[k, s] for k in m.K) + m.v[s] == m.R[s]
m.balance_strict = pyo.Constraint(m.S_STRICT, rule=balance_strict_rule)

def cap_base_power_rule(m, p):
    return sum(m.x[k] * m.ACTIVE[k, p] for k in m.K_BASE) <= m.CAP_BASE[p]
m.cap_base_power = pyo.Constraint(m.P, rule=cap_base_power_rule)

def cap_comp_power_rule(m, p):
    return sum(m.x[k] * m.ACTIVE[k, p] for k in m.K_COMP) <= m.CAP_COMP[p]
m.cap_comp_power = pyo.Constraint(m.P, rule=cap_comp_power_rule)

def cap_exist_total_rule(m):
    return sum(m.x[k] for k in m.K_EXIST) <= float(EXIST_CAP)
m.cap_exist_total = pyo.Constraint(rule=cap_exist_total_rule)

def obj_rule(m):
    pot = sum(float(PPG.get(int(k), 0.0)) * m.x[k] for k in m.K)
    ene = sum(m.PEO[k, s] * m.y[k, s] * m.W[s] for k in m.K for s in m.S)
    vir = sum(VIRTUAL_COST * m.v[s] * m.W[s] for s in m.S)
    return pot + ene + vir
m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

import sys
print("Python executable:", sys.executable)

try:
    import highspy
    print("highspy import OK:", highspy.__file__)
except Exception as e:
    print("highspy import FAILED:", repr(e))

solver_test = pyo.SolverFactory("highs")
print("solver.available() =", solver_test.available())
print("bool(solver.available()) =", bool(solver_test.available()))

solver = pyo.SolverFactory("highs")
results = solver.solve(m, tee=True)

x_sol = {int(k): float(pyo.value(m.x[k]) or 0.0) for k in m.K}
z_sol = {int(k): int(round(pyo.value(m.z[k]) or 0.0)) for k in m.K}
v_sol = {int(s): float(pyo.value(m.v[s]) or 0.0) for s in m.S}

y_rows = [(int(s), int(k), float(pyo.value(m.y[k, s]) or 0.0)) for k in K2 for s in S]
df_y = pd.DataFrame(y_rows, columns=["stage", "cod", "y"]).pivot(index="stage", columns="cod", values="y").fillna(0.0)

print("Obj:", float(pyo.value(m.obj)))
print("Selected:", {k: v for k, v in z_sol.items() if v == 1})
print("Max virtual:", max(v_sol.values()) if v_sol else 0.0)
print("Existentes adjudicados (MW):", sum(x_sol.get(k, 0.0) for k in K_EXIST), " / 500")

# %%
# =========================
# SALIDA: ADJUDICACIÓN STAGE 2
# =========================
meta2 = df_offer_meta.copy()
meta2["cod"] = meta2["cod"].astype(int)
meta2["contractype"] = meta2["contractype"].astype(str).str.upper()

# df_awarded_stage2 = pd.DataFrame([
#     {
#         "cod": int(k),
#         "bidder": meta2.set_index("cod").loc[int(k), "bidder"] if "bidder" in meta2.columns else f"Oferente {k}",
#         "contractype": meta2.set_index("cod").loc[int(k), "contractype"],
#         "block": meta2.set_index("cod").loc[int(k), "blockbidded"] if "blockbidded" in meta2.columns else "",
#         "x_awarded_MW": float(x_sol.get(int(k), 0.0)),
#         "selected_z": int(z_sol.get(int(k), 0)),
#         "PPG": float(PPG.get(int(k), 0.0)),
#         "PMon_input": float(PMON_INPUT_MAP.get(int(k), np.nan))
#     }
#     for k in K2
# ])
df_awarded_stage2 = pd.DataFrame([
    {
        "cod": int(k),
        "bidder": meta2.set_index("cod").loc[int(k), "bidder"] if "bidder" in meta2.columns else f"Oferente {k}",
        "tipo_input": meta2.set_index("cod").loc[int(k), "tipo_input"] if "tipo_input" in meta2.columns else "",
        "contractype": meta2.set_index("cod").loc[int(k), "contractype"],
        "block": meta2.set_index("cod").loc[int(k), "blockbidded"] if "blockbidded" in meta2.columns else "",
        "x_awarded_MW": float(x_sol.get(int(k), 0.0)),
        "selected_z": int(z_sol.get(int(k), 0)),
        "PPG": float(PPG.get(int(k), 0.0)),
        "PMon_input": float(PMON_INPUT_MAP.get(int(k), np.nan)),
        "max_contract_years": int(meta2.set_index("cod").loc[int(k), "max_contract_years"]) if "max_contract_years" in meta2.columns else np.nan,
        "start_date": meta2.set_index("cod").loc[int(k), "start_date"] if "start_date" in meta2.columns else pd.NaT,
        "end_date_exclusive": meta2.set_index("cod").loc[int(k), "end_date_exclusive"] if "end_date_exclusive" in meta2.columns else pd.NaT,
    }
    for k in K2
])


df_awarded_stage2 = df_awarded_stage2.sort_values("x_awarded_MW", ascending=False)

print("\n=== ADJUDICACIÓN STAGE 2 (OCE + DCC) ===")
print(df_awarded_stage2)
print("\nResumen Stage 2:")
print("Total MW adjudicados (OCE+DCC) =", df_awarded_stage2["x_awarded_MW"].sum())
print("Seleccionados =", df_awarded_stage2.loc[df_awarded_stage2["selected_z"] == 1, "cod"].tolist())
print("\nMW por tipo de contrato:")
print(df_awarded_stage2.groupby("contractype")["x_awarded_MW"].sum())
print("\nMW por bloque ofertado:")
print(df_awarded_stage2.groupby("block")["x_awarded_MW"].sum())

# %%
# ============================================================
# PLOTTER ACTUALIZADO (Stage 1 EG + Stage 2 OCE/DCC)
# - CURRENT_ROUND solo se incorpora al título
# ============================================================
VIRTUAL_BAR_COLOR = "#808080"

def build_df_y_eg_from_stage1(
    S: list,
    K_EG: list,
    A_dict: dict,
    g_sol_EG: dict,
) -> pd.DataFrame:
    data = {}
    for k in K_EG:
        k = int(k)
        gk = float(g_sol_EG.get(k, 0.0))
        if gk <= 0:
            continue
        data[k] = [float(A_dict.get((k, int(s)), 0.0)) * gk for s in S]
    if not data:
        return pd.DataFrame(index=pd.Index(S, name="stage"))
    df = pd.DataFrame(data, index=pd.Index(S, name="stage"))
    return df

def plot_window_12m_supply_vs_req_two_stage(
    year: int,
    month: int,
    df_req_4320: pd.DataFrame,
    df_y_stage_cod: pd.DataFrame,
    df_offer_meta: pd.DataFrame,
    A_dict: dict,
    g_sol_EG: dict,
    v_sol: dict,
    current_round: int,
    title_prefix: str = "",
    tick_step: int = 48
):
    start = pd.Timestamp(year=year, month=month, day=1)
    end_12m = start + pd.DateOffset(months=12)

    cal = df_req_4320[["stage", "year", "month", "hour"]].copy()
    cal["date"] = pd.to_datetime(dict(year=cal["year"], month=cal["month"], day=1))

    win = cal[(cal["date"] >= start) & (cal["date"] < end_12m)].copy()
    if win.empty:
        raise ValueError("Ventana vacía: revisar year/month vs horizonte.")

    stage_win = win["stage"].astype(int).tolist()
    req_total = df_req_4320.set_index("stage").loc[stage_win, "requirement_total"].astype(float).values

    meta = df_offer_meta.copy()
    meta["cod"] = meta["cod"].astype(int)
    meta["contractype"] = meta["contractype"].astype(str).str.upper()
    meta["blockbidded"] = meta.get("blockbidded", "").fillna("").astype(str).str.upper()

    base_cods = meta.loc[meta["contractype"].isin(["DCC", "OCE"]) & (meta["blockbidded"] == "BASE"), "cod"].tolist()
    comp_cods = meta.loc[meta["contractype"].isin(["DCC", "OCE"]) & (meta["blockbidded"].str.startswith("COMP")), "cod"].tolist()
    eg_cods = meta.loc[meta["contractype"].eq("EG"), "cod"].tolist()

    cols_stage2 = [c for c in (base_cods + comp_cods) if c in df_y_stage_cod.columns]
    df_stage2_win = df_y_stage_cod.loc[stage_win, cols_stage2].copy() if cols_stage2 else pd.DataFrame(index=stage_win)

    S_all = sorted(df_req_4320["stage"].astype(int).unique().tolist())
    K_EG = [int(k) for k in eg_cods]
    df_y_eg_all = build_df_y_eg_from_stage1(S_all, K_EG, A_dict, g_sol_EG)
    df_eg_win = df_y_eg_all.loc[stage_win, [c for c in df_y_eg_all.columns]] if len(df_y_eg_all.columns) else pd.DataFrame(index=stage_win)

    v_win = np.array([float(v_sol.get(int(s), 0.0)) for s in stage_win], dtype=float)

    prof = pd.DataFrame(index=stage_win)
    for c in cols_stage2:
        prof[c] = df_stage2_win[c].values
    for c in df_eg_win.columns:
        prof[c] = df_eg_win[c].values

    stage2_supply_win = df_stage2_win.sum(axis=1).values if len(cols_stage2) else np.zeros(len(stage_win))
    eg_supply_win = df_eg_win.sum(axis=1).values if len(df_eg_win.columns) else np.zeros(len(stage_win))
    #req_residual = np.maximum(0.0, req_total - stage2_supply_win - eg_supply_win)
    #prof["VIRTUAL"] = np.minimum(v_win, req_residual)
    req_residual = req_total - stage2_supply_win - eg_supply_win
    prof["VIRTUAL"] = v_win
    
    
    req_net = np.maximum(0.0, req_total - eg_supply_win)

    labels = (
        win["year"].astype(str) + "-" +
        win["month"].astype(str).str.zfill(2) + " h" +
        win["hour"].astype(str).str.zfill(2)
    ).tolist()

    # plt.figure(figsize=(16, 5))
    # bottom = np.zeros(len(stage_win))

    # for cod in prof.columns:
    #     plt.bar(range(len(stage_win)), prof[cod].values, bottom=bottom, label=str(cod))
    #     bottom += prof[cod].values
    plt.figure(figsize=(16, 5))
    bottom = np.zeros(len(stage_win))

    for cod in prof.columns:
        if str(cod).upper() == "VIRTUAL":
            plt.bar(
                range(len(stage_win)),
                prof[cod].values,
                bottom=bottom,
                label=str(cod),
                color=VIRTUAL_BAR_COLOR
            )
        else:
            plt.bar(
                range(len(stage_win)),
                prof[cod].values,
                bottom=bottom,
                label=str(cod)
            )
        bottom += prof[cod].values


    plt.plot(range(len(stage_win)), req_total, linewidth=2, label="Req TOTAL")
    plt.plot(range(len(stage_win)), req_net, linewidth=2, label="Req NETO (TOTAL - EG)")

    plt.xticks(
        ticks=list(range(0, len(stage_win), tick_step)),
        labels=[labels[i] for i in range(0, len(stage_win), tick_step)],
        rotation=90
    )
    plt.ylim(bottom=0)
    plt.xlabel("Año-Mes-Hora (ventana)")
    plt.ylabel("MW (hora)")
    plt.title(f"{title_prefix} | Ronda {current_round} | Ventana 12 meses desde {start.strftime('%Y-%m')} (n={len(stage_win)} horas)")
    plt.legend(ncol=6)
    plt.tight_layout()
    plt.show()

# %%
# ---- RUN PLOT (Two-Stage: EG + OCE/DCC) ----
plot_window_12m_supply_vs_req_two_stage(
    year=2040,
    month=5,
    df_req_4320=df_req_4320,
    df_y_stage_cod=df_y,
    df_offer_meta=df_offer_meta,
    A_dict=A_dict,
    g_sol_EG=g_sol_EG,
    v_sol=v_sol,
    current_round=CURRENT_ROUND,
    title_prefix="PEG5",
    tick_step=48
)

# %%
# %%
# ============================================================
# POSTPROCESO FINAL (2 etapas): TABLA RESUMEN + CSV
# - PMon se toma DIRECTAMENTE desde List_Bidders.csv
# - La referencia para reducción es:
#     mayor PMon entre ofertas seleccionadas del Stage 2 (OCE/DCC), excluyendo VIRTUAL
# - Se agregan comprobadores para validar el target usado
# ============================================================
def _get_stage1(df_req_4320: pd.DataFrame, stage1: int = 1) -> int:
    S = sorted(pd.to_numeric(df_req_4320["stage"], errors="coerce").dropna().astype(int).unique().tolist())
    return int(stage1) if int(stage1) in S else int(S[0])

def _extract_ppg_from_oe(offers_by_bidder: dict) -> dict:
    ppg_map = {}
    for cod, pack in offers_by_bidder.items():
        oe = pack.get("tables", {}).get("OE")
        if oe is None or len(oe) == 0:
            continue
        d = _norm_cols(oe)
        if "ppg" not in d.columns:
            continue
        val = pd.to_numeric(d["ppg"].iloc[0], errors="coerce")
        if pd.notna(val):
            ppg_map[int(cod)] = float(val)
    return ppg_map

def _extract_block_from_ot_t1(offers_by_bidder: dict) -> dict:
    blk = {}
    for cod, pack in offers_by_bidder.items():
        t1 = pack.get("tables", {}).get("OT_T1")
        if t1 is None or len(t1) == 0:
            continue
        d = _norm_cols(t1)
        b = ""
        if "blockbidded" in d.columns:
            b = str(d["blockbidded"].iloc[0]).strip().upper()
        blk[int(cod)] = b
    return blk

MW_TOL = 1e-6
STAGE1_WANTED = 1

CSV_ADJ = "resultados_adjudicados.csv"
CSV_NO = "resultados_no_adjudicados.csv"
CSV_ALL = "resultados_todos.csv"

TOTAL_OBJETIVO_MW = 1400.0
VIRTUAL_COST = float(globals().get("VIRTUAL_COST", 10000.0))

if "g_sol_EG" not in globals():
    raise NameError("Falta g_sol_EG (Stage 1 EG). Defínelo antes del postproceso.")

meta = df_offer_meta.copy()
meta["cod"] = meta["cod"].astype(int)
meta["contractype"] = meta["contractype"].astype(str).str.upper()

bd = df_bidders.copy()
bd.columns = [str(c).strip().lower() for c in bd.columns]
bd["cod"] = pd.to_numeric(bd["cod"], errors="raise").astype(int)
bd["bidder"] = bd["bidder"].astype(str).str.strip()
bd["tipo"] = bd["tipo"].astype(str).str.strip().str.upper()
bd["preciomonomico"] = pd.to_numeric(bd["preciomonomico"], errors="raise").astype(float)

bidder_map = bd.set_index("cod")["bidder"].to_dict()
tipo_input_map = bd.set_index("cod")["tipo"].to_dict()
pmon_input_map = bd.set_index("cod")["preciomonomico"].to_dict()

s1_used = _get_stage1(df_req_4320, stage1=STAGE1_WANTED)
peo_map = {int(k): float(PEO_stage.get((int(k), int(s1_used)), np.nan)) for k in meta["cod"].unique().tolist()}
ppg_map = _extract_ppg_from_oe(offers_by_bidder)
block_ot1 = _extract_block_from_ot_t1(offers_by_bidder)

def _mw_awarded(cod: int, ct: str) -> float:
    ct = str(ct).upper()
    if ct == "EG":
        return float(g_sol_EG.get(int(cod), 0.0))
    return float(x_sol.get(int(cod), 0.0))

meta["MW Adjudicados"] = meta.apply(lambda r: _mw_awarded(int(r["cod"]), str(r["contractype"])), axis=1)

def _bloque_ofertado(cod: int, ct: str) -> str:
    ct = str(ct).upper()
    if ct == "EG":
        return "POTENCIA INSTALADA"
    b = ""
    if "blockbidded" in meta.columns:
        b = str(meta.loc[meta["cod"].eq(cod), "blockbidded"].iloc[0]) if (meta["cod"].eq(cod).any()) else ""
    b = str(b or "").strip().upper()
    if b == "" and int(cod) in block_ot1:
        b = str(block_ot1[int(cod)] or "").strip().upper()
    return b

meta["Bloque Ofertado"] = meta.apply(lambda r: _bloque_ofertado(int(r["cod"]), str(r["contractype"])), axis=1)
meta["PPG"] = meta.apply(
    lambda r: 0.0 if str(r["contractype"]).upper() == "EG" else float(ppg_map.get(int(r["cod"]), np.nan)),
    axis=1
)

# PMon source of truth from List_Bidders.csv
meta["PMon_input"] = meta["cod"].map(pmon_input_map).astype(float)
if meta["PMon_input"].isna().any():
    missing_pmon = meta.loc[meta["PMon_input"].isna(), "cod"].astype(int).tolist()
    raise ValueError(f"Falta preciomonomico en List_Bidders.csv para los cod: {missing_pmon}")

df_results_all = pd.DataFrame({
    "COD": meta["cod"].astype(int),
    "Oferente": meta["cod"].map(bidder_map).fillna(meta["cod"].astype(str)),
    "Tipo Input": meta["cod"].map(tipo_input_map).fillna(""),
    "Tipo de Contrato": meta["contractype"].astype(str),
    "Bloque Ofertado": meta["Bloque Ofertado"].astype(str),
    "MW Adjudicados": meta["MW Adjudicados"].astype(float),
    "PPG": meta["PPG"].astype(float),
    "PEO": meta["cod"].map(peo_map).astype(float),
    "PMon": meta["PMon_input"].astype(float),
    "Años Máximos": meta["max_contract_years"].astype(float),
    "Inicio Contrato": pd.to_datetime(meta["start_date"]),
    "Fin Contrato (exclusivo)": pd.to_datetime(meta["end_date_exclusive"]),
})

mw_total_adj_real = float(df_results_all["MW Adjudicados"].sum())
mw_virtual = float(max(0.0, TOTAL_OBJETIVO_MW - mw_total_adj_real))

row_virtual = pd.DataFrame([{
    "COD": 0,
    "Oferente": "OFERTA VIRTUAL",
    "Tipo Input": "VIRTUAL",
    "Tipo de Contrato": "VIRTUAL",
    "Bloque Ofertado": "VIRTUAL",
    "MW Adjudicados": mw_virtual,
    "PPG": 0.0,
    "PEO": float(VIRTUAL_COST),
    "PMon": float(VIRTUAL_COST),
}])

df_results_all = pd.concat([df_results_all, row_virtual], ignore_index=True)

num_cols = ["MW Adjudicados", "PPG", "PEO", "PMon"]
df_results_all[num_cols] = df_results_all[num_cols].round(2)

# ============================================================
# REFERENCIA CORRECTA PARA LA REDUCCIÓN
# mayor PMon entre ofertas seleccionadas Stage 2 (OCE/DCC), sin virtual
# ============================================================
df_results_all["MW Adjudicados"] = pd.to_numeric(df_results_all["MW Adjudicados"], errors="coerce").fillna(0.0)
df_results_all["PMon"] = pd.to_numeric(df_results_all["PMon"], errors="coerce")

mask_selected_stage2_real = (
    df_results_all["MW Adjudicados"].gt(MW_TOL)
    & df_results_all["Tipo de Contrato"].astype(str).str.upper().isin(["OCE", "DCC"])
    & df_results_all["PMon"].notna()
)

if mask_selected_stage2_real.sum() == 0:
    raise ValueError(
        "No hay ofertas seleccionadas OCE/DCC en Stage 2 para definir el precio monómico frontera."
    )

target_pmon = float(df_results_all.loc[mask_selected_stage2_real, "PMon"].max())

# inicializar columnas
df_results_all["USD/MWh a disminuir"] = np.nan
df_results_all["% mínimo a disminuir"] = np.nan

# calcular SOLO para ofertas reales NO seleccionadas OCE/DCC
mask_not_selected_stage2_real = (
    df_results_all["MW Adjudicados"].le(MW_TOL)
    & df_results_all["Tipo de Contrato"].astype(str).str.upper().isin(["OCE", "DCC"])
    & df_results_all["PMon"].notna()
)

df_results_all.loc[mask_not_selected_stage2_real, "USD/MWh a disminuir"] = (
    df_results_all.loc[mask_not_selected_stage2_real, "PMon"] - target_pmon
).clip(lower=0.0)

df_results_all.loc[mask_not_selected_stage2_real, "% mínimo a disminuir"] = (
    df_results_all.loc[mask_not_selected_stage2_real, "USD/MWh a disminuir"]
    / df_results_all.loc[mask_not_selected_stage2_real, "PMon"].replace(0.0, np.nan)
).fillna(0.0) * 100.0

# para seleccionados reales del Stage 2 dejar 0
df_results_all.loc[mask_selected_stage2_real, "USD/MWh a disminuir"] = 0.0
df_results_all.loc[mask_selected_stage2_real, "% mínimo a disminuir"] = 0.0

# EG y VIRTUAL sin reducción
mask_eg_or_virtual = df_results_all["Tipo de Contrato"].astype(str).str.upper().isin(["EG", "VIRTUAL"])
df_results_all.loc[mask_eg_or_virtual, ["USD/MWh a disminuir", "% mínimo a disminuir"]] = np.nan

df_results_all[["USD/MWh a disminuir", "% mínimo a disminuir"]] = (
    df_results_all[["USD/MWh a disminuir", "% mínimo a disminuir"]].round(2)
)

# reconstruir tablas finales
df_resumen = df_results_all[df_results_all["MW Adjudicados"] > MW_TOL].sort_values(
    ["Tipo de Contrato", "Bloque Ofertado", "COD"]
).reset_index(drop=True)

df_no_adj = df_results_all[df_results_all["MW Adjudicados"] <= MW_TOL].sort_values(
    ["Tipo de Contrato", "Bloque Ofertado", "COD"]
).reset_index(drop=True)

print(f"PEO tomado desde etapa: {s1_used}")
print(f"MW total adjudicados (reales: EG stage1 + OCE/DCC stage2): {mw_total_adj_real:.2f}")
print(f"MW oferta virtual agregada: {mw_virtual:.2f} (objetivo {TOTAL_OBJETIVO_MW:.2f})")
print(f"PMon real tomado desde List_Bidders.csv (preciomonomico)")
print(f"target_pmon_stage2_real = {target_pmon:.2f}")

print("\n=== OFERTAS SELECCIONADAS STAGE 2 USADAS COMO FRONTERA ===")
print(
    df_results_all.loc[
        mask_selected_stage2_real,
        ["COD", "Oferente", "Tipo de Contrato", "Bloque Ofertado", "MW Adjudicados", "PMon"]
    ].sort_values(["PMon", "COD"], ascending=[False, True]).to_string(index=False)
)

print("\n=== OFERENTES ADJUDICADOS ===")
print(df_resumen.to_string(index=False))

print("\n=== OFERENTES NO ADJUDICADOS ===")
print(df_no_adj.to_string(index=False))

# comprobador específico
df_reduction_check = df_results_all.loc[
    df_results_all["Tipo de Contrato"].astype(str).str.upper().isin(["OCE", "DCC"]),
    ["COD", "Oferente", "Tipo de Contrato", "MW Adjudicados", "PMon", "USD/MWh a disminuir", "% mínimo a disminuir"]
].copy()

print("\n=== CHECK REDUCCIÓN ===")
print(df_reduction_check.sort_values(["MW Adjudicados", "PMon"], ascending=[False, False]).to_string(index=False))

# %%
# ============================================================
# EXPORT CSV base
# ============================================================
#df_resumen.to_csv(CSV_ADJ, index=False, encoding="utf-8-sig")
#df_no_adj.to_csv(CSV_NO, index=False, encoding="utf-8-sig")
#df_results_all.to_csv(CSV_ALL, index=False, encoding="utf-8-sig")

#print("\nCSV generados:")
#print(" -", CSV_ADJ)
#print(" -", CSV_NO)
#print(" -", CSV_ALL)

# %%
# ============================================================
# MODULO UNICO DE EXPORTACIÓN A c:\PEG5MA\Resultados\
# ============================================================
RESULTS_DIR = os.path.join(BASE_PATH, "Resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

def _p(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)

print("Exportando resultados a:", RESULTS_DIR)

written = []
skipped = []

def _export_df(df: pd.DataFrame, filename: str, **to_csv_kwargs):
    path = _p(filename)
    df.to_csv(path, index=False, encoding="utf-8-sig", **to_csv_kwargs)
    written.append(filename)

try:
    if "req_vector_4320" in globals() and req_vector_4320 is not None:
        req_vector_4320.to_csv(_p("requirement_total_4320.csv"), header=["requirement_total"], encoding="utf-8-sig")
        written.append("requirement_total_4320.csv")
    elif "df_req_4320" in globals() and df_req_4320 is not None:
        _export_df(df_req_4320.copy(), "requirement_total_4320.csv")
    else:
        skipped.append(("requirement_total_4320.csv", "no existe req_vector_4320 ni df_req_4320"))
except Exception as e:
    skipped.append(("requirement_total_4320.csv", f"error: {e}"))

try:
    if "df_resumen" in globals() and df_resumen is not None:
        _export_df(df_resumen.copy(), "resultados_adjudicados.csv")
    else:
        skipped.append(("resultados_adjudicados.csv", "no existe df_resumen"))
except Exception as e:
    skipped.append(("resultados_adjudicados.csv", f"error: {e}"))

try:
    if "df_no_adj" in globals() and df_no_adj is not None:
        _export_df(df_no_adj.copy(), "resultados_no_adjudicados.csv")
    else:
        skipped.append(("resultados_no_adjudicados.csv", "no existe df_no_adj"))
except Exception as e:
    skipped.append(("resultados_no_adjudicados.csv", f"error: {e}"))

try:
    if "df_results_all" in globals() and df_results_all is not None:
        _export_df(df_results_all.copy(), "resultados_todos.csv")
    else:
        skipped.append(("resultados_todos.csv", "no existe df_results_all"))
except Exception as e:
    skipped.append(("resultados_todos.csv", f"error: {e}"))

try:
    if "df_awarded_eg" in globals() and df_awarded_eg is not None:
        _export_df(df_awarded_eg.copy(), "stage1_eg_adjudicacion.csv")
    else:
        skipped.append(("stage1_eg_adjudicacion.csv", "no existe df_awarded_eg"))
except Exception as e:
    skipped.append(("stage1_eg_adjudicacion.csv", f"error: {e}"))

try:
    if "df_awarded_stage2" in globals() and df_awarded_stage2 is not None:
        _export_df(df_awarded_stage2.copy(), "stage2_adjudicacion.csv")
    else:
        skipped.append(("stage2_adjudicacion.csv", "no existe df_awarded_stage2"))
except Exception as e:
    skipped.append(("stage2_adjudicacion.csv", f"error: {e}"))

try:
    if "df_y" in globals() and df_y is not None:
        df_y_out = df_y.copy()
        df_y_out.index.name = "stage"
        df_y_out.reset_index().round(2).to_csv(_p("stage2_y_entregas_stage_cod.csv"), index=False, encoding="utf-8-sig")
        written.append("stage2_y_entregas_stage_cod.csv")
    else:
        skipped.append(("stage2_y_entregas_stage_cod.csv", "no existe df_y"))
except Exception as e:
    skipped.append(("stage2_y_entregas_stage_cod.csv", f"error: {e}"))

try:
    if "df_y" in globals() and df_y is not None and "g_sol_EG" in globals():
        # reconstruir entregas EG por oferente
        meta_exp = df_offer_meta.copy()
        meta_exp["cod"] = meta_exp["cod"].astype(int)
        eg_cods_exp = meta_exp.loc[meta_exp["contractype"].astype(str).str.upper().eq("EG"), "cod"].astype(int).tolist()

        S_all_exp = sorted(df_req_4320["stage"].astype(int).unique().tolist())
        df_y_eg_exp = build_df_y_eg_from_stage1(S_all_exp, eg_cods_exp, A_dict, g_sol_EG)

        df_y_total = df_y.copy()
        for c in df_y_eg_exp.columns:
            df_y_total[int(c)] = df_y_eg_exp[c]

        df_y_total = df_y_total.reindex(sorted(df_y_total.columns), axis=1)
        df_y_total.index.name = "stage"
        df_y_total.reset_index().round(2).to_csv(_p("entregas_totales_stage_cod.csv"), index=False, encoding="utf-8-sig")
        written.append("entregas_totales_stage_cod.csv")
    else:
        skipped.append(("entregas_totales_stage_cod.csv", "faltan df_y o g_sol_EG"))
except Exception as e:
    skipped.append(("entregas_totales_stage_cod.csv", f"error: {e}"))

try:
    if "v_sol" in globals() and isinstance(v_sol, dict) and len(v_sol):
        pd.DataFrame({"stage": list(v_sol.keys()), "virtual_MW": list(v_sol.values())}) \
            .sort_values("stage") \
            .to_csv(_p("stage2_virtual_por_stage.csv"), index=False, encoding="utf-8-sig")
        written.append("stage2_virtual_por_stage.csv")
    else:
        skipped.append(("stage2_virtual_por_stage.csv", "no existe v_sol (dict) o está vacío"))
except Exception as e:
    skipped.append(("stage2_virtual_por_stage.csv", f"error: {e}"))

try:
    if "EG_supply" in globals() and isinstance(EG_supply, dict) and len(EG_supply):
        pd.DataFrame({"stage": list(EG_supply.keys()), "EG_supply_MW": list(EG_supply.values())}) \
            .sort_values("stage") \
            .to_csv(_p("stage1_EG_supply_por_stage.csv"), index=False, encoding="utf-8-sig")
        written.append("stage1_EG_supply_por_stage.csv")
    else:
        skipped.append(("stage1_EG_supply_por_stage.csv", "no existe EG_supply (dict) o está vacío"))
except Exception as e:
    skipped.append(("stage1_EG_supply_por_stage.csv", f"error: {e}"))

try:
    if "R_net" in globals() and isinstance(R_net, dict) and len(R_net):
        pd.DataFrame({"stage": list(R_net.keys()), "requirement_net_MW": list(R_net.values())}) \
            .sort_values("stage") \
            .to_csv(_p("requirement_net_4320.csv"), index=False, encoding="utf-8-sig")
        written.append("requirement_net_4320.csv")
    else:
        skipped.append(("requirement_net_4320.csv", "no existe R_net (dict) o está vacío"))
except Exception as e:
    skipped.append(("requirement_net_4320.csv", f"error: {e}"))

# PEO check
try:
    rows = [{"cod": int(k), "year": int(y), "month": int(m), "PEO": float(v)} for (k, y, m), v in PEO_KYM.items()]
    df_peo_check_all = pd.DataFrame(rows).sort_values(["cod", "year", "month"]).reset_index(drop=True)
    _m = df_offer_meta.copy()
    _m["cod"] = _m["cod"].astype(int)
    _m["bidder"] = _m.get("bidder", _m["cod"].astype(str))
    _m["contractype"] = _m.get("contractype", "")
    _m["contractype"] = _m["contractype"].astype(str).str.upper()
    _m["preciomonomico_input"] = _m["cod"].map(PMON_INPUT_MAP).astype(float)
    df_peo_check_all = df_peo_check_all.merge(
        _m[["cod", "bidder", "contractype", "preciomonomico_input"]].drop_duplicates("cod"),
        on="cod",
        how="left"
    )
    num_cols = df_peo_check_all.select_dtypes(include=[np.number]).columns
    df_peo_check_all[num_cols] = df_peo_check_all[num_cols].round(2)
    _export_df(df_peo_check_all.copy(), "PEO_check_all_oferentes.csv")
except Exception as e:
    skipped.append(("PEO_check_all_oferentes.csv", f"error: {e}"))

# Input bidders check
try:
    df_bidders_export = df_bidders.copy()
    _export_df(df_bidders_export, "List_Bidders_input_usado.csv")
except Exception as e:
    skipped.append(("List_Bidders_input_usado.csv", f"error: {e}"))

print("\n======================")
print("EXPORTACIÓN COMPLETA")
print("======================")
print("\nArchivos escritos (", len(written), "):")
for f in written:
    print(" -", _p(f))

print("\nArchivos NO generados (", len(skipped), "):")
for f, why in skipped:
    print(" -", f, "=>", why)


# %%
# ============================================================
# MODULO FINAL DE AJUSTE SOBRE CSV GENERADOS
# - Lee resultados_adjudicados.csv y resultados_no_adjudicados.csv
# - Toma como frontera el PMon máximo adjudicado excluyendo OFERTA VIRTUAL
# - Calcula:
#       USD/MWh a disminuir = max(0, PMon_frontera - PMon_i)
#       % mínimo a disminuir = USD/MWh a disminuir / PMon_i * 100
# - Sobrescribe ambos CSV y además guarda versiones *_ajustado.csv
# ============================================================

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(BASE_PATH, "Resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

PATH_ADJ = os.path.join(RESULTS_DIR, "resultados_adjudicados.csv")
PATH_NO  = os.path.join(RESULTS_DIR, "resultados_no_adjudicados.csv")

PATH_ADJ_AJUST = os.path.join(RESULTS_DIR, "resultados_adjudicados_ajustado.csv")
PATH_NO_AJUST  = os.path.join(RESULTS_DIR, "resultados_no_adjudicados_ajustado.csv")
PATH_CHECK     = os.path.join(RESULTS_DIR, "check_reduccion_desde_csv.csv")

if not os.path.exists(PATH_ADJ):
    raise FileNotFoundError(f"No existe: {PATH_ADJ}")
if not os.path.exists(PATH_NO):
    raise FileNotFoundError(f"No existe: {PATH_NO}")

df_adj_csv = pd.read_csv(PATH_ADJ, encoding="utf-8-sig")
df_no_csv  = pd.read_csv(PATH_NO, encoding="utf-8-sig")

# --- normalización de columnas esperadas ---
def _ensure_cols(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["COD", "Oferente", "Tipo de Contrato", "PMon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"En {name} faltan columnas: {missing}. Disponibles: {list(df.columns)}")

    if "USD/MWh a disminuir" not in df.columns:
        df["USD/MWh a disminuir"] = np.nan
    if "% mínimo a disminuir" not in df.columns:
        df["% mínimo a disminuir"] = np.nan

    df["PMon"] = pd.to_numeric(df["PMon"], errors="coerce")
    if "MW Adjudicados" in df.columns:
        df["MW Adjudicados"] = pd.to_numeric(df["MW Adjudicados"], errors="coerce").fillna(0.0)

    return df

df_adj_csv = _ensure_cols(df_adj_csv, "resultados_adjudicados.csv")
df_no_csv  = _ensure_cols(df_no_csv, "resultados_no_adjudicados.csv")

# --- frontera: máximo PMon adjudicado excluyendo virtual ---
mask_adj_real = (
    df_adj_csv["Tipo de Contrato"].astype(str).str.strip().str.upper().ne("VIRTUAL")
    & df_adj_csv["PMon"].notna()
)

if mask_adj_real.sum() == 0:
    raise ValueError("No hay ofertas adjudicadas reales en resultados_adjudicados.csv para definir la frontera.")

pmon_frontera = float(df_adj_csv.loc[mask_adj_real, "PMon"].max())

# --- función de cálculo ---
def _apply_gap_columns(df: pd.DataFrame, pmon_ref: float) -> pd.DataFrame:
    df = df.copy()

    mask_real = (
        df["Tipo de Contrato"].astype(str).str.strip().str.upper().ne("VIRTUAL")
        & df["PMon"].notna()
    )

    # diferencia respecto de la oferta adjudicada más cara
    df.loc[mask_real, "USD/MWh a disminuir"] = (
        pmon_ref - df.loc[mask_real, "PMon"]
    ).clip(lower=0.0)

    # porcentaje respecto del PMon propio
    df.loc[mask_real, "% mínimo a disminuir"] = (
        df.loc[mask_real, "USD/MWh a disminuir"]
        / df.loc[mask_real, "PMon"].replace(0.0, np.nan)
    ).fillna(0.0) * 100.0

    # virtual queda vacío
    mask_virtual = df["Tipo de Contrato"].astype(str).str.strip().str.upper().eq("VIRTUAL")
    df.loc[mask_virtual, ["USD/MWh a disminuir", "% mínimo a disminuir"]] = np.nan

    df[["USD/MWh a disminuir", "% mínimo a disminuir"]] = (
        df[["USD/MWh a disminuir", "% mínimo a disminuir"]].round(2)
    )

    return df

df_adj_csv = _apply_gap_columns(df_adj_csv, pmon_frontera)
df_no_csv  = _apply_gap_columns(df_no_csv, pmon_frontera)

# --- guardar archivos ajustados y sobrescribir los originales ---
df_adj_csv.to_csv(PATH_ADJ, index=False, encoding="utf-8-sig")
df_no_csv.to_csv(PATH_NO, index=False, encoding="utf-8-sig")

#df_adj_csv.to_csv(PATH_ADJ_AJUST, index=False, encoding="utf-8-sig")
#df_no_csv.to_csv(PATH_NO_AJUST, index=False, encoding="utf-8-sig")

# --- archivo de chequeo ---
df_check = pd.concat(
    [
        df_adj_csv.assign(archivo_origen="adjudicados"),
        df_no_csv.assign(archivo_origen="no_adjudicados"),
    ],
    ignore_index=True
)

df_check["PMon_frontera"] = pmon_frontera
df_check = df_check[
    [
        "archivo_origen",
        "COD",
        "Oferente",
        "Tipo de Contrato",
        "PMon",
        "PMon_frontera",
        "USD/MWh a disminuir",
        "% mínimo a disminuir",
    ]
]
df_check.to_csv(PATH_CHECK, index=False, encoding="utf-8-sig")

print("\n==============================")
print("AJUSTE FINAL SOBRE CSV GENERADOS")
print("==============================")
print(f"PMon frontera usado = {pmon_frontera:.2f}")
print(f"Archivo actualizado: {PATH_ADJ}")
print(f"Archivo actualizado: {PATH_NO}")
print(f"Archivo copia:       {PATH_ADJ_AJUST}")
print(f"Archivo copia:       {PATH_NO_AJUST}")
print(f"Archivo check:       {PATH_CHECK}")

print("\n--- Adjudicados ajustados ---")
print(
    df_adj_csv[
        ["COD", "Oferente", "Tipo de Contrato", "PMon", "USD/MWh a disminuir", "% mínimo a disminuir"]
    ].to_string(index=False)
)

print("\n--- No adjudicados ajustados ---")
print(
    df_no_csv[
        ["COD", "Oferente", "Tipo de Contrato", "PMon", "USD/MWh a disminuir", "% mínimo a disminuir"]
    ].to_string(index=False)
)





# %%
# ============================================================
# GUARDAR FIGURAS ANUALES CON RONDA EN TÍTULO Y NOMBRE
# ============================================================
out_dir = os.path.join(BASE_PATH, "resultados")
os.makedirs(out_dir, exist_ok=True)

plt.show = lambda *args, **kwargs: None

saved = []
for y in range(2030, 2045):
    plt.close("all")

    plot_window_12m_supply_vs_req_two_stage(
        year=y,
        month=5,
        df_req_4320=df_req_4320,
        df_y_stage_cod=df_y,
        df_offer_meta=df_offer_meta,
        A_dict=A_dict,
        g_sol_EG=g_sol_EG,
        v_sol=v_sol,
        current_round=CURRENT_ROUND,
        title_prefix="PEG5",
        tick_step=48
    )

    fig = plt.gcf()
    fpath = os.path.join(out_dir, f"Resultados_Ronda_{CURRENT_ROUND:02d}_05_{y}.png")
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(fpath)

print("\n".join(saved))
print("Fin de la simulación")
print("GME-Global - Damos valor a tu energía")