from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

from crack_estimator import build_feature_table
from data_loader import TRAINING_SPECIMENS, load_all
from pipeline import predict_specimen, train_estimator
from signal_processing import FEATURE_NAMES
from variable_loading import (
    equivalent_stress_ratio,
    fit_paris_law_exponent,
    variable_loading_exponent,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "PHMDC2019_Data"
SOURCE_XLSX = SCRIPT_DIR / "PHM2019_ScoringSpreadsheet_m10_fixed.xlsx"
OUTPUT_XLSX = SCRIPT_DIR / "PHM2019_ScoringSpreadsheet_T1_T8_evaluated.xlsx"

CALC_GROUPS = {
    "T1": ("A", "B", "C"), "T2": ("D", "E", "F"), "T3": ("G", "H", "I"),
    "T4": ("J", "K", "L"), "T5": ("M", "N", "O"), "T6": ("P", "Q", "R"),
    "T7": ("S", "T", "U"), "T8": ("V", "W", "X"),
}
SCORE_GROUPS = {
    "T1": ("D", "E"), "T2": ("F", "G"), "T3": ("H", "I"),
    "T4": ("J", "K"), "T5": ("L", "M"), "T6": ("N", "O"),
    "T7": ("P", "Q"), "T8": ("R", "S"),
}

T7_LAM_TRANSLOCATE = 30000.0
T8_LAM_TRANSLOCATE = 2000.0


def evaluate_training_specimens() -> pd.DataFrame:
    trained = train_estimator(DATA_ROOT)
    print(f"RMSE de entrenamiento (modelo del notebook, Seccion 5, C=100 gamma=1.0): "
          f"{trained.train_rmse:.3f} mm")

    specimens = load_all(DATA_ROOT, TRAINING_SPECIMENS)
    table = build_feature_table(specimens)
    X = table[FEATURE_NAMES].to_numpy()
    table["estimated"] = trained.model.predict(X)
    return trained, table


def evaluate_validation_specimens(trained) -> dict[str, pd.DataFrame]:
    results = {}

    t7_specimens = load_all(DATA_ROOT, TRAINING_SPECIMENS + ["T7"])
    result_t7 = predict_specimen(
        trained, t7_specimens["T7"], ["T3", "T4"], t7_specimens,
        lam_translocate=T7_LAM_TRANSLOCATE,
    )
    print("T7 - curvas de referencia:", result_t7["reference_selections"])
    results["T7"] = result_t7["score_table"][["cycle", "estimated", "true"]].rename(
        columns={"true": "crack_length_mm"}
    )

    t8_specimens = load_all(DATA_ROOT, TRAINING_SPECIMENS + ["T8"])
    m_fit, _ = fit_paris_law_exponent(load_all(DATA_ROOT, TRAINING_SPECIMENS))
    ratio = equivalent_stress_ratio(
        t8_specimens["T1"].loading_profile(), t8_specimens["T8"].loading_profile()
    )
    exponent = variable_loading_exponent(m_fit, ratio)
    result_t8 = predict_specimen(
        trained, t8_specimens["T8"], ["T3", "T4"], t8_specimens,
        lam_translocate=T8_LAM_TRANSLOCATE,
        variable_loading_exponent=exponent,
        baseline_override=50000,
    )
    print("T8 - curvas de referencia:", result_t8["reference_selections"])
    print(f"T8 - exponente de transformacion de ciclos (Eq. 24): {exponent:.4f}")
    results["T8"] = result_t8["score_table"][["cycle", "estimated", "true"]].rename(
        columns={"true": "crack_length_mm"}
    )

    return results


def fill_workbook(training_table: pd.DataFrame, validation_tables: dict[str, pd.DataFrame]) -> None:
    wb = openpyxl.load_workbook(SOURCE_XLSX, data_only=False)
    score_ws = wb["Score sheet"]
    calc_ws = wb["Calculations sheet"]

    for ws in (score_ws, calc_ws):
        for rng in list(ws.merged_cells.ranges):
            ws.unmerge_cells(str(rng))

    all_tables = {name: training_table[training_table["specimen"] == name].sort_values("cycle")
                  for name in TRAINING_SPECIMENS}
    all_tables.update({name: df.sort_values("cycle") for name, df in validation_tables.items()})

    for name, sub in all_tables.items():
        ccol, dcol, ncol = CALC_GROUPS[name]
        scol, ecol = SCORE_GROUPS[name]
        n = len(sub)
        last_row_calc = 3 + n

        for r in range(2, 64):
            calc_ws[f"{ccol}{r}"] = None
            calc_ws[f"{dcol}{r}"] = None
            calc_ws[f"{ncol}{r}"] = None
        for r in range(6, 20):
            score_ws[f"{scol}{r}"] = None
            score_ws[f"{ecol}{r}"] = None

        calc_ws[f"{ccol}2"] = name
        calc_ws[f"{ccol}3"] = "Number of cycle"
        calc_ws[f"{dcol}3"] = "crack length (mm)"
        calc_ws[f"{ncol}3"] = "normalized crack length (-)"
        score_ws[f"{scol}6"] = name
        score_ws[f"{scol}7"] = "Number of cycle"
        score_ws[f"{ecol}7"] = "Crack length (mm) [ESTIMADO/PREDICHO por el modelo del notebook]"

        for i, (_, row) in enumerate(sub.iterrows()):
            r_calc = 4 + i
            r_score = 8 + i
            calc_ws[f"{ccol}{r_calc}"] = int(row["cycle"])
            calc_ws[f"{dcol}{r_calc}"] = float(row["crack_length_mm"])
            calc_ws[f"{ncol}{r_calc}"] = f"={dcol}{r_calc}/${dcol}${last_row_calc}"
            score_ws[f"{scol}{r_score}"] = int(row["cycle"])
            score_ws[f"{ecol}{r_score}"] = round(float(row["estimated"]), 6)

        for i in range(n):
            r_calc = 4 + i
            r_pen = 16 + i
            calc_ws[f"{ncol}{r_pen}"] = f"=2+10*{ncol}{r_calc}"

        for i in range(n):
            r_calc = 4 + i
            r_score = 8 + i
            r_pen = 28 + i
            calc_ws[f"{ncol}{r_pen}"] = (
                f"=IF(('Score sheet'!{ecol}{r_score}/'Calculations sheet'!${dcol}${last_row_calc}"
                f"-'Calculations sheet'!{ncol}{r_calc})>=0, "
                f"EXP(ABS('Score sheet'!{ecol}{r_score}/'Calculations sheet'!${dcol}${last_row_calc}"
                f"-'Calculations sheet'!{ncol}{r_calc})/0.5)-1, "
                f"EXP(ABS('Score sheet'!{ecol}{r_score}/'Calculations sheet'!${dcol}${last_row_calc}"
                f"-'Calculations sheet'!{ncol}{r_calc})/0.2)-1)"
            )

        calc_ws[f"{ncol}40"] = 1
        for i in range(1, n):
            r_score = 8 + i
            r_score_prev = 7 + i
            r_pen = 40 + i
            calc_ws[f"{ncol}{r_pen}"] = (
                f"=IF(('Score sheet'!{ecol}{r_score}/'Calculations sheet'!${dcol}${last_row_calc}"
                f"-'Score sheet'!{ecol}{r_score_prev}/'Calculations sheet'!${dcol}${last_row_calc})<0, "
                f"1+10*ABS('Score sheet'!{ecol}{r_score}/'Calculations sheet'!${dcol}${last_row_calc}"
                f"-'Score sheet'!{ecol}{r_score_prev}/'Calculations sheet'!${dcol}${last_row_calc}), 1)"
            )

        for i in range(n):
            r_time = 16 + i
            r_asym = 28 + i
            r_mono = 40 + i
            r_pen = 52 + i
            calc_ws[f"{ncol}{r_pen}"] = f"={ncol}{r_mono}*{ncol}{r_asym}*{ncol}{r_time}"

        last_pen_row = 52 + n - 1
        calc_ws[f"{ncol}63"] = f"=SUM({ncol}52:{ncol}{last_pen_row})"
        score_ws[f"{ecol}19"] = f"='Calculations sheet'!{ncol}63"

    score_ws["A1"] = (
        "Plantilla oficial (m=10 corregido) rellenada con datos REALES de T1-T8 "
        "(PHMDC2019_Data). T1-T6: modelo SVR del notebook (Seccion 5, C=100, "
        "gamma=1.0), evaluacion en-muestra. T7-T8: pipeline completo del notebook "
        "(estimacion SVR + prediccion trans-fitting, Secciones 7.2/8.3)."
    )
    score_ws["D19"] = "PENALTY SCORE (m=10)"

    wb.save(OUTPUT_XLSX)
    print(f"Guardado: {OUTPUT_XLSX}")


def main() -> None:
    trained, training_table = evaluate_training_specimens()
    validation_tables = evaluate_validation_specimens(trained)
    fill_workbook(training_table, validation_tables)


if __name__ == "__main__":
    main()
