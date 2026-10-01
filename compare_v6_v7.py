# -*- coding: utf-8 -*-
"""
Автоматическое сравнение MassPropertiesCalculatorV6 и V7.

Файл должен находиться в той же папке, что и:
    mass_properties_calculator(1).py
    mass_properties_calculator_v7_fixed.py
    inter.py
    Fesulyadge_class.py
    Krilo_class.py
    ruli_class.py

Запуск:
    python compare_v6_v7.py

Результат:
    v6_vs_v7_report.txt

Важно:
- оба расчёта выполняются на ОДНОЙ и той же геометрии;
- входные параметры задаются один раз;
- графики не строятся;
- результаты округляются только при выводе, расчёт ведётся
  с исходной точностью.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

V6_FILE = BASE_DIR / "mass_properties_calculator.py"
V7_FILE = BASE_DIR / "mass_properties_calculator_v7_fixed.py"
REPORT_FILE = BASE_DIR / "v6_vs_v7_report.txt"

# Те же входные данные, что использовались в твоём тесте.
SKT_VOLUME = 0.00345
BC_VOLUME = 0.00345
DU_LENGTH = 2.0
FUEL_MASS = 18.52
FUEL_DENSITY = 1650.0
DX = 0.001

CONFIG_SCHEME = "no_rudders"
NUM_FRONT_CONSOLES = 0
NUM_REAR_CONSOLES = 6


def load_module(module_name: str, filename: Path):
    """Загрузить Python-модуль по имени файла."""
    if not filename.exists():
        raise FileNotFoundError(f"Не найден файл: {filename}")

    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось создать загрузчик для {filename}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_aircraft():
    """
    Создать один объект AircraftAerodynamics.

    Важно: новый объект создаётся отдельно для V6 и V7,
    чтобы изменение x0_rel в одном расчёте не влияло на другой.
    """
    from inter import AircraftAerodynamics
    from Krilo_class import WingGeometry

    wing_geo = WingGeometry()

    return AircraftAerodynamics(
        H=wing_geo.H,
        x0_rel=0.5,
        config_scheme=CONFIG_SCHEME,
        num_front_consoles=NUM_FRONT_CONSOLES,
        num_rear_consoles=NUM_REAR_CONSOLES,
    )


def run_v6(v6):
    """Расчёт V6 без графиков и без сохранения штатного файла."""
    aircraft = make_aircraft()

    calculator = v6.MassPropertiesCalculatorV6(
        aircraft.fuselage_geo,
        config_scheme=CONFIG_SCHEME,
        num_front_consoles=NUM_FRONT_CONSOLES,
        num_rear_consoles=NUM_REAR_CONSOLES,
        skt_volume=SKT_VOLUME,
        bc_volume=BC_VOLUME,
        du_length=DU_LENGTH,
        fuel_mass=FUEL_MASS,
        fuel_density=FUEL_DENSITY,
    )

    # V6 печатает много диагностической информации.
    # Для автоматического отчёта это подавляется.
    with contextlib.redirect_stdout(io.StringIO()):
        result = calculator.calculate_mass_properties_disk_method()

    return aircraft, result


def run_v7(v7):
    """Расчёт V7 без графиков и без сохранения штатного файла."""
    aircraft = make_aircraft()

    calculator = v7.MassPropertiesCalculatorV7(
        aircraft.fuselage_geo,
        config_scheme=CONFIG_SCHEME,
        num_front_consoles=NUM_FRONT_CONSOLES,
        num_rear_consoles=NUM_REAR_CONSOLES,
        skt_volume=SKT_VOLUME,
        bc_volume=BC_VOLUME,
        du_length=DU_LENGTH,
        fuel_mass=FUEL_MASS,
        fuel_density=FUEL_DENSITY,
        dx=DX,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        result = calculator.calculate_mass_properties_disk_method()

    return aircraft, result


def get_mass(result, component):
    data = result.get("component_properties", {}).get(component)
    if data is None:
        data = result.get("components", {}).get(component)
    if data is None:
        return 0.0
    return float(data.get("mass", 0.0))


def get_component_cm(result, component):
    data = result.get("component_properties", {}).get(component)
    if data is None:
        data = result.get("components", {}).get(component)
    if data is None:
        return None

    value = data.get("center_of_mass")
    if value is None:
        value = data.get("center_of_mass_global")

    return None if value is None else float(value)


def get_console_total_mass(result):
    total = 0.0
    for side in ("front", "rear"):
        data = result.get("console_properties", {}).get(side, {})
        total += float(data.get("total_mass", 0.0))
    return total


def get_metric(result, metric):
    if metric == "mass":
        return float(result["total_mass"])

    if metric == "cm":
        return float(result["overall_center_of_mass"])

    if metric == "Ixx":
        return float(result["moments_of_inertia"]["Ixx"])

    if metric == "Iyy":
        return float(result["moments_of_inertia"]["Iyy"])

    if metric == "Izz":
        return float(result["moments_of_inertia"]["Izz"])

    if metric == "volume":
        return float(result.get("total_volume", float("nan")))

    if metric == "elements":
        details = result.get("disk_method_details", {})
        return float(details.get("elements_count", float("nan")))

    raise KeyError(metric)


def compare_values(v6, v7):
    """Вернуть абсолютную и относительную разницу V7 относительно V6."""
    absolute = v7 - v6

    if abs(v6) > 1e-15:
        relative_percent = absolute / abs(v6) * 100.0
    else:
        relative_percent = float("nan")

    return absolute, relative_percent


def fmt(value, digits=6):
    if value is None:
        return "нет данных"
    if isinstance(value, float) and math.isnan(value):
        return "нет данных"
    return f"{value:.{digits}f}"


def add_metric(lines, name, unit, v6, v7):
    delta, percent = compare_values(v6, v7)
    lines.append(
        f"{name:<28} "
        f"{fmt(v6, 9):>16} "
        f"{fmt(v7, 9):>16} "
        f"{fmt(delta, 9):>16} "
        f"{fmt(percent, 4):>12} %"
        f"  {unit}"
    )


def build_report(v6, v7):
    aircraft_v6, result_v6 = run_v6(v6)
    aircraft_v7, result_v7 = run_v7(v7)

    lines = []

    lines.append("=" * 110)
    lines.append("АВТОМАТИЧЕСКИЙ ОТЧЁТ СРАВНЕНИЯ V6 vs V7")
    lines.append("=" * 110)
    lines.append("")
    lines.append("Входные данные:")
    lines.append(f"  Схема:                 {CONFIG_SCHEME}")
    lines.append(f"  Передние консоли:      {NUM_FRONT_CONSOLES}")
    lines.append(f"  Задние консоли:        {NUM_REAR_CONSOLES}")
    lines.append(f"  Объём СКТ:              {SKT_VOLUME:.9f} м³")
    lines.append(f"  Объём БЧ:               {BC_VOLUME:.9f} м³")
    lines.append(f"  Длина ДУ:               {DU_LENGTH:.9f} м")
    lines.append(f"  Масса топлива:          {FUEL_MASS:.9f} кг")
    lines.append(f"  Плотность топлива:      {FUEL_DENSITY:.3f} кг/м³")
    lines.append(f"  Шаг V7:                 {DX:.9f} м")
    lines.append("")
    lines.append(
        f"{'Параметр':<28} {'V6':>16} {'V7':>16} "
        f"{'V7 - V6':>16} {'Δ, %':>12}  Единица"
    )
    lines.append("-" * 110)

    for metric, name, unit in (
        ("mass", "Общая масса", "кг"),
        ("cm", "Центр масс от носа", "м"),
        ("Ixx", "Ixx относительно ЦМ", "кг·м²"),
        ("Iyy", "Iyy относительно ЦМ", "кг·м²"),
        ("Izz", "Izz относительно ЦМ", "кг·м²"),
        ("volume", "Общий объём", "м³"),
        ("elements", "Количество элементов", "шт."),
    ):
        add_metric(
            lines,
            name,
            unit,
            get_metric(result_v6, metric),
            get_metric(result_v7, metric),
        )

    lines.append("")
    lines.append("=" * 110)
    lines.append("СРАВНЕНИЕ МАСС КОМПОНЕНТОВ")
    lines.append("=" * 110)
    lines.append(
        f"{'Компонент':<28} {'V6, кг':>16} {'V7, кг':>16} "
        f"{'V7 - V6, кг':>16} {'Δ, %':>12}"
    )
    lines.append("-" * 95)

    components = []
    for source in (
        result_v6.get("component_properties", {}),
        result_v7.get("component_properties", {}),
    ):
        for name in source:
            if name not in components:
                components.append(name)

    if "консоли" not in components:
        components.append("консоли")

    for component in components:
        if component == "консоли":
            m6 = get_console_total_mass(result_v6)
            m7 = get_console_total_mass(result_v7)
        else:
            m6 = get_mass(result_v6, component)
            m7 = get_mass(result_v7, component)

        delta, percent = compare_values(m6, m7)

        lines.append(
            f"{component:<28} {m6:>16.9f} {m7:>16.9f} "
            f"{delta:>16.9f} {percent:>12.4f}"
        )

    lines.append("")
    lines.append("=" * 110)
    lines.append("СРАВНЕНИЕ ЦЕНТРОВ МАСС КОМПОНЕНТОВ")
    lines.append("=" * 110)
    lines.append(
        f"{'Компонент':<28} {'V6, м':>16} {'V7, м':>16} "
        f"{'V7 - V6, м':>16}"
    )
    lines.append("-" * 80)

    for component in components:
        if component == "консоли":
            # Для консолей общий ЦМ считаем через статический момент,
            # если обе версии его предоставляют.
            def console_cm(result):
                total_m = get_console_total_mass(result)
                if total_m <= 0:
                    return None
                moment = 0.0
                for side in ("front", "rear"):
                    c = result.get("console_properties", {}).get(side, {})
                    moment += float(c.get("static_moment", 0.0))
                return moment / total_m if total_m else None

            c6 = console_cm(result_v6)
            c7 = console_cm(result_v7)
        else:
            c6 = get_component_cm(result_v6, component)
            c7 = get_component_cm(result_v7, component)

        if c6 is None or c7 is None:
            lines.append(
                f"{component:<28} {'нет данных':>16} {'нет данных':>16} "
                f"{'нет данных':>16}"
            )
        else:
            lines.append(
                f"{component:<28} {c6:>16.9f} {c7:>16.9f} "
                f"{c7-c6:>16.9f}"
            )

    lines.append("")
    lines.append("=" * 110)
    lines.append("СРАВНЕНИЕ ГРАНИЦ ОТСЕКОВ")
    lines.append("=" * 110)
    lines.append(
        f"{'Отсек':<12} {'V6 начало':>14} {'V7 начало':>14} "
        f"{'Δ начало':>14} {'V6 конец':>14} {'V7 конец':>14} {'Δ конец':>14}"
    )
    lines.append("-" * 100)

    boundary_names = []
    for source in (
        result_v6.get("boundaries", {}),
        result_v7.get("boundaries", {}),
    ):
        for name in source:
            if name not in boundary_names:
                boundary_names.append(name)

    for name in boundary_names:
        b6 = result_v6.get("boundaries", {}).get(name)
        b7 = result_v7.get("boundaries", {}).get(name)

        if b6 is None or b7 is None:
            lines.append(f"{name:<12} нет данных")
            continue

        s6, s7 = float(b6["start"]), float(b7["start"])
        e6, e7 = float(b6["end"]), float(b7["end"])

        lines.append(
            f"{name:<12} {s6:>14.9f} {s7:>14.9f} {s7-s6:>14.9f} "
            f"{e6:>14.9f} {e7:>14.9f} {e7-e6:>14.9f}"
        )

    lines.append("")
    lines.append("=" * 110)
    lines.append("ПРОВЕРКИ V7")
    lines.append("=" * 110)

    checks = result_v7.get("checks", {})
    if checks:
        for name, value in checks.items():
            lines.append(f"  {name}: {value}")
    else:
        lines.append("  V7 не вернул словарь checks.")

    lines.append("")
    lines.append("=" * 110)
    lines.append("АВТОМАТИЧЕСКИЕ ВЫВОДЫ")
    lines.append("=" * 110)

    total_mass_v6 = get_metric(result_v6, "mass")
    total_mass_v7 = get_metric(result_v7, "mass")
    cm_v6 = get_metric(result_v6, "cm")
    cm_v7 = get_metric(result_v7, "cm")

    mass_delta, mass_percent = compare_values(total_mass_v6, total_mass_v7)
    cm_delta = cm_v7 - cm_v6

    lines.append(
        f"  Разница общей массы V7 относительно V6: "
        f"{mass_delta:.9f} кг ({mass_percent:.6f} %)"
    )
    lines.append(
        f"  Смещение ЦМ при переходе V6 -> V7: "
        f"{cm_delta:.9f} м"
    )

    if abs(mass_percent) < 0.1:
        lines.append(
            "  Общая масса двух реализаций отличается менее чем на 0.1%."
        )
    else:
        lines.append(
            "  Общая масса изменилась более чем на 0.1%; "
            "следует проверить компоненты и границы отсеков."
        )

    # Важная проверка: одинаковая геометрия корпуса.
    l6 = getattr(result_v6, "total_length", None)
    l6 = result_v6.get("total_length")
    l7 = result_v7.get("total_length")

    if l6 is not None and l7 is not None:
        lines.append(
            f"  Длина корпуса V6/V7: {l6:.9f} / {l7:.9f} м; "
            f"разница = {l7-l6:.3e} м"
        )

    lines.append("")
    lines.append(
        "Примечание: данный отчёт сравнивает фактически возвращённые "
        "результаты двух реализаций на одинаковых входных данных. "
        "Он не делает вывода о физической корректности модели; "
        "он показывает, где именно V6 и V7 расходятся."
    )
    lines.append("=" * 110)

    return "\n".join(lines), result_v6, result_v7


def main():
    print("=" * 80)
    print("АВТОМАТИЧЕСКОЕ СРАВНЕНИЕ V6 vs V7")
    print("=" * 80)

    print("\nЗагрузка V6...")
    v6 = load_module("mass_properties_v6_for_compare", V6_FILE)

    print("Загрузка V7...")
    v7 = load_module("mass_properties_v7_for_compare", V7_FILE)

    print("Расчёт V6...")
    print("Расчёт V7...")
    report, result_v6, result_v7 = build_report(v6, v7)

    REPORT_FILE.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nОтчёт сохранён: {REPORT_FILE}")

    return result_v6, result_v7


if __name__ == "__main__":
    main()
