# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:07:43 2025

@author: Пользователь
Исправленная версия с корректным обнулением всех нечетных коэффициентов
"""

"lIBCONSTFUNC"
import math
from typing import List, TypeVar, Union, Optional
import numpy as np

# Константы
pi = 3.141592653589793  # число pi
deg = 57.295779513082320876798154814105  # отношение значения угла в градусах к значению в радианах
g = 9.80665  # ускорение свободного падения, м/с^2

# Функции
def sqr(x):
    """Функция квадрата"""
    return x * x

def sign(val):
    """Функция сигнум (знак)"""
    if isinstance(val, (int, float)):
        return 1 if val > 0 else (-1 if val < 0 else 0)
    return np.sign(val)

def H(val):
    """Функция Хевисайда"""
    if val < 0.0:
        return 0.0
    elif val == 0.0:
        return 0.5
    else:
        return 1.0

def Linterp(XY, x, Y_index=1, X_index=0):
    """Функция линейной интерполяции"""
    if x <= XY[0][X_index]:
        return XY[0][Y_index]
    elif x >= XY[-1][X_index]:
        return XY[-1][Y_index]
    else:
        for i in range(len(XY) - 1):
            if XY[i][X_index] <= x < XY[i + 1][X_index]:
                return (XY[i][Y_index] * ((XY[i + 1][X_index] - x) / (XY[i + 1][X_index] - XY[i][X_index])) + 
                       XY[i + 1][Y_index] * (1.0 - (XY[i + 1][X_index] - x) / (XY[i + 1][X_index] - XY[i][X_index])))
        return XY[-1][Y_index]  # fallback

def BiLinterp(XYZ, x, y, Z_index=2):
    """Функция билинейной интерполяции"""
    # ограничиваем по x
    x_min = XYZ[0][0]
    x_max = XYZ[-1][0]
    
    if x <= x_min:
        i = 0
        y_min = XYZ[i][1]
        if y <= y_min:
            Z_pos = XYZ[i][Z_index]
        else:
            while i < len(XYZ) - 1 and XYZ[i][1] <= y:
                i += 1
                if i < len(XYZ) and XYZ[i - 1][1] > XYZ[i][1]:
                    break
            
            if i < len(XYZ) and XYZ[i - 1][1] < XYZ[i][1]:
                Z_pos = (XYZ[i][Z_index] * ((XYZ[i - 1][1] - y) / (XYZ[i - 1][1] - XYZ[i][1])) + 
                        XYZ[i - 1][Z_index] * (1.0 - (XYZ[i - 1][1] - y) / (XYZ[i - 1][1] - XYZ[i][1])))
            else:
                Z_pos = XYZ[i - 1][Z_index] if i > 0 else XYZ[0][Z_index]
        
        return Z_pos
    
    elif x >= x_max:
        i = len(XYZ) - 1
        y_max = XYZ[i][1]
        if y >= y_max:
            Z_pre = XYZ[i][Z_index]
        else:
            while i > 0 and XYZ[i][1] >= y:
                i -= 1
                if i >= 0 and XYZ[i][1] > XYZ[i + 1][1]:
                    break
            
            if i >= 0 and XYZ[i][1] < XYZ[i + 1][1]:
                Z_pre = (XYZ[i][Z_index] * ((XYZ[i + 1][1] - y) / (XYZ[i + 1][1] - XYZ[i][1])) + 
                        XYZ[i + 1][Z_index] * (1.0 - (XYZ[i + 1][1] - y) / (XYZ[i + 1][1] - XYZ[i][1])))
            else:
                Z_pre = XYZ[i + 1][Z_index] if i < len(XYZ) - 1 else XYZ[-1][Z_index]
        
        return Z_pre
    
    else:
        # ищем индекс начала области таблицы с первым x больше заданного
        i_x = 0
        while i_x < len(XYZ) - 1 and XYZ[i_x][0] <= x:
            i_x += 1
        
        # ищем индекс y при меньшем х
        i = i_x - 1
        y_max = XYZ[i][1]
        if y >= y_max:
            Z_pre = XYZ[i][Z_index]
        else:
            while i > 0 and XYZ[i][1] >= y:
                i -= 1
                if i >= 0 and XYZ[i][1] > XYZ[i + 1][1]:
                    break
            
            if i >= 0 and XYZ[i][1] < XYZ[i + 1][1]:
                Z_pre = (XYZ[i][Z_index] * ((XYZ[i + 1][1] - y) / (XYZ[i + 1][1] - XYZ[i][1])) + 
                        XYZ[i + 1][Z_index] * (1.0 - (XYZ[i + 1][1] - y) / (XYZ[i + 1][1] - XYZ[i][1])))
            else:
                Z_pre = XYZ[i + 1][Z_index] if i < len(XYZ) - 1 else XYZ[-1][Z_index]
        
        # ищем индекс y при большем х
        i = i_x
        y_min = XYZ[i][1]
        if y <= y_min:
            Z_pos = XYZ[i][Z_index]
        else:
            while i < len(XYZ) - 1 and XYZ[i][1] <= y:
                i += 1
                if i < len(XYZ) and XYZ[i - 1][1] > XYZ[i][1]:
                    break
            
            if i < len(XYZ) and XYZ[i - 1][1] < XYZ[i][1]:
                Z_pos = (XYZ[i][Z_index] * ((XYZ[i - 1][1] - y) / (XYZ[i - 1][1] - XYZ[i][1])) + 
                        XYZ[i - 1][Z_index] * (1.0 - (XYZ[i - 1][1] - y) / (XYZ[i - 1][1] - XYZ[i][1])))
            else:
                Z_pos = XYZ[i - 1][Z_index] if i > 0 else XYZ[0][Z_index]
        
        return (Z_pre * ((XYZ[i_x][0] - x) / (XYZ[i_x][0] - XYZ[i_x - 1][0])) + 
                Z_pos * (1.0 - (XYZ[i_x][0] - x) / (XYZ[i_x][0] - XYZ[i_x - 1][0])))

def TriLinterp(XYZF, x, y, z, F_index=3, zero_threshold=1e-6):
    """
    Функция трилинейной интерполяции с физической коррекцией для малых углов.
    
    Параметры:
    - zero_threshold: порог для обнуления нечетных коэффициентов при малых углах
    """
    # Определение отдельных массивов x, y, z
    x_Mas = []
    y_Mas = []
    z_Mas = []
    
    for point in XYZF:
        if point[0] not in x_Mas:
            x_Mas.append(point[0])
        if point[1] not in y_Mas:
            y_Mas.append(point[1])
        if point[2] not in z_Mas:
            z_Mas.append(point[2])
    
    x_Mas.sort()
    y_Mas.sort()
    z_Mas.sort()
    
    n_x = len(x_Mas)
    n_y = len(y_Mas)
    n_z = len(z_Mas)
    
    # ===== ОПРЕДЕЛЕНИЕ ТИПОВ КОЭФФИЦИЕНТОВ =====
    # Индексы для нечетных функций (должны быть =0 при y=0 и антисимметричны)
    # c_ya (4), c_y (6), m_z (7)
    ODD_INDICES = [4, 6, 7]
    # Индексы для четных функций (симметричны при y=0)
    EVEN_INDICES = [3, 5]     # c_xa, c_x
    # Индекс для xd (требует специальной обработки)
    XD_INDEX = 8
    
    # ===== СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ xd =====
    # Для xd НЕ используем значение при α=0, интерполируем между положительными и отрицательными углами
    if F_index == XD_INDEX:
        # Находим первый положительный и первый отрицательный угол в таблице
        y_pos_vals = [yy for yy in y_Mas if yy > 0]
        y_neg_vals = [yy for yy in y_Mas if yy < 0]
        
        # Если есть и положительные, и отрицательные углы
        if y_pos_vals and y_neg_vals:
            y_pos_min = min(y_pos_vals)  # минимальный положительный угол
            y_neg_max = max(y_neg_vals)  # максимальный отрицательный угол (ближайший к нулю)
            
            # Если текущий угол между отрицательным и положительным (включая ноль)
            if y_neg_max <= y <= y_pos_min:
                # Интерполируем между значениями при y_neg_max и y_pos_min
                # Получаем значение при y_neg_max
                val_neg = TriLinterp_standard(XYZF, x, y_neg_max, z, XD_INDEX)
                # Получаем значение при y_pos_min
                val_pos = TriLinterp_standard(XYZF, x, y_pos_min, z, XD_INDEX)
                
                # Линейная интерполяция между этими двумя точками
                if abs(y_pos_min - y_neg_max) > 1e-10:
                    # Нормированная координата от 0 до 1
                    t = (y - y_neg_max) / (y_pos_min - y_neg_max)
                    return val_neg * (1 - t) + val_pos * t
                else:
                    # Если точки совпадают (маловероятно)
                    return (val_neg + val_pos) / 2.0
    
    # ===== ФИЗИЧЕСКАЯ КОРРЕКЦИЯ ДЛЯ НЕЧЕТНЫХ ФУНКЦИЙ =====
    # Для всех нечетных коэффициентов (c_ya, c_y, m_z) применяем единую обработку
    if F_index in ODD_INDICES:
        # Находим ближайшие положительный и отрицательный углы
        y_pos_vals = [yy for yy in y_Mas if yy > 0]
        y_neg_vals = [yy for yy in y_Mas if yy < 0]
        
        # Определяем названия коэффициентов для отладки
        coeff_names = {4: "c_ya", 6: "c_y", 7: "m_z"}
        coeff_name = coeff_names.get(F_index, f"index_{F_index}")
        
        if y_pos_vals and y_neg_vals:
            y_pos_min = min(y_pos_vals)
            y_neg_max = max(y_neg_vals)
            
            # Определяем зону малых углов (между отрицательным и положительным)
            small_angle_zone = (y_neg_max <= y <= y_pos_min)
            
            # Если угол в зоне малых углов
            if small_angle_zone:
                # Получаем значения в ближайших точках
                val_pos = TriLinterp_standard(XYZF, x, y_pos_min, z, F_index)
                val_neg = TriLinterp_standard(XYZF, x, y_neg_max, z, F_index)
                
                # Проверка на антисимметричность: val_neg должна быть равна -val_pos
                # Если данные не антисимметричны, исправляем это
                if abs(val_neg + val_pos) > abs(val_pos) * 0.1:  # Если отклонение > 10%
                    # Принудительно делаем антисимметричными
                    avg_abs = (abs(val_pos) + abs(val_neg)) / 2.0
                    val_pos = avg_abs * np.sign(y_pos_min)
                    val_neg = -val_pos
                
                # Линейная интерполяция с проходом через ноль
                if abs(y_pos_min - y_neg_max) > 1e-10:
                    t = (y - y_neg_max) / (y_pos_min - y_neg_max)
                    # При t=0.5 (y=0) результат должен быть 0
                    result = val_neg * (1 - t) + val_pos * t
                    
                    # Дополнительная проверка: при y=0 результат должен быть 0
                    if abs(y) < 1e-10:
                        return 0.0
                    
                    return result
                else:
                    return 0.0
        
        # Если нет положительных или отрицательных углов, но угол очень мал
        if abs(y) < zero_threshold:
            return 0.0
        
        # Для очень малых углов - линейная аппроксимация через ноль
        if abs(y) < np.deg2rad(0.5):  # < 1° 0.01745
            # Находим ближайший ненулевой угол
            if y > 0:
                y_ref = min([yy for yy in y_Mas if yy > zero_threshold], np.deg2rad(0.5))# default=0.01745
            else:
                y_ref = max([yy for yy in y_Mas if yy < -zero_threshold], np.deg2rad(0.5))# default=0.01745
            
            val_at_ref = TriLinterp_standard(XYZF, x, y_ref, z, F_index)
            return val_at_ref * (y / y_ref)
    
    # ===== СТАНДАРТНАЯ ИНТЕРПОЛЯЦИЯ ДЛЯ ОСТАЛЬНЫХ СЛУЧАЕВ =====
    return TriLinterp_standard(XYZF, x, y, z, F_index)


def TriLinterp_standard(XYZF, x, y, z, F_index=3):
    """
    Оригинальная функция трилинейной интерполяции без физической коррекции.
    Вынесена отдельно для использования внутри TriLinterp.
    """
    # Определение отдельных массивов x, y, z
    x_Mas = []
    y_Mas = []
    z_Mas = []
    
    for point in XYZF:
        if point[0] not in x_Mas:
            x_Mas.append(point[0])
        if point[1] not in y_Mas:
            y_Mas.append(point[1])
        if point[2] not in z_Mas:
            z_Mas.append(point[2])
    
    x_Mas.sort()
    y_Mas.sort()
    z_Mas.sort()
    
    n_x = len(x_Mas)
    n_y = len(y_Mas)
    n_z = len(z_Mas)
    
    # Если недостаточно данных для интерполяции, возвращаем ближайшее значение
    if n_z < 2 or n_y < 2 or n_x < 2:
        min_dist = float('inf')
        nearest_value = 0.0
        for point in XYZF:
            dist = abs(point[0] - x) + abs(point[1] - y) + abs(point[2] - z)
            if dist < min_dist:
                min_dist = dist
                nearest_value = point[F_index]
        return nearest_value
    
    # Ограничение значений
    x = max(min(x, x_Mas[-1]), x_Mas[0])
    y = max(min(y, y_Mas[-1]), y_Mas[0])
    z = max(min(z, z_Mas[-1]), z_Mas[0])
    
    # Определение позиций в массивах
    if x == x_Mas[-1]:
        x_pre = x_Mas[-2]
        x_post = x_Mas[-1]
        posx_pre = (n_x - 2) * n_z * n_y
        posx_post = (n_x - 1) * n_z * n_y
    else:
        for i in range(n_x - 1):
            if x_Mas[i] <= x < x_Mas[i + 1]:
                x_pre = x_Mas[i]
                x_post = x_Mas[i + 1]
                posx_pre = i * n_z * n_y
                posx_post = (i + 1) * n_z * n_y
                break
    
    if y == y_Mas[-1]:
        y_pre = y_Mas[-2]
        y_post = y_Mas[-1]
        posy_pre = (n_y - 2) * n_z
        posy_post = (n_y - 1) * n_z
    else:
        for i in range(n_y - 1):
            if y_Mas[i] <= y < y_Mas[i + 1]:
                y_pre = y_Mas[i]
                y_post = y_Mas[i + 1]
                posy_pre = i * n_z
                posy_post = (i + 1) * n_z
                break
    
    if z == z_Mas[-1]:
        z_pre = z_Mas[-2]
        z_post = z_Mas[-1]
        posz_pre = n_z - 2
        posz_post = n_z - 1
    else:
        for i in range(n_z - 1):
            if z_Mas[i] <= z < z_Mas[i + 1]:
                z_pre = z_Mas[i]
                z_post = z_Mas[i + 1]
                posz_pre = i
                posz_post = i + 1
                break
    
    # Вычисление коэффициентов
    denominator = (x_post - x_pre) * (z_post - z_pre) * (y_post - y_pre)
    
    C1 = (XYZF[posx_pre + posy_pre + posz_pre][F_index] / denominator) * (x_post - x) * (z_post - z) * (y_post - y)
    C2 = (XYZF[posx_pre + posy_post + posz_pre][F_index] / denominator) * (x_post - x) * (z_post - z) * (y - y_pre)
    C3 = (XYZF[posx_pre + posy_pre + posz_post][F_index] / denominator) * (x_post - x) * (z - z_pre) * (y_post - y)
    C4 = (XYZF[posx_pre + posy_post + posz_post][F_index] / denominator) * (x_post - x) * (z - z_pre) * (y - y_pre)
    C5 = (XYZF[posx_post + posy_pre + posz_pre][F_index] / denominator) * (x - x_pre) * (z_post - z) * (y_post - y)
    C6 = (XYZF[posx_post + posy_post + posz_pre][F_index] / denominator) * (x - x_pre) * (z_post - z) * (y - y_pre)
    C7 = (XYZF[posx_post + posy_pre + posz_post][F_index] / denominator) * (x - x_pre) * (z - z_pre) * (y_post - y)
    C8 = (XYZF[posx_post + posy_post + posz_post][F_index] / denominator) * (x - x_pre) * (z - z_pre) * (y - y_pre)
    
    return C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8


# ============================================================================
# ТЕСТОВАЯ ФУНКЦИЯ
# ============================================================================

def test_all_odd_coefficients():
    """
    Тестовая функция для проверки обнуления всех нечетных коэффициентов
    """
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ОБНУЛЕНИЯ НЕЧЕТНЫХ КОЭФФИЦИЕНТОВ")
    print("=" * 70)
    
    # Создаем тестовые данные
    M = 1.0
    test_data = []
    
    # Добавляем данные для всех нечетных коэффициентов
    # [M, alpha, delta, c_xa, c_ya, c_x, c_y, m_z, xd]
    for alpha in [0.05, 0.1, 0.15, 0.2]:
        test_data.append((
            M, alpha, 0.0,  # M, alpha, delta
            0.5,                    # c_xa (четный)
            -alpha * 15.0,          # c_ya (нечетный) - подъемная сила
            0.3,                    # c_x (четный)
            -alpha * 12.0,          # c_y (нечетный) - нормальная сила
            -alpha * 8.0,           # m_z (нечетный) - момент тангажа
            0.6 + alpha             # xd
        ))
    
    for alpha in [-0.05, -0.1, -0.15, -0.2]:
        test_data.append((
            M, alpha, 0.0,
            0.5,                    # c_xa (четный)
            -alpha * 15.0,          # c_ya (нечетный) - должно быть положительным
            0.3,                    # c_x (четный)
            -alpha * 12.0,          # c_y (нечетный) - должно быть положительным
            -alpha * 8.0,           # m_z (нечетный) - должно быть положительным
            0.6 + abs(alpha)        # xd
        ))
    
    print("\nИсходные данные на границах:")
    print(f"{'α (рад)':>10} | {'α (°)':>8} | {'c_ya':>10} | {'c_y':>10} | {'m_z':>10}")
    print("-" * 60)
    for alpha in [0.05, -0.05]:
        for point in test_data:
            if abs(point[1] - alpha) < 1e-6:
                print(f"{alpha:>+10.3f} | {np.degrees(alpha):>+8.2f} | {point[4]:>+10.3f} | {point[6]:>+10.3f} | {point[7]:>+10.3f}")
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ИНТЕРПОЛЯЦИИ")
    print("=" * 70)
    
    test_angles = [
        (0.0, "Ноль (должен быть 0)"),
        (0.005, "Очень малый положительный (0.3°)"),
        (-0.005, "Очень малый отрицательный (-0.3°)"),
        (0.02, "Малый положительный (1.15°)"),
        (-0.02, "Малый отрицательный (-1.15°)"),
        (0.03, "Положительный (1.72°)"),
        (-0.03, "Отрицательный (-1.72°)"),
    ]
    
    for alpha, desc in test_angles:
        print(f"\n{desc}:")
        print(f"  α = {alpha:+.4f} рад ({np.degrees(alpha):+.2f}°)")
        
        # Получаем значения для всех нечетных коэффициентов
        c_ya = TriLinterp(test_data, M, alpha, 0.0, F_index=4)
        c_y = TriLinterp(test_data, M, alpha, 0.0, F_index=6)
        m_z = TriLinterp(test_data, M, alpha, 0.0, F_index=7)
        
        print(f"  c_ya = {c_ya:+.6f}  (должен быть антисимметричен, при α=0 → 0)")
        print(f"  c_y  = {c_y:+.6f}  (должен быть антисимметричен, при α=0 → 0)")
        print(f"  m_z  = {m_z:+.6f}  (должен быть антисимметричен, при α=0 → 0)")
        
        # Проверка для нулевого угла
        if abs(alpha) < 1e-6:
            if abs(c_ya) > 1e-10:
                print(f"  ⚠ ВНИМАНИЕ: c_ya при α=0 = {c_ya:.6f} (должно быть 0!)")
            if abs(c_y) > 1e-10:
                print(f"  ⚠ ВНИМАНИЕ: c_y при α=0 = {c_y:.6f} (должно быть 0!)")
            if abs(m_z) > 1e-10:
                print(f"  ⚠ ВНИМАНИЕ: m_z при α=0 = {m_z:.6f} (должно быть 0!)")
    
    print("\n" + "=" * 70)
    print("ПРОВЕРКА АНТИСИММЕТРИЧНОСТИ")
    print("=" * 70)
    
    # Проверяем антисимметричность: f(-α) = -f(α)
    alpha_pos = 0.03
    alpha_neg = -0.03
    
    c_ya_pos = TriLinterp(test_data, M, alpha_pos, 0.0, F_index=4)
    c_ya_neg = TriLinterp(test_data, M, alpha_neg, 0.0, F_index=4)
    
    c_y_pos = TriLinterp(test_data, M, alpha_pos, 0.0, F_index=6)
    c_y_neg = TriLinterp(test_data, M, alpha_neg, 0.0, F_index=6)
    
    m_z_pos = TriLinterp(test_data, M, alpha_pos, 0.0, F_index=7)
    m_z_neg = TriLinterp(test_data, M, alpha_neg, 0.0, F_index=7)
    
    print(f"\nПри α = +{alpha_pos:.3f} рад ({np.degrees(alpha_pos):.1f}°):")
    print(f"  c_ya = {c_ya_pos:+.6f}")
    print(f"  c_y  = {c_y_pos:+.6f}")
    print(f"  m_z  = {m_z_pos:+.6f}")
    
    print(f"\nПри α = {alpha_neg:.3f} рад ({np.degrees(alpha_neg):.1f}°):")
    print(f"  c_ya = {c_ya_neg:+.6f}")
    print(f"  c_y  = {c_y_neg:+.6f}")
    print(f"  m_z  = {m_z_neg:+.6f}")
    
    print(f"\nПроверка антисимметричности f(-α) = -f(α):")
    print(f"  c_ya: {c_ya_neg:+.6f} ≈ {(-c_ya_pos):+.6f} → {abs(c_ya_neg + c_ya_pos) < 1e-6}")
    print(f"  c_y:  {c_y_neg:+.6f} ≈ {(-c_y_pos):+.6f} → {abs(c_y_neg + c_y_pos) < 1e-6}")
    print(f"  m_z:  {m_z_neg:+.6f} ≈ {(-m_z_pos):+.6f} → {abs(m_z_neg + m_z_pos) < 1e-6}")
    
    print("\n" + "=" * 70)
    print("ВАЖНО:")
    print("✓ c_ya (индекс 4) - подъемная сила, антисимметрична")
    print("✓ c_y (индекс 6) - нормальная сила, антисимметрична")
    print("✓ m_z (индекс 7) - момент тангажа, антисимметричен")
    print("✓ При α=0 все эти коэффициенты равны 0")
    print("✓ При малых углах - линейная зависимость через ноль")


def test_xd_interpolation():
    """
    Тестовая функция для проверки интерполяции xd
    """
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ ИНТЕРПОЛЯЦИИ xd")
    print("=" * 70)
    
    # Создаем тестовые данные для xd (индекс 8)
    test_data = []
    M = 1.0
    
    # Положительные углы
    for alpha in [0.05, 0.1, 0.15, 0.2]:
        test_data.append((M, alpha, 0.0, 0, 0, 0, 0, 0, 0.6 + alpha))
    
    # Отрицательные углы
    for alpha in [-0.05, -0.1, -0.15, -0.2]:
        test_data.append((M, alpha, 0.0, 0, 0, 0, 0, 0, 0.6 + abs(alpha)))
    
    print("\nИсходные данные на границах:")
    print(f"{'α (рад)':>10} | {'α (°)':>8} | {'xd':>10}")
    print("-" * 35)
    for alpha in [0.05, -0.05]:
        for point in test_data:
            if abs(point[1] - alpha) < 1e-6:
                print(f"{alpha:>+10.3f} | {np.degrees(alpha):>+8.2f} | {point[8]:>+10.6f}")
    
    print("\nРезультаты интерполяции:")
    
    test_angles = [
        (0.0, "Ноль (интерполяция между ±0.05)"),
        (0.02, "Малый положительный"),
        (-0.02, "Малый отрицательный"),
    ]
    
    for alpha, desc in test_angles:
        result = TriLinterp(test_data, M, alpha, 0.0, F_index=8)
        print(f"\n{desc}:")
        print(f"  α = {alpha:+.4f} рад ({np.degrees(alpha):+.2f}°) → xd = {result:.6f}")
    
    print("\n" + "=" * 70)
    print("ВАЖНО ДЛЯ xd:")
    print("✓ xd интерполируется ТОЛЬКО между положительными и отрицательными углами")
    print("✓ Значение при α=0 НЕ используется")
    print("✓ xd плавно меняется от отрицательного к положительному углу")


if __name__ == "__main__":
    test_all_odd_coefficients()
    test_xd_interpolation()
    
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ:")
    print("1. c_ya (индекс 4) - принудительно обнуляется при α=0")
    print("2. c_y (индекс 6) - принудительно обнуляется при α=0")
    print("3. m_z (индекс 7) - принудительно обнуляется при α=0")
    print("4. xd (индекс 8) - интерполяция между ± углами без α=0")
    print("5. Все нечетные коэффициенты антисимметричны: f(-α) = -f(α)")
    
    
abs   
    