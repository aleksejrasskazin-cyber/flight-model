# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:07:43 2025

@author: Пользователь
Оптимизированная версия с кэшированием и бинарным поиском
"""

"lIBCONSTFUNC"
import math
from typing import List, TypeVar, Union, Optional
import numpy as np
import bisect
from functools import lru_cache

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


# ============================================================================
# ОПТИМИЗИРОВАННАЯ ТРИЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ С КЭШИРОВАНИЕМ
# ============================================================================

class _TriLinterpCache:
    """
    Внутренний класс для кэширования предварительно обработанных данных
    """
    def __init__(self, XYZF, F_index):
        self.F_index = F_index
        
        # Извлекаем уникальные координаты
        self.x_vals = sorted(set(point[0] for point in XYZF))
        self.y_vals = sorted(set(point[1] for point in XYZF))
        self.z_vals = sorted(set(point[2] for point in XYZF))
        
        self.nx = len(self.x_vals)
        self.ny = len(self.y_vals)
        self.nz = len(self.z_vals)
        
        # Создаем словарь для быстрого доступа к значениям
        self.value_dict = {}
        for point in XYZF:
            key = (point[0], point[1], point[2])
            self.value_dict[key] = point
        
        # Границы
        self.x_min, self.x_max = self.x_vals[0], self.x_vals[-1]
        self.y_min, self.y_max = self.y_vals[0], self.y_vals[-1]
        self.z_min, self.z_max = self.z_vals[0], self.z_vals[-1]
        
        # Предварительно создаем массив для быстрого доступа (если сетка полная)
        self.use_array = False
        if self.nx * self.ny * self.nz == len(XYZF):
            # Сетка полная - можно использовать numpy массив
            self.use_array = True
            self.value_array = np.zeros((self.nx, self.ny, self.nz))
            for i, x in enumerate(self.x_vals):
                for j, y in enumerate(self.y_vals):
                    for k, z in enumerate(self.z_vals):
                        point = self.value_dict.get((x, y, z))
                        if point:
                            self.value_array[i, j, k] = point[F_index]
    
    def get_value(self, i, j, k):
        """Быстрое получение значения по индексам"""
        if self.use_array:
            return self.value_array[i, j, k]
        
        x = self.x_vals[i]
        y = self.y_vals[j]
        z = self.z_vals[k]
        point = self.value_dict.get((x, y, z))
        if point:
            return point[self.F_index]
        return 0.0
    
    def find_nearest(self, x, y, z):
        """Находит ближайшую точку"""
        # Находим ближайшие индексы
        ix = bisect.bisect_left(self.x_vals, x)
        iy = bisect.bisect_left(self.y_vals, y)
        iz = bisect.bisect_left(self.z_vals, z)
        
        ix = max(0, min(ix, self.nx - 1))
        iy = max(0, min(iy, self.ny - 1))
        iz = max(0, min(iz, self.nz - 1))
        
        return self.get_value(ix, iy, iz)


# Глобальный кэш интерполяторов
_interpolator_cache = {}

def _get_interpolator(XYZF, F_index):
    """Получение интерполятора из кэша"""
    # Используем id данных и индекс как ключ
    # Важно: данные не должны меняться во время работы
    key = (id(XYZF), F_index)
    if key not in _interpolator_cache:
        _interpolator_cache[key] = _TriLinterpCache(XYZF, F_index)
    return _interpolator_cache[key]


def TriLinterp(XYZF, x, y, z, F_index=3, zero_threshold=1e-6):
    """
    Оптимизированная функция трилинейной интерполяции с физической коррекцией для малых углов.
    
    Параметры:
    - XYZF: таблица данных [x, y, z, f, ...]
    - x, y, z: координаты для интерполяции
    - F_index: индекс интерполируемого значения
    - zero_threshold: порог для обнуления нечетных коэффициентов при малых углах
    
    Возвращает интерполированное значение
    """
    # ===== ФИЗИЧЕСКАЯ КОРРЕКЦИЯ: Проверка на малые углы =====
    # Индексы для нечетных функций (должны быть =0 при y=0)
    ODD_INDICES = [4, 6, 7]  # c_ya, c_y, m_z
    # Индексы для четных функций (симметричны при y=0)
    EVEN_INDICES = [3, 5]     # c_xa, c_x
    # Индекс для xd (требует специальной обработки)
    XD_INDEX = 8
    
    # Если угол атаки (y) очень мал, применяем специальную обработку
    if abs(y) < zero_threshold and F_index in ODD_INDICES:
        # Для нечетных коэффициентов при y=0 должно быть 0
        if y == 0.0:
            return 0.0
        else:
            # Для очень малых ненулевых углов - линейная аппроксимация с обнулением в нуле
            # Получаем интерполятор для доступа к данным
            interp = _get_interpolator(XYZF, F_index)
            
            # Находим ближайшее ненулевое значение y из таблицы
            y_pos = min([yy for yy in interp.y_vals if yy > 0], default=0.01745)
            y_neg = max([yy for yy in interp.y_vals if yy < 0], default=-0.01745)
            
            if y > 0:
                y_ref = y_pos
                # Интерполируем значение при y_ref используя стандартную интерполяцию
                val_at_ref = _TriLinterp_standard(XYZF, x, y_ref, z, F_index)
                # Линейная аппроксимация через ноль
                return val_at_ref * (y / y_ref)
            else:
                y_ref = y_neg
                val_at_ref = _TriLinterp_standard(XYZF, x, y_ref, z, F_index)
                return val_at_ref * (y / y_ref)
    
    # ===== СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ xd =====
    if F_index == XD_INDEX and abs(y) < 0.0349:  # < 2°
        # Для xd при малых углах используем плавную аппроксимацию
        # Находим xd при y=0
        xd_zero = _TriLinterp_standard(XYZF, x, 0.0, z, XD_INDEX)
        
        if abs(y) < 0.01745:  # < 1°
            return xd_zero
        else:
            # Находим xd при y=2°
            xd_at_2deg = _TriLinterp_standard(XYZF, x, 0.0349, z, XD_INDEX)
            factor = (abs(y) - 0.01745) / 0.01745
            return xd_zero + factor * (xd_at_2deg - xd_zero)
    
    # ===== СТАНДАРТНАЯ ИНТЕРПОЛЯЦИЯ ДЛЯ ОСТАЛЬНЫХ СЛУЧАЕВ =====
    return _TriLinterp_standard(XYZF, x, y, z, F_index)


def _TriLinterp_standard(XYZF, x, y, z, F_index=3):
    """
    Оригинальная функция трилинейной интерполяции без физической коррекции.
    Оптимизирована с использованием кэширования и бинарного поиска.
    """
    # Получаем интерполятор из кэша
    interp = _get_interpolator(XYZF, F_index)
    
    # Если недостаточно данных для интерполяции, возвращаем ближайшее значение
    if interp.nx < 2 or interp.ny < 2 or interp.nz < 2:
        return interp.find_nearest(x, y, z)
    
    # Ограничение значений
    x = max(interp.x_min, min(x, interp.x_max))
    y = max(interp.y_min, min(y, interp.y_max))
    z = max(interp.z_min, min(z, interp.z_max))
    
    # Бинарный поиск индексов
    # Для x
    ix = bisect.bisect_left(interp.x_vals, x)
    if ix == 0:
        ix1, ix2 = 0, 0
    elif ix == interp.nx:
        ix1, ix2 = interp.nx - 1, interp.nx - 1
    else:
        if interp.x_vals[ix] == x:
            ix1, ix2 = ix, ix
        else:
            ix1, ix2 = ix - 1, ix
    
    # Для y
    iy = bisect.bisect_left(interp.y_vals, y)
    if iy == 0:
        iy1, iy2 = 0, 0
    elif iy == interp.ny:
        iy1, iy2 = interp.ny - 1, interp.ny - 1
    else:
        if interp.y_vals[iy] == y:
            iy1, iy2 = iy, iy
        else:
            iy1, iy2 = iy - 1, iy
    
    # Для z
    iz = bisect.bisect_left(interp.z_vals, z)
    if iz == 0:
        iz1, iz2 = 0, 0
    elif iz == interp.nz:
        iz1, iz2 = interp.nz - 1, interp.nz - 1
    else:
        if interp.z_vals[iz] == z:
            iz1, iz2 = iz, iz
        else:
            iz1, iz2 = iz - 1, iz
    
    # Если все точки совпадают
    if ix1 == ix2 and iy1 == iy2 and iz1 == iz2:
        return interp.get_value(ix1, iy1, iz1)
    
    # Получаем значения в узлах
    f111 = interp.get_value(ix1, iy1, iz1)
    f211 = interp.get_value(ix2, iy1, iz1)
    f121 = interp.get_value(ix1, iy2, iz1)
    f221 = interp.get_value(ix2, iy2, iz1)
    f112 = interp.get_value(ix1, iy1, iz2)
    f212 = interp.get_value(ix2, iy1, iz2)
    f122 = interp.get_value(ix1, iy2, iz2)
    f222 = interp.get_value(ix2, iy2, iz2)
    
    # Вычисление нормированных координат
    if ix1 == ix2:
        x_norm = 0.0
    else:
        x_norm = (x - interp.x_vals[ix1]) / (interp.x_vals[ix2] - interp.x_vals[ix1])
    
    if iy1 == iy2:
        y_norm = 0.0
    else:
        y_norm = (y - interp.y_vals[iy1]) / (interp.y_vals[iy2] - interp.y_vals[iy1])
    
    if iz1 == iz2:
        z_norm = 0.0
    else:
        z_norm = (z - interp.z_vals[iz1]) / (interp.z_vals[iz2] - interp.z_vals[iz1])
    
    # Трилинейная интерполяция
    # Интерполяция по x
    c00 = f111 * (1 - x_norm) + f211 * x_norm
    c01 = f121 * (1 - x_norm) + f221 * x_norm
    c10 = f112 * (1 - x_norm) + f212 * x_norm
    c11 = f122 * (1 - x_norm) + f222 * x_norm
    
    # Интерполяция по y
    c0 = c00 * (1 - y_norm) + c01 * y_norm
    c1 = c10 * (1 - y_norm) + c11 * y_norm
    
    # Интерполяция по z
    return c0 * (1 - z_norm) + c1 * z_norm


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ДЛЯ РАБОТЫ С ИНТЕРПОЛЯТОРАМИ
# ============================================================================

def clear_interpolation_cache():
    """Очистка кэша интерполяторов (полезно при перезагрузке данных)"""
    global _interpolator_cache
    _interpolator_cache.clear()

def get_cache_info():
    """Получение информации о кэше"""
    return {
        'cache_size': len(_interpolator_cache),
        'keys': list(_interpolator_cache.keys())
    }


# Для обратной совместимости - сохраняем старую функцию как TriLinterp_old
# (можно удалить, если не нужно)
TriLinterp_old = None  # Здесь была бы старая версия, но мы её заменяем