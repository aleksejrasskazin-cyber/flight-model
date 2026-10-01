# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:10:18 2025

@author: Пользователь
"""
 


 
 
from enum import Enum
from typing import List, Optional
import math
 
# Определения структур для передачи исходных данных в функцию

class CalcParameters:
    """Параметры расчёта"""
    def __init__(self):
        self.dt: float = 0.0      # шаг по времени, с
        self.r_por: float = 0.0   # расстояние поражения, м

class InitConditions:
    """Начальные условия"""
    def __init__(self):
        self.Theta_c0: float = 0.0    # угол направления вектора скорости цели, рад
        self.v_c0: float = 0.0        # скорость цели, м/с
        self.x_c0: float = 0.0        # координата цели, м
        self.y_c0: float = 0.0        # координата цели, м
        self.Psi_c0: float = 0.0    # начальный курсовой угол цели, рад
        self.Psi_n0: float = 0.0    # начальный курсовой угол носителя, рад
        self.z_c0: float = 0.0       # начальная координата z цели, м
        self.z_n0: float = 0.0       # начальная координата z носителя, м
        self.z_r0: float = 0.0       # начальная координата z ракеты, м
        
        
        self.Theta_n0: float = 0.0    # угол направления вектора скорости носителя, рад {если NaN, то равен Theta_r}
        self.v_n0: float = 0.0        # скорость носителя, м/с
        self.x_pusk: float = 0.0      # координата пуска (начальная координата носителя и ракеты), м
        self.y_pusk: float = 0.0      # координата пуска (начальная координата носителя и ракеты), м
        self.z_pusk: float = 0.0
        
        
        self.Theta_r0: float = 0.0    # угол направления вектора скорости ракеты, рад {если NaN, то определяется методом наведения}
        self.Psi_r0: float = 0.0
        self.v_r0: float = 0.0        # скорость ракеты, м/с
        self.omega_x: float = 0.0
        self.phi: float = 0.0
        self.M_x: float = 0.0
        self.I_x: float = 0.0
        self.M_y: float = 0.0
        self.I_y: float = 0.0
        self.M_z: float = 0.0
        self.I_z: float = 0.0
        self.omega_x0: float = 0.0  # начальная угловая скорость вокруг оси x
        self.phi0: float = 0.0      # начальный угол крена
        self.gamma_r0:float = 0.0
class LimitConditions:
    """Ограничения"""
    def __init__(self):
        self.n_ya_r_max: float = 0.0        # максимальная перегрузка ракеты
        self.t_min: float = 0.0             # минимальное время полёта ракеты, с
        self.t_max: float = 0.0             # максимальное время полёта ракеты, с

        self.R_min: Optional[float] = None  # минимальная дальность полёта ракеты, м {None: не проверяется}
        self.R_max: Optional[float] = None  # максимальная дальность полёта ракеты, м {None: не проверяется}
        self.y_r_min: Optional[float] = None # минимальная высота полёта ракеты, м {None: не проверяется}
        self.y_r_max: Optional[float] = None # максимальная высота полёта ракеты, м {None: не проверяется}
        self.epsilon_nc_min: Optional[float] = None # минимальный угол наклона линии визирования носитель-цель, рад {None: не проверяется}
        self.epsilon_nc_max: Optional[float] = None # максимальный угол наклона линии визирования носитель-цель, рад {None: не проверяется}
        self.v_r_min: Optional[float] = None # минимальная скорость полёта ракеты, м/с {None: не проверяется}
        self.v_r_max: Optional[float] = None # максимальная скорость полёта ракеты, м/с {None: не проверяется}
 
class C_Data:
    """Данные по цели"""
    def __init__(self):
        self.n_xa_c_pot: List[List[float]] = [] # тангенциальная перегрузка цели (таблица время-перегрузка)
        self.n_ya_c_pot: List[List[float]] = [] # нормальная скоростная перегрузка цели (таблица время-перегрузка)
        self.n_za_c_pot: List[List[float]] = [] # боковая перегрузка цели (таблица время-перегрузка)
class N_Data:
    """Данные по носителю"""
    def __init__(self):
        self.n_xa_n_pot: List[List[float]] = [] # тангенциальная перегрузка носителя (таблица время-перегрузка)
        self.n_ya_n_pot: List[List[float]] = [] # нормальная скоростная перегрузка носителя (таблица время-перегрузка)
        self.n_za_n_pot: List[List[float]] = [] # боковая перегрузка носителя (таблица время-перегрузка)
class R_Data:
    """Данные по ракете"""
    def __init__(self):
        self.k_cx = 1.0   # множитель для Cx (анализ чувствительности)
        self.k_cy = 1.0
        self.t_P_G_tabl: List[List[float]] = [] # таблица время-тяга-расход
        self.m_r0: float = 0.0       # стартовая масса ракеты, кг
        self.S_a: float = 0.0        # площадь выходного сечения сопла, м^2
        self.S_m: float = 0.0        # площадь миделя, м^2
        self.AD_tabl: List[List[float]] = []   # таблица АДХ (M - alpha - delta - c_xa - c_ya - c_x - c_y - m_z)
        self.x_O: float = 0.0        # координата центра масс к которой приведены моментные АДХ
        self.L: float = 0.0          # характерный линейный размер
        self.AD_tabl2: List[List[float]] = [] # таблица АДХ(h - M - Delta_c_xa0_h - c_xa_0_dn)
        self.MC_tabl: List[List[float]] = []  # таблица МЦХ(m_r - x_cm - I_z)
        self.AD_tabl3: List[List[float]] = [] # таблица АДХ(x_0 - M - m_z_omega_z - m_z_dot_alpha)

class GuidanceMethod(Enum):
    """Методы наведения (перечисление доступных для выбора методов)"""
    Null = 0                       # нет управления (полёт с нулевой перегрузкой)
    PurePursuit = 1                # метод чистой погони (самонаведение)
    DeviatedPursuit = 2            # метод погони с постоянным углом упреждения (самонаведение)
    ParallelNav = 3                # метод параллельного сближения (самонаведение)
    ProportionalNav_const = 4      # метод пропорционального сближения с постоянным коэфф. (самонаведение)
    ProportionalNav_var = 5        # метод пропорционального сближения с переменным коэфф. (самонаведение)
    BeamRiding_LineOfSight = 6     # метод совмещения (теленаведение)
    BeamRiding_ConstAlignment = 7  # метод "С" (теленаведение)
    BeamRiding_Alignment = 8       # метод спрямления (теленаведение)

class GuidanceMethod_Data:
    """Данные о методе наведения"""
    def __init__(self):
        self.Method: GuidanceMethod = GuidanceMethod.Null # метод наведения
        self.phi_upr: float = 0.0  # угол упреждения, рад (для метода DeviatedPursuit, всегда положительный)
        self.k: float = 0.0        # коэффициент пропорциональности (для метода ProportionalNav_const)
        self.lambda_: float = 0.0  # коэффициент пропорциональности (для метода ProportionalNav_var)
        self.C: float = 0.0        # коэффициент в методе "С" (для метода BeamRiding_ConstAlignment)
        self.m: float = 0.0        # коэффициент в методе спрямления (для метода BeamRiding_Alignment)

class TrajectoryParameters:
    """Результаты расчёта траектории (одна точка по времени)"""
    def __init__(self):
        self.t: float = 0.0       # время, с

        # Цель
        self.Theta_c: float = 0.0 # угол направления вектора скорости цели, рад
        self.v_c: float = 0.0     # скорость цели, м/с
        self.x_c: float = 0.0     # координата x цели, м
        self.y_c: float = 0.0     # координата y цели, м
        self.z_c: float = 0.0 
        self.n_xa_c: float = 0.0  # тангенциальная перегрузка цели, -
        self.n_ya_c: float = 0.0  # нормальная скоростная перегрузка цели, -
        self.Psi_c: float = 0.0      # курсовой угол цели, рад
        self.n_za_c: float = 0.0      # боковая перегрузка цели, 
        # Носитель
        self.Theta_n: float = 0.0 # угол направления вектора скорости носителя, рад
        self.v_n: float = 0.0     # скорость носителя, м/с
        self.x_n: float = 0.0     # координата x носителя, м
        self.y_n: float = 0.0     # координата y носителя, м
        self.z_n: float = 0.0
        self.n_xa_n: float = 0.0  # тангенциальная перегрузка носителя, -
        self.n_ya_n: float = 0.0  # нормальная скоростная перегрузка носителя, -
        self.Psi_n: float = 0.0      # курсовой угол носителя, рад
        self.n_za_n: float = 0.0      # боковая перегрузка носителя, -
        
        # Ракета
        self.Theta_r: float = 0.0 # угол направления вектора скорости ракеты, рад
        self.Psi_r: float = 0.0
        self.n_za_r: float = 0.0      # боковая перегрузка ракеты, -
        self.gamma_a: float = 0.0     # скоростной угол крена, рад
        self.alpha_n: float = 0.0     # пространственный угол атаки, рад
        self.v_r: float = 0.0     # скорость ракеты, м/с
        self.x_r: float = 0.0     # координата x ракеты, м
        self.y_r: float = 0.0     # координата y ракеты, м
        self.z_r: float = 0.0
        self.n_xa_r: float = 0.0  # тангенциальная перегрузка ракеты, -
        self.n_ya_r: float = 0.0  # нормальная скоростная перегрузка ракеты, -
        self.m_r: float = 0.0     # масса ракеты, кг
        self.P: float = 0.0       # тяга, Н
        self.G_t: float = 0.0     # массовый расход топлива, кг/с
        self.M: float = 0.0       # число Маха
        self.alpha: float = 0.0   # угол атаки, рад
        self.delta: float = 0.0
        self.beta: float = 0.0
        self.I_z: float = 0.0
        self.omega_z: float = 0.0
        self.omega_y: float = 0.0
        self.dot_alpha: float = 0.0
        self.theta_r: float = 0.0
        self.M_z: float = 0.0
        self.n_ya_r_pot: float = 0.0
        self.omega_x: float = 0.0  # Угловая скорость вокруг оси x (крен)
        self.phi: float = 0.0      # Угол крена
        self.M_x: float = 0.0      # Момент вокруг оси x
        self.I_x: float = 0.0      # Момент инерции вокруг оси x
        self.M_y: float = 0.0     
        self.I_y: float = 0.0
        
        self.c_xa: float = 0.0
        self.X_a: float = 0.0
        self.Y_a: float = 0.0
        self.Z_a: float = 0.0
        
        self.X: float = 0.0
        self.Y: float = 0.0
        self.Z: float = 0.0 
        
        self.M_z_st: float = 0.0
        self.M_z_omega: float = 0.0
        
        self.beta_rc: float = 0.0      # угол ЛВ в горизонтальной плоскости, рад
        self.dot_beta_rc: float = 0.0  # угловая скорость ЛВ в горизонтальной плоскости, рад/с
        self.n_za_r_pot: float = 0.0   # потребная боковая перегрузка
        
 
        
        self.delta: float = 0.0   # угол поворота рулей, рад
        self.x_cm: float = 0.0    # координата ЦМ
        self.n_ya_rasp_plus: float = 0.0  # положительная располагаемая перегрузка
        self.n_ya_rasp_minus: float = 0.0 # отрицательная располагаемая перегрузка

        # Геометрические параметры
        self.r_rc: float = 0.0    # Расстояние между ракетой и целью, м
        self.dot_r_rc: float = 0.0 # Скорость изменения расстояния между ракетой и целью, м/с
        self.ddot_r_rc: float = 0.0 # Ускорение изменения расстояния между ракетой и целью, м/с^2

        self.epsilon_rc: float = 0.0 # Угол наклона линии ракета-цель, рад
        self.dot_epsilon_rc: float = 0.0 # Угловая скорость линии ракета-цель, рад/с
        self.ddot_epsilon_rc: float = 0.0 # Угловое ускорение линии ракета-цель, рад/с^2

        self.r_nr: float = 0.0    # Расстояние между носителем и ракетой, м
        self.dot_r_nr: float = 0.0 # Скорость изменения расстояния между носителем и ракетой, м/с
        self.ddot_r_nr: float = 0.0 # Ускорение изменения расстояния между носителем и ракетой, м/с^2

        self.epsilon_nr: float = 0.0 # Угол наклона линии носитель-ракета, рад
        self.dot_epsilon_nr: float = 0.0 # Угловая скорость линии носитель-ракета, рад/с
        self.ddot_epsilon_nr: float = 0.0 # Угловое ускорение линии носитель-ракета, рад/с^2

        self.r_nc: float = 0.0    # Расстояние между носителем и целью, м
        self.dot_r_nc: float = 0.0 # Скорость изменения расстояния между носителем и целью, м/с
        self.ddot_r_nc: float = 0.0 # Ускорение изменения расстояния между носителем и целью, м/с^2

        self.epsilon_nc: float = 0.0 # Угол наклона линии носитель-цель, рад
        self.dot_epsilon_nc: float = 0.0 # Угловая скорость линии носитель-цель, рад/с
        self.ddot_epsilon_nc: float = 0.0 # Угловое ускорение линии носитель-цель, рад/с^2

        self.Delta_r: float = 0.0 # Разность расстояний между носителем и целью и носителем и ракетой, м
        self.dot_Delta_r: float = 0.0 # Скорость изменения разности расстояний, м/с
        self.ddot_Delta_r: float = 0.0 # Ускорение изменения разности расстояний, м/с^2

        self.epsilon_l: float = 0.0 # Угол наклона луча (теленаведение), рад
        self.dot_epsilon_l: float = 0.0 # Угловая скорость луча (теленаведение), рад/с
        self.ddot_epsilon_l: float = 0.0 # Угловое ускорение луча (теленаведение), рад/с^2
        