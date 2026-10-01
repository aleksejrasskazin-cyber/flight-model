import numpy as np
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass
import math
import random

from Headlines import *
from GuidanceMethods import *
from KinematicFunc2D import *
from DATASTRUCTS import *
from rotation import rpm
from scipy.optimize import bisect


class Model:
    """
    Модель интегрирования для пространственного (6-DOF) движения
    С переключаемым режимом:
    - BALLISTIC_MODE = True  -> материальная точка (совпадает с BallPrRo)
    - BALLISTIC_MODE = False -> полная 6-DOF модель (ваша оригинальная)
    """
    
    def __init__(self, calcParams: CalcParameters, inits: InitConditions, 
                 limits: LimitConditions, CData: C_Data, NData: N_Data,
                 RData: R_Data, MethodData: GuidanceMethod_Data):
        
        self.calcParams = calcParams
        self.inits = inits
        self.limits = limits
        self.CData = CData
        self.NData = NData
        self.RData = RData
        self.MethodData = MethodData
        
        # Вектор состояния: 25 компонент
        self.States = np.zeros(25)
        self.error = ""
        self.i = 0
        
        # Инициализация переменных состояния
        self.s = TrajectoryParameters()
        self.s_last = TrajectoryParameters()
        
        self.a_xa_c = 0.0
        self.a_ya_c = 0.0
        self.a_xa_n = 0.0
        self.a_ya_n = 0.0
        self.a_xa_r = 0.0
        self.a_ya_r = 0.0
        
        # Инициализация метода Рунге-Кутта 4 порядка
        self.stepper = self._runge_kutta4_step
        
        # ========== ПЕРЕКЛЮЧАЕМЫЙ РЕЖИМ ==========
        # True  -> баллистический режим (материальная точка, как BallPrRo)
        # False -> полная пространственная модель (ваша оригинальная)
        self.BALLISTIC_MODE = False  # <-- МЕНЯЙТЕ ЗДЕСЬ
        
    def _runge_kutta4_step(self, func, y, t, dt):
        """Реализация метода Рунге-Кутта 4 порядка"""
        k1 = func(y, t)
        k2 = func(y + dt/2 * k1, t + dt/2)
        k3 = func(y + dt/2 * k2, t + dt/2)
        k4 = func(y + dt * k3, t + dt)
        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def _compute_angles(self):
        """
        Вычисление углов атаки, скольжения и скоростного крена для 3D движения.
        """
        # Тригонометрические функции
        cos_Theta = np.cos(self.s.Theta_r)
        sin_Theta = np.sin(self.s.Theta_r)
        cos_theta = np.cos(self.s.theta_r)
        sin_theta = np.sin(self.s.theta_r)
        cos_gamma = np.cos(self.s.gamma_r)
        sin_gamma = np.sin(self.s.gamma_r)
        
        # Разница между курсовым углом и углом рыскания
        dPsi = self.s.Psi_r - self.s.psi_r
        cos_dPsi = np.cos(dPsi)
        sin_dPsi = np.sin(dPsi)
        
        # 1. Угол скольжения (β)
        beta_expr = (cos_Theta * (sin_theta * sin_gamma * cos_dPsi - cos_gamma * sin_dPsi) - 
                      sin_Theta * cos_theta * sin_gamma)
        beta_expr = np.clip(beta_expr, -0.999999, 0.999999)
        self.s.beta = np.arcsin(beta_expr)
        
        # 2. Угол атаки (α)
        cos_beta = np.cos(self.s.beta)
        if abs(cos_beta) > 1e-10:
            alpha_expr = ((sin_theta * cos_gamma * cos_dPsi + sin_gamma * sin_dPsi) * cos_Theta -
                          sin_Theta * cos_theta * cos_gamma) / cos_beta
            alpha_expr = np.clip(alpha_expr, -0.999999, 0.999999)
            self.s.alpha = np.arcsin(alpha_expr)
        else:
            self.s.alpha = 0.0
        
        # 3. Скоростной угол крена (γ_a)
        if abs(cos_Theta) > 1e-10:
            gamma_a_expr = (np.sin(self.s.alpha) * sin_theta + cos_gamma * cos_theta * np.cos(self.s.alpha)) / cos_Theta
            gamma_a_expr = np.clip(gamma_a_expr, -0.999999, 0.999999)
            self.s.gamma_a = np.arccos(gamma_a_expr)
        else:
            self.s.gamma_a = self.s.gamma_r
        
        # 4. Пространственный угол атаки (α_n)
        cos_alpha_n = np.cos(self.s.alpha) * np.cos(self.s.beta)
        cos_alpha_n = np.clip(cos_alpha_n, -1, 1)
        self.s.alpha_n = np.arccos(cos_alpha_n)
    
    def _get_Cxa_ballistic(self, M):
        """Аналитическая функция C_xa(M) из баллистической модели BallPrRo"""
        if M < 0.61:
            return 0.308
        elif 0.61 <= M < 1:
            return 0.505 * (M - 0.61)**2.31 + 0.308
        elif 1 <= M < 1.4:
            return 0.4485 * (M - 1)**0.505 * np.exp(-5.68 * (M - 1)) + 0.551
        elif 1.4 <= M < 4:
            return M / (0.356 * M**2 + 2.237 * M - 1.4)
        else:  # M >= 4
            return 0.302
    
    def Init(self):
        """Инициализация начальных условий"""
        self.s.t = 0.0
        
        # ЦЕЛЬ
        self.s.Theta_c = self.inits.Theta_c0
        self.s.Psi_c = getattr(self.inits, 'Psi_c0', 0.0) 
        self.s.v_c = self.inits.v_c0
        self.s.x_c = self.inits.x_c0
        self.s.y_c = self.inits.y_c0
        self.s.z_c = getattr(self.inits, 'z_c0', 0.0)
        
        # НОСИТЕЛЬ
        self.s.Theta_n = self.inits.Theta_n0
        self.s.Psi_n = getattr(self.inits, 'Psi_n0', 0.0)
        self.s.v_n = self.inits.v_n0
        self.s.x_n = self.inits.x_pusk
        self.s.y_n = self.inits.y_pusk
        self.s.z_n = getattr(self.inits, 'z_pusk', 0.0)
        
        # РАКЕТА
        self.s.Theta_r = self.inits.Theta_r0
        self.s.Psi_r = self.inits.Psi_r0  
        self.s.v_r = self.inits.v_r0
        self.s.x_r = self.inits.x_pusk
        self.s.y_r = self.inits.y_pusk
        self.s.z_r = self.inits.z_pusk
        self.s.theta_r = self.s.Theta_r
        self.s.psi_r = getattr(self.inits, 'psi_r', 0.0)
        self.s.gamma_r = getattr(self.inits, 'gamma_r0', 0.0)
        
        # Угловая скорость крена
        if hasattr(self.inits, 'omega_x0') and self.inits.omega_x0 != 0.0:
            self.s.omega_x = self.inits.omega_x0
        else:
            self.s.omega_x = rpm * 2 * np.pi / 60
        
        self.s.alpha = 0.0
        self.s.delta = 0.0
        self.s.omega_y = 0.0
        self.s.omega_z = 0.0
        self.s.m_r = self.RData.m_r0
        
        # Перегрузки
        self.s.n_xa_c = Linterp(self.CData.n_xa_c_pot, 0.0)
        self.s.n_ya_c = Linterp(self.CData.n_ya_c_pot, 0.0)
        self.s.n_xa_n = Linterp(self.NData.n_xa_n_pot, 0.0)
        self.s.n_ya_n = Linterp(self.NData.n_ya_n_pot, 0.0)
        self.s.n_xa_r = 0.0
        self.s.n_ya_r = 0.0
        self.s.n_ya_r_pot = 0.0
        
        # Массо-центровочные характеристики
        if len(self.RData.MC_tabl) > 0:
            self.s.x_cm = Linterp(self.RData.MC_tabl, self.s.m_r, 1)
            self.s.I_z = Linterp(self.RData.MC_tabl, self.s.m_r, 2)
            self.s.I_x = Linterp(self.RData.MC_tabl, self.s.m_r, 3)
            self.s.I_y = Linterp(self.RData.MC_tabl, self.s.m_r, 4)
        else:
            self.s.x_cm = 0.6
            self.s.I_z = 4.0
            self.s.I_x = 0.5
            self.s.I_y = 4.0
        
        # Тяга и расход
        self.s.P = 0.0
        self.s.G_t = 0.0
        self.s.M = 0.0
        
        # Углы
        self.s.dot_alpha = 0.0
        
        # Моменты
        self.s.M_x = 0.0
        self.s.M_y = 0.0  
        self.s.M_z = 0.0
        self.s.M_z_st = 0.0
        self.s.M_z_omega = 0.0
        
        # Ускорения
        self.a_xa_c = (self.s.n_xa_c - np.sin(self.s.Theta_c)) * g
        self.a_ya_c = (self.s.n_ya_c - np.cos(self.s.Theta_c)) * g
        self.a_xa_n = (self.s.n_xa_n - np.sin(self.s.Theta_n)) * g
        self.a_ya_n = (self.s.n_ya_n - np.cos(self.s.Theta_n)) * g
        self.a_xa_r = (self.s.n_xa_r - np.sin(self.s.Theta_r)) * g
        self.a_ya_r = (self.s.n_ya_r - np.cos(self.s.Theta_r)) * g
        
        # Инициализация вектора состояния (25 компонент)
        # Цель (0-5)
        self.States[0] = self.s.Theta_c
        self.States[1] = self.s.Psi_c
        self.States[2] = self.s.v_c
        self.States[3] = self.s.x_c
        self.States[4] = self.s.y_c
        self.States[5] = self.s.z_c
        
        # Носитель (6-11)
        self.States[6] = self.s.Theta_n
        self.States[7] = self.s.Psi_n 
        self.States[8] = self.s.v_n
        self.States[9] = self.s.x_n
        self.States[10] = self.s.y_n
        self.States[11] = self.s.z_n
        
        # Ракета (12-24)
        self.States[12] = self.s.Theta_r
        self.States[13] = self.s.Psi_r
        self.States[14] = self.s.v_r
        self.States[15] = self.s.x_r
        self.States[16] = self.s.y_r
        self.States[17] = self.s.z_r
        self.States[18] = self.s.theta_r
        self.States[19] = self.s.psi_r
        self.States[20] = self.s.gamma_r
        self.States[21] = self.s.omega_x
        self.States[22] = self.s.omega_y
        self.States[23] = self.s.omega_z
        self.States[24] = self.s.m_r
        
        self.error = ""
    
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Правая часть системы дифференциальных уравнений"""
        
        # Обновление состояния из вектора
        # Цель
        self.s.Theta_c = x[0]
        self.s.Psi_c = x[1]
        self.s.v_c = x[2]
        self.s.x_c = x[3]
        self.s.y_c = x[4]
        self.s.z_c = x[5]
        
        # Носитель
        self.s.Theta_n = x[6]
        self.s.Psi_n = x[7]
        self.s.v_n = x[8]
        self.s.x_n = x[9]
        self.s.y_n = x[10]
        self.s.z_n = x[11]
        
        # Ракета
        self.s.Theta_r = x[12]
        self.s.Psi_r = x[13]
        self.s.v_r = x[14]
        self.s.x_r = x[15]
        self.s.y_r = x[16]
        self.s.z_r = x[17]
        self.s.theta_r = x[18]
        self.s.psi_r = x[19]
        self.s.gamma_r = x[20]
        self.s.omega_x = x[21]
        self.s.omega_y = x[22]
        self.s.omega_z = x[23]
        self.s.m_r = x[24]
        
        # ========== БАЛЛИСТИЧЕСКИЙ РЕЖИМ (материальная точка) ==========
        if self.BALLISTIC_MODE:
            # Принудительно фиксируем все углы и угловые скорости
            self.s.theta_r = self.s.Theta_r
            self.s.psi_r = 0.0
            self.s.Psi_r = 0.0
            self.s.gamma_r = 0.0
            self.s.gamma_a = 0.0
            self.s.alpha = 0.0
            self.s.beta = 0.0
            self.s.alpha_n = 0.0
            self.s.omega_x = 0.0
            self.s.omega_y = 0.0
            self.s.omega_z = 0.0
        
        # ========== ПАРАМЕТРЫ АТМОСФЕРЫ ==========
        rho = rho_atm(self.s.y_r)
        p_atm_val = p_atm(self.s.y_r)
        a_atm_val = a_atm(self.s.y_r)
        
        # Число Маха
        if a_atm_val > 1e-3:
            self.s.M = self.s.v_r / a_atm_val
        else:
            self.s.M = 0.0
        
        # Перегрузки цели и носителя
        self.s.n_xa_c = Linterp(self.CData.n_xa_c_pot, t)
        self.s.n_ya_c = Linterp(self.CData.n_ya_c_pot, t)
        self.s.n_xa_n = Linterp(self.NData.n_xa_n_pot, t)
        self.s.n_ya_n = Linterp(self.NData.n_ya_n_pot, t)
        
        # ========== ТЯГА И РАСХОД ==========
        P_y0 = Linterp(self.RData.t_P_G_tabl, t, 1)
        sign_P_y0 = sign(P_y0)
        H_P_y0 = H(P_y0)
        
        self.s.P = P_y0 + (p_atm(0.0) - p_atm_val) * self.RData.S_a * (sign_P_y0 * H_P_y0)
        self.s.G_t = Linterp(self.RData.t_P_G_tabl, t, 2)
        
        # Массо-центровочные характеристики
        if len(self.RData.MC_tabl) > 0:
            self.s.x_cm = Linterp(self.RData.MC_tabl, self.s.m_r, 1)
            self.s.I_z = Linterp(self.RData.MC_tabl, self.s.m_r, 2)
            self.s.I_x = Linterp(self.RData.MC_tabl, self.s.m_r, 3)
            self.s.I_y = Linterp(self.RData.MC_tabl, self.s.m_r, 4)
        self._compute_angles()#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ========== АЭРОДИНАМИЧЕСКИЕ КОЭФФИЦИЕНТЫ ==========
        if self.BALLISTIC_MODE:
            # Баллистический режим: аналитическая C_xa(M), все остальные = 0
            c_ya = 0.0
            c_xa_pas = self._get_Cxa_ballistic(self.s.M)
            c_za = 0.0
            c_x = c_xa_pas
            c_y = 0.0
            c_z = 0.0
            xd = self.s.x_cm
            m_z = 0.0
            m_y = 0.0
        else:
            # Полный режим: интерполяция из таблиц
            try:
                c_ya = TriLinterp(self.RData.AD_tabl, self.s.M, self.s.alpha, self.s.delta, 4)
                c_xa_pas = TriLinterp(self.RData.AD_tabl, self.s.M, self.s.alpha_n, self.s.delta, 3)
                c_za = TriLinterp(self.RData.AD_tabl, self.s.M, self.s.beta, self.s.delta, 4)
                c_x = TriLinterp(self.RData.AD_tabl, self.s.M, self.s.alpha, self.s.delta, 5)
                c_y = TriLinterp(self.RData.AD_tabl, self.s.M, self.s.alpha, self.s.delta, 6)
                c_z = TriLinterp(self.RData.AD_tabl, self.s.M, self.s.beta, self.s.delta, 6)
                xd = TriLinterp(self.RData.AD_tabl, self.s.M, self.s.alpha_n, self.s.delta, 8)
                m_z = TriLinterp(self.RData.AD_tabl, self.s.M, self.s.alpha, self.s.delta, 7)
                m_y = m_z
            except:
                c_ya = 0.0
                c_xa_pas = 0.0
                c_za = 0.0
                c_x = 0.0
                c_y = 0.0
                c_z = 0.0
                xd = self.s.x_cm
                m_z = 0.0
                m_y = 0.0
        
        # Масштабирующие коэффициенты (только для полного режима)
        if hasattr(self.RData, 'k_cx') and not self.BALLISTIC_MODE:
            c_xa_pas = c_xa_pas * self.RData.k_cx
        if hasattr(self.RData, 'k_cy') and not self.BALLISTIC_MODE:
            c_ya = c_ya * self.RData.k_cy
            c_za = c_za * self.RData.k_cy
        
        # Высотная поправка
        if self.BALLISTIC_MODE:
            Delta_c_xa0_h = 0.0
            c_xa0_dn = 0.0
        else:
            try:
                Delta_c_xa0_h = BiLinterp(self.RData.AD_tabl2, self.s.y_r, self.s.M, 2)
                c_xa0_dn = BiLinterp(self.RData.AD_tabl2, self.s.y_r, self.s.M, 3)
            except:
                Delta_c_xa0_h = 0.0
                c_xa0_dn = 0.0
        
        sign_P = sign(self.s.P)
        H_P_val = H(self.s.P)
        
        c_xa = c_xa_pas + Delta_c_xa0_h - c_xa0_dn * self.RData.S_a / self.RData.S_m * (sign_P * H_P_val)
        self.s.c_xa = c_xa
        
        # ========== АЭРОДИНАМИЧЕСКИЕ СИЛЫ ==========
        q = rho * self.s.v_r**2 / 2.0 * self.RData.S_m
        
        X_a = c_xa * q
        Y_a =  c_ya * q
        Z_a = c_za * q
        
        self.s.X_a = X_a
        self.s.Y_a = Y_a
        self.s.Z_a = Z_a
        
        self.s.c_y = c_y
        self.s.c_ya = c_ya
        
        # ========== ПЕРЕГРУЗКИ ==========
        if self.s.m_r * g > 1e-3:
            self.s.n_xa_r = (-X_a + self.s.P * np.cos(self.s.alpha_n)) / (self.s.m_r * g)
            
            if self.BALLISTIC_MODE:
                # Баллистический режим: нет нормальной перегрузки
                self.s.n_ya_r = 0.0
            else:
                # Полный режим: есть подъёмная сила и вклад от тяги
                self.s.n_ya_r = 0* (Y_a + self.s.P * np.sin(self.s.alpha)) / (self.s.m_r * g)
            
            self.s.n_za_r = (Z_a - self.s.P * np.cos(self.s.alpha) * np.sin(self.s.beta)) / (self.s.m_r * g)
        else:
            self.s.n_xa_r = 0.0
            self.s.n_ya_r = 0.0
            self.s.n_za_r = 0.0
        
        # ========== СИЛЫ В СВЯЗАННОЙ СК ==========
        X = c_x * q
        Y = c_y * q
        Z = c_z * q
        
        self.s.X = X
        self.s.Y = Y
        self.s.Z = Z
        
        # ========== МОМЕНТЫ ==========
        if self.BALLISTIC_MODE:
            # Баллистический режим: все моменты обнулены
            M_z_st = 0.0
            M_z_omega = 0.0
            M_y_st = 0.0
            M_y_omega = 0.0
        else:
            # Полный режим: моменты из аэродинамики
            M_z_st = Y * (self.s.x_cm - xd)
            M_y_st = Z * (self.s.x_cm - xd)
            
            # Демпфирующие моменты
            if self.s.v_r > 1.0:
                try:
                    m_z_bar_omega_z = BiLinterp(self.RData.AD_tabl3, self.s.x_cm, self.s.M, 2)
                    m_y_bar_omega_y = BiLinterp(self.RData.AD_tabl3, self.s.x_cm, self.s.M, 2)
                except:
                    m_z_bar_omega_z = 0.0
                    m_y_bar_omega_y = 0.0
                
                q_damp = q * self.RData.L * self.RData.L / self.s.v_r
                M_z_omega = m_z_bar_omega_z * q_damp * self.s.omega_z
                M_y_omega = m_y_bar_omega_y * q_damp * self.s.omega_y
            else:
                M_z_omega = 0.0
                M_y_omega = 0.0
        
        self.s.M_z_st = M_z_st
        self.s.M_z_omega = M_z_omega
        self.s.M_z = M_z_st + M_z_omega
        self.s.M_y = M_y_st + M_y_omega
        self.s.M_x = 0.0
        
        # ========== ПОТРЕБНАЯ ПЕРЕГРУЗКА (только для режима управления) ==========
        if not self.BALLISTIC_MODE and self.MethodData.Method == GuidanceMethod.Null:
            self.s.n_ya_r_pot = n_ya_pot_Null(self.s.dot_epsilon_rc, self.s.v_r, 
                                              self.MethodData.k, self.s.Theta_r)
        
        # ========== УПРАВЛЕНИЕ (только для режима управления) ==========
        if not self.BALLISTIC_MODE and len(self.RData.AD_tabl) > 0:
            # ... код поиска балансировочных углов (оставлен как в оригинале) ...
            self.s.delta = 0.0
        else:
            self.s.delta = 0.0
        
        # ========== УСКОРЕНИЯ ==========
        self.a_xa_c = (self.s.n_xa_c - np.sin(self.s.Theta_c)) * g
        self.a_ya_c = (self.s.n_ya_c - np.cos(self.s.Theta_c)) * g
        self.a_xa_n = (self.s.n_xa_n - np.sin(self.s.Theta_n)) * g
        self.a_ya_n = (self.s.n_ya_n - np.cos(self.s.Theta_n)) * g
        self.a_xa_r = (self.s.n_xa_r - np.sin(self.s.Theta_r)) * g
        self.a_ya_r = (self.s.n_ya_r - np.cos(self.s.Theta_r)) * g
        
        # ========== ОТНОСИТЕЛЬНЫЕ ПАРАМЕТРЫ ==========
        self.s.r_rc = r_3d(self.s.x_r, self.s.y_r, self.s.z_r, 
                           self.s.x_c, self.s.y_c, self.s.z_c)
        self.s.epsilon_rc = epsilon(self.s.x_r, self.s.y_r, self.s.x_c, self.s.y_c)
        self.s.beta_rc = beta_3d(self.s.x_r, self.s.y_r, self.s.z_r,
                                 self.s.x_c, self.s.y_c, self.s.z_c)
        self.s.dot_epsilon_rc = dot_epsilon(self.s.x_r, self.s.y_r, self.s.x_c, self.s.y_c,
                                           self.s.v_r, self.s.v_c, self.s.Theta_r, self.s.Theta_c)
        self.s.dot_beta_rc = dot_beta_3d(self.s.x_r, self.s.y_r, self.s.z_r,
                                         self.s.x_c, self.s.y_c, self.s.z_c,
                                         self.s.v_r, self.s.v_c, 
                                         self.s.Theta_r, self.s.Theta_c,
                                         self.s.Psi_r, self.s.Psi_c)
        
        
        if not self.BALLISTIC_MODE and self.i % 50 == 0:
            print(f"\nt={self.s.t:.2f}с")
            print(f"  X_a={X_a:.1f}Н, Y_a={Y_a:.1f}Н")
            print(f"  n_xa_r={self.s.n_xa_r:.3f}, n_ya_r={self.s.n_ya_r:.3f}")
            print(f"  alpha={np.rad2deg(self.s.alpha):.2f}°")
            print(f"  P*sin(alpha)={self.s.P * np.sin(self.s.alpha):.1f}Н")
            print(f"  v_r={self.s.v_r:.1f}м/с, y_r={self.s.y_r:.1f}м")
        
        
        
        # ========== ПРОИЗВОДНЫЕ ==========
        dxdt = np.zeros(25)
        
        # Цель
        if abs(self.s.v_c) > 1.0:
            dxdt[0] = (self.s.n_ya_c - np.cos(self.s.Theta_c)) * g / self.s.v_c
            dxdt[1] = -self.s.n_za_c * g / (self.s.v_c * np.cos(self.s.Theta_c))
        else:
            dxdt[0] = 0.0
            dxdt[1] = 0.0
        
        dxdt[2] = (self.s.n_xa_c - np.sin(self.s.Theta_c)) * g
        dxdt[3] = self.s.v_c * np.cos(self.s.Theta_c) * np.cos(self.s.Psi_c)
        dxdt[4] = self.s.v_c * np.sin(self.s.Theta_c)
        dxdt[5] = -self.s.v_c * np.cos(self.s.Theta_c) * np.sin(self.s.Psi_c)
        
        # Носитель
        if abs(self.s.v_n) > 1.0:
            dxdt[6] = (self.s.n_ya_n - np.cos(self.s.Theta_n)) * g / self.s.v_n
            dxdt[7] = -self.s.n_za_n * g / (self.s.v_n * np.cos(self.s.Theta_n))
        else:
            dxdt[6] = 0.0
            dxdt[7] = 0.0
        
        dxdt[8] = (self.s.n_xa_n - np.sin(self.s.Theta_n)) * g
        dxdt[9] = self.s.v_n * np.cos(self.s.Theta_n) * np.cos(self.s.Psi_n)
        dxdt[10] = self.s.v_n * np.sin(self.s.Theta_n)
        dxdt[11] = -self.s.v_n * np.cos(self.s.Theta_n) * np.sin(self.s.Psi_n)
        
        # ========== ПРОИЗВОДНЫЕ ДЛЯ РАКЕТЫ ==========
        
        # dTheta_r/dt
        if abs(self.s.v_r) > 1.0:
            if self.BALLISTIC_MODE:
                # Баллистический режим: только гравитация
                dxdt[12] = -np.cos(self.s.Theta_r) * g / self.s.v_r
            else:
                # Полный режим: с учётом перегрузок
                dxdt[12] = (self.s.n_ya_r * np.cos(self.s.gamma_a) - 
                            self.s.n_za_r * np.sin(self.s.gamma_a) - 
                            np.cos(self.s.Theta_r)) * g / self.s.v_r
        else:
            dxdt[12] = 0.0
        
        # dPsi_r/dt
        if self.BALLISTIC_MODE:
            dxdt[13] = 0.0
        else:
            if abs(self.s.v_r) > 1.0 and abs(np.cos(self.s.Theta_r)) > 1e-3:
                dxdt[13] = -(self.s.n_ya_r * np.sin(self.s.gamma_a) +
                             self.s.n_za_r * np.cos(self.s.gamma_a)) * g / (self.s.v_r * np.cos(self.s.Theta_r))
            else:
                dxdt[13] = 0.0
        
        # dv_r/dt
        dxdt[14] = (self.s.n_xa_r - np.sin(self.s.Theta_r)) * g
        
        # dx_r/dt, dy_r/dt, dz_r/dt
        if self.BALLISTIC_MODE:
            dxdt[15] = self.s.v_r * np.cos(self.s.Theta_r)
            dxdt[16] = self.s.v_r * np.sin(self.s.Theta_r)
            dxdt[17] = 0.0
        else:
            dxdt[15] = self.s.v_r * np.cos(self.s.Theta_r) * np.cos(self.s.Psi_r)
            dxdt[16] = self.s.v_r * np.sin(self.s.Theta_r)
            dxdt[17] = -self.s.v_r * np.cos(self.s.Theta_r) * np.sin(self.s.Psi_r)
        
        # ========== ПРОИЗВОДНЫЕ УГЛОВ И УГЛОВЫХ СКОРОСТЕЙ ==========
        if self.BALLISTIC_MODE:
            # Баллистический режим: все производные углов обнулены
            dxdt[18] = dxdt[12]  # dtheta_r/dt
            dxdt[19] = 0.0  # dpsi_r/dt
            dxdt[20] = 0.0  # dgamma_r/dt
            dxdt[21] = 0.0  # domega_x/dt
            dxdt[22] = 0.0  # domega_y/dt
            dxdt[23] = 0.0  # domega_z/dt
        else:
            # Полный режим: пространственная динамика
            # dtheta_r/dt
            dxdt[18] = self.s.omega_y * np.sin(self.s.gamma_r) + self.s.omega_z * np.cos(self.s.gamma_r)
            
            # dpsi_r/dt
            if abs(np.cos(self.s.theta_r)) > 1e-3:
                dxdt[19] = (self.s.omega_y * np.cos(self.s.gamma_r) - 
                            self.s.omega_z * np.sin(self.s.gamma_r)) / np.cos(self.s.theta_r)
            else:
                dxdt[19] = 0.0
            
            # dgamma_r/dt
            dxdt[20] = (self.s.omega_x - 
                        np.tan(self.s.theta_r) * (self.s.omega_y * np.cos(self.s.gamma_r) - 
                                                  self.s.omega_z * np.sin(self.s.gamma_r)))
            
            # domega_x/dt
            if self.s.I_x > 0.01:
                dxdt[21] = (self.s.M_x - (self.s.I_z - self.s.I_y) * self.s.omega_y * self.s.omega_z) / self.s.I_x
            else:
                dxdt[21] = 0.0
            
            # domega_y/dt
            if self.s.I_y > 0.01:
                dxdt[22] = (self.s.M_y - (self.s.I_x - self.s.I_z) * self.s.omega_z * self.s.omega_x) / self.s.I_y
            else:
                dxdt[22] = 0.0
            
            # domega_z/dt
            if self.s.I_z > 0.01:
                dxdt[23] = (self.s.M_z - (self.s.I_y - self.s.I_x) * self.s.omega_x * self.s.omega_y) / self.s.I_z
            else:
                dxdt[23] = 0.0
        
        # dm_r/dt
        dxdt[24] = -self.s.G_t
        
        return dxdt
    
    def DoStep(self, t: float, dt: float = None):
        """Выполнение одного шага интегрирования"""
        self.s_last = self.s
        
        states_before = self.States.copy()
        
        try:
            self.States = self.stepper(self.__call__, self.States, t, dt)
            
            # Обновление состояния из вектора
            # Цель
            self.s.Theta_c = self.States[0]
            self.s.Psi_c = self.States[1]     
            self.s.v_c = self.States[2]
            self.s.x_c = self.States[3]
            self.s.y_c = self.States[4]
            self.s.z_c = self.States[5]
            
            # Носитель
            self.s.Theta_n = self.States[6]
            self.s.Psi_n = self.States[7] 
            self.s.v_n = self.States[8]
            self.s.x_n = self.States[9]
            self.s.y_n = self.States[10]
            self.s.z_n = self.States[11]
            
            # Ракета
            self.s.Theta_r = self.States[12]
            self.s.Psi_r = self.States[13]
            self.s.v_r = self.States[14]
            self.s.x_r = self.States[15]
            self.s.y_r = self.States[16]
            self.s.z_r = self.States[17]
            self.s.theta_r = self.States[18]
            self.s.psi_r = self.States[19]
            self.s.gamma_r = self.States[20]
            self.s.omega_x = self.States[21]
            self.s.omega_y = self.States[22]
            self.s.omega_z = self.States[23]
            self.s.m_r = self.States[24]
            
            self.s.t = t + dt
            self.i += 1
            
            return True
            
        except Exception as e:
            self.error = str(e)
            self.States = states_before
            return False