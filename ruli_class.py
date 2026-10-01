"ruli_class"
import numpy as np
import math
import AeroBDSM
from math import *
import matplotlib.pyplot as plt
from Atmosphere import atmo
from mpl_toolkits.mplot3d import Axes3D
from Krilo_class import WingGeometry, WingAerodynamicCalculator as WingCalculator

class RudderGeometry:
    """Класс для хранения геометрических параметров рулей"""
     
    def __init__(self):
        # Основные геометрические параметры
         
        self.K_w = 1 
        self.Re = 10**6
        self.b = 0.13
        self.c = 25 * 0.001
        self.xx_t = 0
        self.h = 0
        self.S = 0.013  # площадь несущей поверхности
        self.η = 2.094  # сужение несущей поверхности
        self.X_05 = np.deg2rad(48)
        self.l = 0.281  
        self.l_k = 0.281  
        self.cc = self.c / self.b  # относительная толщина профиля
        self.xx_c = 0.5  # относительная координата линии максимальных толщин
        self.x_c = self.xx_c * self.b
        self.hh = self.h / self.b
        self.m = 0  # расстояние в долях хорды
        self.lambda_p = 1  # относительное удлинение консолей
        self.eta_p = 2.89  # относительное сужение консолей
        self.lambda_k = 0.66
        self.eta_k = 2.89  # относительное сужение консолей
        self.delta = 0  # угол отклонения руля
        
        # Расчетные параметр
        self.zeta_k = 1 / self.eta_k
        self.zeta_p = 1 / self.eta_p
        self.zeta = 1 / self.η  # обратное сужение несущей поверхности
        
        # Расчет хорд
        self.b_b = self.S / self.l * 2 / (1 + self.zeta_p)  # бортовая хорда
        self.b1 = self.b_b * self.zeta_p  # концевая хорда
        self.λ = self.l**2 / self.S  # удлинение несущей поверхности
        
        # Расчет углов стреловидности
        self.X_0 = atan(tan(self.X_05) + 2/self.lambda_p * (1-self.zeta_p)/(1+self.zeta_p))
        self.X_1 = atan(tan(self.X_0) + 4/self.lambda_p * (1-self.zeta_p)/(1+self.zeta_p))
        self.X_c = atan(tan(self.X_0) - 4*self.xx_c/self.lambda_p * (1-self.zeta_p)/(1+self.zeta_p))
        self.X_m = atan(tan(self.X_0) - 4*self.m/self.lambda_p * (1-self.zeta_p)/(1+self.zeta_p))
        
        # Расчет параметров САХ
        self.bA = 4/3 * self.S/self.l * (1 - self.zeta_p/(1+self.zeta_p)**2)
        self.zA = self.l/6 * ((1+2*self.zeta_p)/(1+self.zeta_p))
        self.xA = self.zA * tan(self.X_0)
        self.x_цS = self.xA + self.bA/2  # координата центра тяжести площади консоли
        self.z_цS = self.zA
        
        # Диапазоны расчетов
        self.wing_geo = WingGeometry()
        self.wing_calc = WingCalculator(self.wing_geo)
        self.H = self.wing_geo.H 
        
        
        self.M_list = self.wing_geo.M_list
        self.alpha_list = self.wing_geo.alpha_list # Углы атаки в радианах   
        

class RudderAerodynamicCalculator:
    """Класс для расчета аэродинамических характеристик рулей"""
    
    def __init__(self, geometry):
        self.geo = geometry
        
    def calculate_effective_angle(self, alpha):
        """Расчет эффективного угла атаки с учетом отклонения руля"""
        nn_1 = AeroBDSM.get_nn_Eff1(self.geo.l, self.geo.l_k, self.geo.zeta_p)
        k_sch = 1
        n_eff = nn_1 * k_sch
        # print("nn_1 ->", nn_1)
        return alpha + n_eff * self.geo.delta
    
    def calculate_cy_alpha(self, M):
        """Расчет производной коэффициента нормальной силы"""
        cy_alpha = AeroBDSM.get_Cy_alpha_IsP(M, self.geo.lambda_p, self.geo.cc, 
                                            self.geo.X_05, self.geo.eta_p)
        return cy_alpha.Value
    
    def calculate_cn_iz_p(self, M, alpha_eff):
        """Расчет коэффициента нормальной силы"""
        cy_alpha = self.calculate_cy_alpha(M)
        A = AeroBDSM.get_A_IsP(M, self.geo.zeta_p, cy_alpha)
        return (cy_alpha * sin(alpha_eff) * cos(alpha_eff) + 
                A * sin(alpha_eff)**2 * np.sign(alpha_eff))
    
    def calculate_critical_mach(self, M, cn):
        """Расчет критического числа Маха"""
        Mcr_pr = AeroBDSM.get_Mcr_pr(self.geo.cc, abs(cn), self.geo.xx_c)
        ΔMcr_λ = AeroBDSM.get_DELTA_Mcr0_lambda(Mcr_pr, self.geo.lambda_p)
        ΔMcr_χ = AeroBDSM.get_DELTA_Mcr0_chi(Mcr_pr, self.geo.X_m)
        return Mcr_pr + ΔMcr_λ + ΔMcr_χ
    
    def calculate_wave_drag(self, M, Mcr_iz_p):
        """Расчет коэффициента волнового сопротивления"""
        if M < Mcr_iz_p:
            return 0
        elif Mcr_iz_p < M < 1:
            a0 = AeroBDSM.get_Cxw_Rhomb_M1(1, self.geo.cc, self.geo.zeta_p, 
                                          self.geo.X_c, self.geo.lambda_p)
            a1 = a0 / (Mcr_iz_p**2 - 2*Mcr_iz_p + 1)
            a2 = -2 * a1 * Mcr_iz_p
            a3_koef = AeroBDSM.get_Cxw_Rhomb_M1(1, self.geo.cc, self.geo.zeta_p, 
                                               self.geo.X_c, self.geo.lambda_p) - a1*(1-2*Mcr_iz_p)
            return a1 * M**2 + a2 * M + a3_koef
        else:
            cxa_w = AeroBDSM.get_Cxw_Rhomb_M1(M, self.geo.cc, self.geo.zeta_p, 
                                             self.geo.X_c, self.geo.lambda_p)
            sigma_pw = AeroBDSM.get_Sigma_pw(M, self.geo.X_c)
            return cxa_w * (1 + sigma_pw * (self.geo.K_w - 1))
    
    def calculate_profile_drag(self, M):
        """Расчет коэффициента профильного сопротивления"""
        Re = (self.geo.bA * M * atmo.a(self.geo.H)) / atmo.nu(self.geo.H)
        Cf_M0 = AeroBDSM.get_Cf_M0(Re, self.geo.xx_t)
        Sigma_M = AeroBDSM.get_Sigma_M(M, self.geo.xx_t)
        Cf = Cf_M0 * Sigma_M
        xx_t_iz_p = self.geo.xx_t * AeroBDSM.get_Sigma_chi(self.geo.X_0)
        Sigma_C = AeroBDSM.get_Sigma_c(self.geo.cc, xx_t_iz_p)
        return 2 * Cf * Sigma_C
    
    def calculate_base_drag(self, M):
        """Расчет донного сопротивления"""
        return AeroBDSM.get_Cxdon_pr(M) * self.geo.hh
    
    def calculate_zero_angle_drag(self, M, cxa_w, cxa_prof, cxa_don):
        """Расчет лобового сопротивления при нулевом угле атаки"""
        return cxa_prof + cxa_don + cxa_w
    
    def calculate_cy_rul_iz_p(self, cn, c_xa_0_iz_profil_rul, alpha, alpha_eff):
        """Расчет коэффициента нормальной силы руля"""
        return cn * cos(alpha_eff - alpha) - c_xa_0_iz_profil_rul * sin(alpha_eff - alpha)
    
    def calculate_induced_drag(self, cy_rul, alpha_eff, c_fa=0):
        """Расчет индуктивного сопротивления"""
        return cy_rul * sin(alpha_eff) - c_fa * cos(alpha_eff)
    
    def calculate_total_drag(self, c_xa_0_iz_profil_rul, cxa_i):
        """Расчет общего лобового сопротивления"""
        return c_xa_0_iz_profil_rul + cxa_i
    
    def calculate_cx_iz_p(self, cxa, cy, alpha):
        """Расчет коэффициента продольной силы"""
        if cos(alpha) != 0:
            return (cxa - cy * sin(alpha)) / cos(alpha)
        else:
            return 0
    
    def calculate_cya_iz_p(self, cy, cxa, alpha):
        """Расчет коэффициента подъемной силы"""
        if cos(alpha) != 0:
            return (cy - cxa * sin(alpha)) / cos(alpha)
        else:
            return 0
    
    def calculate_focus_position(self, M):
        """Расчет положения фокуса по углу атаки"""
        xxF = AeroBDSM.get_Coordinate_xxF_IsP(M, self.geo.lambda_p, self.geo.X_05, self.geo.zeta_p)
        return self.geo.xA + xxF * self.geo.bA
    
    def calculate_lateral_focus(self, M):
        """Расчет поперечной координаты фокуса"""
        zz_d = AeroBDSM.get_Coordinate_zz_d(M, self.geo.lambda_p, self.geo.X_05, self.geo.zeta_p)
        return self.geo.l / 2 * zz_d
    
    def calculate_center_of_pressure(self, M, alpha_eff, cy_alpha, cy_rul):
        """Расчет центра давления"""
        A = AeroBDSM.get_A_IsP(M, self.geo.zeta_p, cy_alpha)
        x_focus = self.calculate_focus_position(M)
        
        numerator = (cy_alpha * sin(alpha_eff) * cos(alpha_eff) * x_focus + 
                    A * sin(alpha_eff)**2 * np.sign(alpha_eff) * self.geo.x_цS)
        
        if cy_rul != 0:
            return numerator / cy_rul
        else:
            return 0
    
    def calculate_pitching_moment_derivative(self, cy_alpha, x_cp, x_cg):
        """Расчет производной коэффициента продольного момента"""
        if self.geo.bA != 0:
            return cy_alpha * (x_cg - x_cp) / self.geo.bA
        else:
            return 0
    
    def run_calculations(self):
        """Основной метод выполнения расчетов"""
        results = {}
        
        for M in self.geo.M_list:
            results[M] = {}
            
            # Базовые характеристики
            results[M]['cy_alpha'] = self.calculate_cy_alpha(M)
            results[M]['cxa_w'] = 0  # Будет пересчитано позже
            results[M]['cxa_prof'] = self.calculate_profile_drag(M)
            results[M]['cxa_don'] = self.calculate_base_drag(M)
            results[M]['c_xa_0_iz_profil_rul'] = 0  # Будет пересчитано позже
            results[M]['x_alpha_F'] = self.calculate_focus_position(M)
            results[M]['z_alpha_F'] = self.calculate_lateral_focus(M)
            
            # Расчет для разных углов атаки
            results[M]['alpha_results'] = {}
            
            for alpha in self.geo.alpha_list:
                alpha_deg = round(np.rad2deg(alpha))
                results[M]['alpha_results'][alpha_deg] = {}
                
                # Эффективный угол атаки
                alpha_eff = self.calculate_effective_angle(alpha)
                results[M]['alpha_results'][alpha_deg]['alpha_eff'] = alpha_eff
                
                # Коэффициент нормальной силы
                cn = self.calculate_cn_iz_p(M, alpha_eff)
                results[M]['alpha_results'][alpha_deg]['cn'] = cn
                
                # Критическое число Маха
                Mcr_iz_p = self.calculate_critical_mach(M, cn)
                results[M]['alpha_results'][alpha_deg]['Mcr_iz_p'] = Mcr_iz_p
                
                # Волновое сопротивление
                cxa_w = self.calculate_wave_drag(M, Mcr_iz_p)
                results[M]['cxa_w'] = cxa_w  # Обновляем для данного M
                
                # Лобовое сопротивление при нулевом угле атаки
                c_xa_0_iz_profil_rul = self.calculate_zero_angle_drag(M, cxa_w, 
                                                      results[M]['cxa_prof'], 
                                                      results[M]['cxa_don'])
                results[M]['c_xa_0_iz_profil_rul'] = c_xa_0_iz_profil_rul  # Обновляем для данного M
                
                # Коэффициент нормальной силы руля
                cy_rul = self.calculate_cy_rul_iz_p(cn, c_xa_0_iz_profil_rul, alpha, alpha_eff)
                results[M]['alpha_results'][alpha_deg]['cy_iz_p'] = cy_rul
                
                # Индуктивное сопротивление
                cxa_i = self.calculate_induced_drag(cy_rul, alpha_eff)
                results[M]['alpha_results'][alpha_deg]['cxa_i'] = cxa_i
                
                # Общее лобовое сопротивление
                cxa_total = self.calculate_total_drag(c_xa_0_iz_profil_rul, cxa_i)
                results[M]['alpha_results'][alpha_deg]['cxa_iz_p'] = cxa_total
                
                # Коэффициент продольной силы
                cx_iz_p = self.calculate_cx_iz_p(cxa_total, cy_rul, alpha)
                results[M]['alpha_results'][alpha_deg]['cx_iz_p'] = cx_iz_p
                
                # Коэффициент подъемной силы
                cya_iz_p = self.calculate_cya_iz_p(cy_rul, cxa_total, alpha)
                results[M]['alpha_results'][alpha_deg]['cya_iz_p'] = cya_iz_p
                
                # Центр давления
                x_cp = self.calculate_center_of_pressure(M, alpha_eff, 
                                                        results[M]['cy_alpha'], cy_rul)
                results[M]['alpha_results'][alpha_deg]['x_d_iz_p'] = x_cp
                
                # Производная коэффициента продольного момента
                mz_alpha = self.calculate_pitching_moment_derivative(
                    results[M]['cy_alpha'], x_cp, self.geo.x_цS)
                results[M]['alpha_results'][alpha_deg]['mz_alpha_iz_p'] = mz_alpha
                
                # Коэффициент продольного момента
                results[M]['alpha_results'][alpha_deg]['mz_iz_p'] = (
                    cy_rul * (self.geo.x_цS - x_cp) / self.geo.bA)
        
        return results
    
    def plot_all_graphs(self, results):
        """Построение всех графиков"""
        
        # Проверка наличия данных
        if not results:
            print("Нет данных для построения графиков")
            return
            
        M_values = sorted(list(results.keys()))
        if not M_values:
            print("Нет данных по числам Маха")
            return
            
        # Получение всех углов атаки из первого доступного числа Маха
        first_M = M_values[0]
        if 'alpha_results' not in results[first_M] or not results[first_M]['alpha_results']:
            print("Нет данных по углам атаки")
            return
            
        alpha_deg_values = sorted(list(results[first_M]['alpha_results'].keys()))
        
        # Создание фигур
        self.plot_cy_cx_graphs(results, M_values, alpha_deg_values)
        self.plot_cya_cxa_graphs(results, M_values, alpha_deg_values)
        self.plot_focus_moment_graphs(results, M_values, alpha_deg_values)
        
    
    def plot_cy_cx_graphs(self, results, M_values, alpha_deg_values):
        """Графики cy_iz_p и cx_iz_p"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Коэффициенты нормальной и продольной силы рулей', fontsize=16)
        
        # 1. cy_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            cy_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cy_values.append(results[M]['alpha_results'][alpha_deg]['cy_iz_p'])
                else:
                    cy_values.append(np.nan)
            
            if not all(np.isnan(cy_values)):
                axes[0, 0].plot(M_values, cy_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 0].set_xlabel('Число Маха, M')
        axes[0, 0].set_ylabel('cy_iz_p')
        axes[0, 0].set_title('Коэффициент нормальной силы от числа Маха')
        if axes[0, 0].get_legend_handles_labels()[0]:
            axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. cy_iz_p от угла атаки для разных M
        # Автоматический выбор характерных чисел Маха
        if len(M_values) >= 4:
            selected_M_indices = [0, len(M_values)//3, 2*len(M_values)//3, -1]
            selected_M = [M_values[i] for i in selected_M_indices]
        else:
            selected_M = M_values
            
        for M in selected_M:
            if M in results:
                cy_values = []
                valid_alpha_values = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        cy_values.append(results[M]['alpha_results'][alpha_deg]['cy_iz_p'])
                        valid_alpha_values.append(alpha_deg)
                
                if cy_values:
                    axes[0, 1].plot(valid_alpha_values, cy_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[0, 1].set_xlabel('Угол атаки, °')
        axes[0, 1].set_ylabel('cy_iz_p')
        axes[0, 1].set_title('Коэффициент нормальной силы от угла атаки')
        if axes[0, 1].get_legend_handles_labels()[0]:
            axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. cx_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            cx_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cx_values.append(results[M]['alpha_results'][alpha_deg]['cx_iz_p'])
                else:
                    cx_values.append(np.nan)
            
            if not all(np.isnan(cx_values)):
                axes[1, 0].plot(M_values, cx_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 0].set_xlabel('Число Маха, M')
        axes[1, 0].set_ylabel('cx_iz_p')
        axes[1, 0].set_title('Коэффициент продольной силы от числа Маха')
        if axes[1, 0].get_legend_handles_labels()[0]:
            axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. cx_iz_p от угла атаки для разных M
        for M in selected_M:
            if M in results:
                cx_values = []
                valid_alpha_values = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        cx_values.append(results[M]['alpha_results'][alpha_deg]['cx_iz_p'])
                        valid_alpha_values.append(alpha_deg)
                
                if cx_values:
                    axes[1, 1].plot(valid_alpha_values, cx_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[1, 1].set_xlabel('Угол атаки, °')
        axes[1, 1].set_ylabel('cx_iz_p')
        axes[1, 1].set_title('Коэффициент продольной силы от угла атаки')
        if axes[1, 1].get_legend_handles_labels()[0]:
            axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cya_cxa_graphs(self, results, M_values, alpha_deg_values):
        """Графики cya_iz_p и cxa_iz_p"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Коэффициенты подъемной силы и сопротивления рулей', fontsize=16)
        
        # 1. cya_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            cya_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cya_values.append(results[M]['alpha_results'][alpha_deg]['cya_iz_p'])
                else:
                    cya_values.append(np.nan)
            
            if not all(np.isnan(cya_values)):
                axes[0, 0].plot(M_values, cya_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 0].set_xlabel('Число Маха, M')
        axes[0, 0].set_ylabel('cya_iz_p')
        axes[0, 0].set_title('Коэффициент подъемной силы от числа Маха')
        if axes[0, 0].get_legend_handles_labels()[0]:
            axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. cya_iz_p от угла атаки для разных M
        # Автоматический выбор характерных чисел Маха
        if len(M_values) >= 4:
            selected_M_indices = [0, len(M_values)//3, 2*len(M_values)//3, -1]
            selected_M = [M_values[i] for i in selected_M_indices]
        else:
            selected_M = M_values
            
        for M in selected_M:
            if M in results:
                cya_values = []
                valid_alpha_values = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        cya_values.append(results[M]['alpha_results'][alpha_deg]['cya_iz_p'])
                        valid_alpha_values.append(alpha_deg)
                
                if cya_values:
                    axes[0, 1].plot(valid_alpha_values, cya_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[0, 1].set_xlabel('Угол атаки, °')
        axes[0, 1].set_ylabel('cya_iz_p')
        axes[0, 1].set_title('Коэффициент подъемной силы от угла атаки')
        if axes[0, 1].get_legend_handles_labels()[0]:
            axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. cxa_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            cxa_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cxa_values.append(results[M]['alpha_results'][alpha_deg]['cxa_iz_p'])
                else:
                    cxa_values.append(np.nan)
            
            if not all(np.isnan(cxa_values)):
                axes[1, 0].plot(M_values, cxa_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 0].set_xlabel('Число Маха, M')
        axes[1, 0].set_ylabel('cxa_iz_p')
        axes[1, 0].set_title('Коэффициент сопротивления от числа Маха')
        if axes[1, 0].get_legend_handles_labels()[0]:
            axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. cxa_iz_p от угла атаки для разных M
        for M in selected_M:
            if M in results:
                cxa_values = []
                valid_alpha_values = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        cxa_values.append(results[M]['alpha_results'][alpha_deg]['cxa_iz_p'])
                        valid_alpha_values.append(alpha_deg)
                
                if cxa_values:
                    axes[1, 1].plot(valid_alpha_values, cxa_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[1, 1].set_xlabel('Угол атаки, °')
        axes[1, 1].set_ylabel('cxa_iz_p')
        axes[1, 1].set_title('Коэффициент сопротивления от угла атаки')
        if axes[1, 1].get_legend_handles_labels()[0]:
            axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_focus_moment_graphs(self, results, M_values, alpha_deg_values):
        """Графики фокуса и моментов"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Положение фокуса и моменты рулей', fontsize=16)
        
        # 1. x_alpha_F от M
        x_focus_values = [results[M]['x_alpha_F'] for M in M_values]
        axes[0, 0].plot(M_values, x_focus_values, 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Число Маха, M')
        axes[0, 0].set_ylabel('x_alpha_F_iz_p')
        axes[0, 0].set_title('Положение фокуса по углу атаки от числа Маха')
        axes[0, 0].grid(True)
        
        # 2. x_d_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            x_d_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    x_d_values.append(results[M]['alpha_results'][alpha_deg]['x_d_iz_p'])
                else:
                    x_d_values.append(np.nan)
            
            if not all(np.isnan(x_d_values)):
                axes[0, 1].plot(M_values, x_d_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 1].set_xlabel('Число Маха, M')
        axes[0, 1].set_ylabel('x_d_iz_p')
        axes[0, 1].set_title('Центр давления от числа Маха')
        if axes[0, 1].get_legend_handles_labels()[0]:
            axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. mz_alpha_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            mz_alpha_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    mz_alpha_values.append(results[M]['alpha_results'][alpha_deg]['mz_alpha_iz_p'])
                else:
                    mz_alpha_values.append(np.nan)
            
            if not all(np.isnan(mz_alpha_values)):
                axes[1, 0].plot(M_values, mz_alpha_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 0].set_xlabel('Число Маха, M')
        axes[1, 0].set_ylabel('mz_alpha_iz_p')
        axes[1, 0].set_title('Производная момента от числа Маха')
        if axes[1, 0].get_legend_handles_labels()[0]:
            axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. mz_iz_p от угла атаки для разных M
        # Автоматический выбор характерных чисел Маха
        if len(M_values) >= 4:
            selected_M_indices = [0, len(M_values)//3, 2*len(M_values)//3, -1]
            selected_M = [M_values[i] for i in selected_M_indices]
        else:
            selected_M = M_values
            
        for M in selected_M:
            if M in results:
                mz_values = []
                valid_alpha_values = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        mz_values.append(results[M]['alpha_results'][alpha_deg]['mz_iz_p'])
                        valid_alpha_values.append(alpha_deg)
                
                if mz_values:
                    axes[1, 1].plot(valid_alpha_values, mz_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[1, 1].set_xlabel('Угол атаки, °')
        axes[1, 1].set_ylabel('mz_iz_p')
        axes[1, 1].set_title('Коэффициент продольного момента от угла атаки')
        if axes[1, 1].get_legend_handles_labels()[0]:
            axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self, results):
        """Вывод результатов расчетов"""
        print('\033[1;32;40m РЕЗУЛЬТАТЫ РАСЧЕТА АДХ РУЛЕЙ \033[0m')
        print('=' * 60)
        
        if not results:
            print("Нет результатов для вывода")
            return
            
        # Автоматический выбор характерных значений
        M_values = sorted(list(results.keys()))
        if not M_values:
            print("Нет данных по числам Маха")
            return
            
        # Выбор нескольких характерных чисел Маха
        if len(M_values) >= 4:
            sample_M = [M_values[0], M_values[len(M_values)//3], 
                       M_values[2*len(M_values)//3], M_values[-1]]
        else:
            sample_M = M_values
            
        # Выбор характерного угла атаки
        first_M = M_values[0]
        if 'alpha_results' in results[first_M] and results[first_M]['alpha_results']:
            alpha_deg_values = sorted(list(results[first_M]['alpha_results'].keys()))
            sample_alpha = alpha_deg_values[len(alpha_deg_values)//2] if alpha_deg_values else 0
        else:
            sample_alpha = 0

        for M in sample_M:
            if M in results and sample_alpha in results[M]['alpha_results']:
                m_results = results[M]
                alpha_results = m_results['alpha_results'][sample_alpha]
                
                print(f'\n\033[1;34;40m М = {M:.2f}, α = {sample_alpha}° \033[0m')
                print(f'Cy_alpha = {m_results["cy_alpha"]:.6f}')
                print(f'cy_iz_p = {alpha_results["cy_iz_p"]:.6f}')
                print(f'cx_iz_p = {alpha_results["cx_iz_p"]:.6f}')
                print(f'cya_iz_p = {alpha_results["cya_iz_p"]:.6f}')
                print(f'cxa_iz_p = {alpha_results["cxa_iz_p"]:.6f}')
                print(f'x_alpha_F_iz_p = {m_results["x_alpha_F"]:.6f}')
                print(f'x_d_iz_p = {alpha_results["x_d_iz_p"]:.6f}')
                print(f'mz_iz_p = {alpha_results["mz_iz_p"]:.6f}')


# Основная программа
if __name__ == "__main__":
    # Создание объектов
    geometry = RudderGeometry()
    calculator = RudderAerodynamicCalculator(geometry)
    
    # Выполнение расчетов
    print("Выполнение расчетов аэродинамических характеристик рулей...")
    results = calculator.run_calculations()
    
    # Вывод результатов
    calculator.print_results(results)
    
    # Построение графиков
    print("\nПостроение графиков...")
    calculator.plot_all_graphs(results)