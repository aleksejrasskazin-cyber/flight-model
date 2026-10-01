"Krilo_class"
import numpy as np
import math
import AeroBDSM
from math import *
import matplotlib.pyplot as plt
from Atmosphere import atmo
from mpl_toolkits.mplot3d import Axes3D
#Mcr_pr
class WingGeometry:
    """Класс для хранения геометрических параметров крыла"""
    
    def __init__(self):
        self.d_op = 0.15
        self.a = 0.075
        self.b = 0.170 #0.85
        self.aa = self.a/ self.b
        self.K_w = 1/(1-self.aa) #1 
        self.c = 15 * 0.001 #25 * 0.001
        self.xx_t = 0
        self.h = 0
        self.S = 0.09 # np.pi*0.139 #0.103 # площадь несущей поверхности
        self.X_05 = np.deg2rad(20.8)#np.deg2rad(15)
        self.l = 0.380#0.380 # размах крыла
        self.cc = self.c / self.b  # относительная толщина профиля 
        self.x_c = (self.b- self.a)/2
        self.xx_c = self.x_c/ self.b # 0.5  относительная координата линии максимальных толщин
        self.x_c = self.xx_c * self.b
        self.hh = self.h / self.b
        self.m = 0  # расстояние в долях хорды
        self.lambda_k =0.4 #0.57  # относительное удлинение консолей
        self.eta_k = 1.8 #2.84  # относительное сужение консолей
        self.poprav =  0.88 # поправка для критического числа Маха 
                           # Если профиль: ромб,4-х и 6 ти угольник или клиновидный то 0,90...0,88
                           # Если сверхзвуковой чечевицеобразный (образованный дугами окружностей или парабол), то 0,97...0,95
        self.l_f = 2.780 #0.4 + 0.1 + 0.4
        self.H = 100
        self.rho = atmo.rho(self.H)  # плотность воздуха на высоте H
        self.a_sound = atmo.a(self.H)  # скорость звука на высоте H
        
        # Расчетные параметры
        self.zeta_k = 1 / self.eta_k
        
        # Расчет хорд
        self.b_b = self.S / self.l * 2 / (1 + self.zeta_k)  # бортовая хорда#!!!!!!!!!!!!
        self.b1 =  self.b_b * self.zeta_k  # концевая хорда
        self.λ = self.l**2 / self.S  # удлинение несущей поверхности
        
        # Расчет углов стреловидности
        self.X_0 =  atan(tan(self.X_05) + 2/self.lambda_k * (1-self.zeta_k)/(1+self.zeta_k))
        self.X_1 = atan(tan(self.X_0) + 4/self.lambda_k * (1-self.zeta_k)/(1+self.zeta_k))
        self.X_c = atan(tan(self.X_0) - 4*self.xx_c/self.lambda_k * (1-self.zeta_k)/(1+self.zeta_k))
        self.X_m = atan(tan(self.X_0) - 4*self.m/self.lambda_k * (1-self.zeta_k)/(1+self.zeta_k))
        
        # Расчет параметров САХ
        self.bA =  4/3 * self.S/self.l * (1 - self.zeta_k/(1+self.zeta_k)**2)
        self.zA = self.l/6 * ((1+2*self.zeta_k)/(1+self.zeta_k))
        self.xA = self.zA * tan(self.X_0)
         
        self.z_цS = self.zA
        self.x_цS = self.xA + self.bA/2  # координата центра тяжести площади консоли
        self.x_cg_percent = 0.05  # 25% САХ#!!!!!!!!!!!!
        self.x_цS = self.xA + self.x_cg_percent * self.bA
        self.x_цS = self.xA + self.bA/2
        # Диапазоны расчетов
#       self.M_list = np.linspace(0.6, 2.0, 10).tolist()  # Более плотная сетка
        self.M_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0] # Это число Махов выбрано не случайно...
        # ...так как у нас для высотной поправки(в функциях c_xa_0 для фезюляжа, рулей и крыльев) в main конкретные числа Маха 0,5 1 1,5 и 2, а с linspace мы в них не попадаем
        #self.alpha_list = np.deg2rad(np.linspace(-40, 40, 500).tolist())#np.deg2rad(np.linspace(-15, 15, 7).tolist()) 
        alphas = np.linspace(-5, 5, 10)
        # Преобразуем в список, добавляем 0, сортируем и удаляем дубликаты
        self.alpha_list = np.deg2rad(sorted(set(alphas.tolist() + [0])))

class WingAerodynamicCalculator:
    """Класс для расчета аэродинамических характеристик крыла"""
    
    def __init__(self, geometry):
        self.geo = geometry
        
    def calculate_cy_alpha(self, M):
        """Расчет производной коэффициента нормальной силы"""
        cy_alpha = AeroBDSM.get_Cy_alpha_IsP(M, self.geo.lambda_k, self.geo.cc, 
                                            self.geo.X_05, self.geo.eta_k)
        return cy_alpha.Value
    
    def calculate_cy_iz_p(self, M, alpha, cy_alpha):
        """Расчет коэффициента нормальной силы"""
        A = AeroBDSM.get_A_IsP(M, self.geo.zeta_k, cy_alpha)
        return (cy_alpha * sin(alpha) * cos(alpha) + 
                A * sin(alpha)**2 * np.sign(alpha))
    
    def calculate_critical_mach(self, M, cy):
        """Расчет критического числа Маха"""
        Mcr_pr = AeroBDSM.get_Mcr_pr(self.geo.cc, abs(cy), self.geo.xx_c) * self.geo.poprav
        ΔMcr_λ = AeroBDSM.get_DELTA_Mcr0_lambda(Mcr_pr, self.geo.lambda_k)
        ΔMcr_χ = AeroBDSM.get_DELTA_Mcr0_chi(Mcr_pr, self.geo.X_m)
        return Mcr_pr + ΔMcr_λ + ΔMcr_χ
    
    def calculate_wave_drag(self, M, Mcr_iz_p):
        """Расчет коэффициента волнового сопротивления"""
        if M < Mcr_iz_p:
            return 0
        elif Mcr_iz_p < M < 1:
            a0 = AeroBDSM.get_Cxw_Rhomb_M1(1, self.geo.cc, self.geo.zeta_k, 
                                          self.geo.X_c, self.geo.lambda_k)
            a1 = a0 / (Mcr_iz_p**2 - 2*Mcr_iz_p + 1)
            a2 = -2 * a1 * Mcr_iz_p
            a3 = AeroBDSM.get_Cxw_Rhomb_M1(1, self.geo.cc, self.geo.zeta_k, 
                                         self.geo.X_c, self.geo.lambda_k) - a1*(1-2*Mcr_iz_p)
            return a1 * M**2 + a2 * M + a3
        else:
            cxa_w = AeroBDSM.get_Cxw_Rhomb_M1(M, self.geo.cc, self.geo.zeta_k, 
                                             self.geo.X_c, self.geo.lambda_k)
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
    
    def calculate_induced_drag(self, cy, alpha, c_fa=0):
        """Расчет индуктивного сопротивления"""
        return cy * sin(alpha) - c_fa * cos(alpha)
    
    def calculate_total_drag(self, c_xa_0_iz_profil_wing, cxa_i):
        """Расчет общего лобового сопротивления"""
        return c_xa_0_iz_profil_wing + cxa_i
    
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
        xxF = AeroBDSM.get_Coordinate_xxF_IsP(M, self.geo.lambda_k, self.geo.X_05, self.geo.zeta_k)
        return self.geo.xA + xxF * self.geo.bA
    
    def calculate_lateral_focus(self, M):
        """Расчет поперечной координаты фокуса"""
        zz_d = AeroBDSM.get_Coordinate_zz_d(M, self.geo.lambda_k, self.geo.X_05, self.geo.zeta_k)
        return self.geo.l / 2 * zz_d
    
    def calculate_center_of_pressure(self, M, alpha, cy_alpha, cy):
        """Расчет центра давления"""
        if alpha == 0:
            # При нулевом угле атаки центр давления совпадает с фокусом
            return self.calculate_focus_position(M)
        # if abs(cy) < 1e-10:#!!!!!!!!!!!!!!!!!!!!!!!
        #     return self.calculate_focus_position(M)
        else:
            A = AeroBDSM.get_A_IsP(M, self.geo.zeta_k, cy_alpha)
            x_focus = self.calculate_focus_position(M)
            
            numerator = (cy_alpha * sin(alpha) * cos(alpha) * x_focus + 
                        A * sin(alpha)**2 * np.sign(alpha) * self.geo.x_цS)
            
            if cy != 0:
                return numerator / cy
            else:
                return 0
    
    def calculate_pitching_moment_derivative(self, cy_alpha, x_cp, x_cg):
        """Расчет производной коэффициента продольного момента"""
        if self.geo.bA != 0:
            return cy_alpha * (x_cg - x_cp) / self.geo.bA
        else:
            return 0
    
    def calculate_pitching_moment(self, cy, x_cp, x_cg):
        """Расчет коэффициента продольного момента"""
        if self.geo.bA != 0:
            return cy * (x_cg - x_cp) / self.geo.bA
        else:
            return 0
    
    def run_calculations(self):
        """Основной метод выполнения расчетов"""
        results = {}
        
        for M in self.geo.M_list:
            results[M] = {}
            
            # Базовые характеристики
            results[M]['cy_alpha'] = self.calculate_cy_alpha(M)
            results[M]['cxa_prof'] = self.calculate_profile_drag(M)
            results[M]['cxa_don'] = self.calculate_base_drag(M)
            results[M]['x_alpha_F'] = self.calculate_focus_position(M)
            results[M]['z_alpha_F'] = self.calculate_lateral_focus(M)
            
            # Расчет для разных углов атаки
            results[M]['alpha_results'] = {}
            
            for alpha in self.geo.alpha_list:
                alpha_deg = round(np.rad2deg(alpha), 6)
                results[M]['alpha_results'][alpha_deg] = {}
                
                # Коэффициент нормальной силы
                cy = self.calculate_cy_iz_p(M, alpha, results[M]['cy_alpha'])
                results[M]['alpha_results'][alpha_deg]['cy_iz_p'] = cy
                
                # Критическое число Маха
                Mcr_iz_p = self.calculate_critical_mach(M, cy)
                results[M]['alpha_results'][alpha_deg]['Mcr_iz_p'] = Mcr_iz_p
                
                # Волновое сопротивление
                cxa_w = self.calculate_wave_drag(M, Mcr_iz_p)
                results[M]['alpha_results'][alpha_deg]['cxa_w'] = cxa_w
                
                # Лобовое сопротивление при нулевом угле атаки
                c_xa_0_iz_profil_wing = self.calculate_zero_angle_drag(M, cxa_w, 
                                                      results[M]['cxa_prof'], 
                                                      results[M]['cxa_don'])
                results[M]['alpha_results'][alpha_deg]['c_xa_0_iz_profil_wing'] = c_xa_0_iz_profil_wing

                
                # Индуктивное сопротивление
                cxa_i = self.calculate_induced_drag(cy, alpha)
                results[M]['alpha_results'][alpha_deg]['cxa_i'] = cxa_i
                
                # Общее лобовое сопротивление
                cxa_total = self.calculate_total_drag(c_xa_0_iz_profil_wing, cxa_i)
                results[M]['alpha_results'][alpha_deg]['cxa_iz_p'] = cxa_total
                
                # Коэффициент продольной силы
                cx_iz_p = self.calculate_cx_iz_p(cxa_total, cy, alpha)
                results[M]['alpha_results'][alpha_deg]['cx_iz_p'] = cx_iz_p
                
                # Коэффициент подъемной силы
                cya_iz_p = self.calculate_cya_iz_p(cy, cxa_total, alpha)
                results[M]['alpha_results'][alpha_deg]['cya_iz_p'] = cya_iz_p
                
                # Центр давления
                x_cp = self.calculate_center_of_pressure(M, alpha, 
                                                        results[M]['cy_alpha'], cy)
                results[M]['alpha_results'][alpha_deg]['x_d_iz_p'] = x_cp
                
                # Производная коэффициента продольного момента
                mz_alpha = self.calculate_pitching_moment_derivative(
                    results[M]['cy_alpha'], x_cp, self.geo.x_цS)
                results[M]['alpha_results'][alpha_deg]['mz_alpha_iz_p'] = mz_alpha
                
                # Коэффициент продольного момента
                mz = self.calculate_pitching_moment(cy, x_cp, self.geo.x_цS)
                results[M]['alpha_results'][alpha_deg]['mz_iz_p'] = mz
        
        return results
    
    def plot_all_graphs(self, results):
        """Построение графиков всех характеристик"""
        
        # Подготовка данных для графиков
        M_values = sorted(results.keys())
        
        # Проверяем, есть ли данные для построения графиков
        if not M_values:
            print("Нет данных для построения графиков")
            return
            
        # Автоматически берем углы атаки из результатов
        first_M = M_values[0]
        alpha_deg_values = sorted(results[first_M]['alpha_results'].keys())
        
        if not alpha_deg_values:
            print("Нет данных по углам атаки для построения графиков")
            return
        
        # Создание фигур
        self.plot_cy_cx_graphs(results, M_values, alpha_deg_values)
        self.plot_cya_cxa_graphs(results, M_values, alpha_deg_values)
        self.plot_focus_moment_graphs(results, M_values, alpha_deg_values)
        self.plot_polar_diagram(results, M_values, alpha_deg_values)
        
    
    def plot_cy_cx_graphs(self, results, M_values, alpha_deg_values):
        """Графики cy_iz_p и cx_iz_p"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Коэффициенты нормальной и продольной силы крыла', fontsize=16)
        
        # 1. cy_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            cy_values = []
            valid_M = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cy_values.append(results[M]['alpha_results'][alpha_deg]['cy_iz_p'])
                    valid_M.append(M)
            if cy_values:  # Проверяем, что есть данные для построения
                axes[0, 0].plot(valid_M, cy_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 0].set_xlabel('Число Маха, M')
        axes[0, 0].set_ylabel('cy_iz_p')
        axes[0, 0].set_title('Коэффициент нормальной силы от числа Маха')
        if alpha_deg_values:  # Добавляем легенду только если есть данные
            axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. cy_iz_p от угла атаки для разных M
        # Автоматически выбираем несколько значений M для отображения
        if len(M_values) > 4:
            selected_M_indices = np.linspace(0, len(M_values)-1, 4, dtype=int)
            selected_M = [M_values[i] for i in selected_M_indices]
        else:
            selected_M = M_values
            
        for M in selected_M:
            cy_values = []
            valid_alpha = []
            for alpha_deg in alpha_deg_values:
                if alpha_deg in results[M]['alpha_results']:
                    cy_values.append(results[M]['alpha_results'][alpha_deg]['cy_iz_p'])
                    valid_alpha.append(alpha_deg)
            if cy_values:  # Проверяем, что есть данные для построения
                axes[0, 1].plot(valid_alpha, cy_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[0, 1].set_xlabel('Угол атаки, °')
        axes[0, 1].set_ylabel('cy_iz_p')
        axes[0, 1].set_title('Коэффициент нормальной силы от угла атаки')
        if selected_M:  # Добавляем легенду только если есть данные
            axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. cx_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            cx_values = []
            valid_M = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cx_values.append(results[M]['alpha_results'][alpha_deg]['cx_iz_p'])
                    valid_M.append(M)
            if cx_values:  # Проверяем, что есть данные для построения
                axes[1, 0].plot(valid_M, cx_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 0].set_xlabel('Число Маха, M')
        axes[1, 0].set_ylabel('cx_iz_p')
        axes[1, 0].set_title('Коэффициент продольной силы от числа Маха')
        if alpha_deg_values:  # Добавляем легенду только если есть данные
            axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. cx_iz_p от угла атаки для разных M
        for M in selected_M:
            cx_values = []
            valid_alpha = []
            for alpha_deg in alpha_deg_values:
                if alpha_deg in results[M]['alpha_results']:
                    cx_values.append(results[M]['alpha_results'][alpha_deg]['cx_iz_p'])
                    valid_alpha.append(alpha_deg)
            if cx_values:  # Проверяем, что есть данные для построения
                axes[1, 1].plot(valid_alpha, cx_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[1, 1].set_xlabel('Угол атаки, °')
        axes[1, 1].set_ylabel('cx_iz_p')
        axes[1, 1].set_title('Коэффициент продольной силы от угла атаки')
        if selected_M:  # Добавляем легенду только если есть данные
            axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cya_cxa_graphs(self, results, M_values, alpha_deg_values):
        """Графики cya_iz_p и cxa_iz_p"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Коэффициенты подъемной силы и сопротивления крыла', fontsize=16)
        
        # 1. cya_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            cya_values = []
            valid_M = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cya_values.append(results[M]['alpha_results'][alpha_deg]['cya_iz_p'])
                    valid_M.append(M)
            if cya_values:  # Проверяем, что есть данные для построения
                axes[0, 0].plot(valid_M, cya_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 0].set_xlabel('Число Маха, M')
        axes[0, 0].set_ylabel('cya_iz_p')
        axes[0, 0].set_title('Коэффициент подъемной силы от числа Маха')
        if alpha_deg_values:  # Добавляем легенду только если есть данные
            axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. cya_iz_p от угла атаки для разных M
        # Автоматически выбираем несколько значений M для отображения
        if len(M_values) > 4:
            selected_M_indices = np.linspace(0, len(M_values)-1, 4, dtype=int)
            selected_M = [M_values[i] for i in selected_M_indices]
        else:
            selected_M = M_values
            
        for M in selected_M:
            cya_values = []
            valid_alpha = []
            for alpha_deg in alpha_deg_values:
                if alpha_deg in results[M]['alpha_results']:
                    cya_values.append(results[M]['alpha_results'][alpha_deg]['cya_iz_p'])
                    valid_alpha.append(alpha_deg)
            if cya_values:  # Проверяем, что есть данные для построения
                axes[0, 1].plot(valid_alpha, cya_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[0, 1].set_xlabel('Угол атаки, °')
        axes[0, 1].set_ylabel('cya_iz_p')
        axes[0, 1].set_title('Коэффициент подъемной силы от угла атаки')
        if selected_M:  # Добавляем легенду только если есть данные
            axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. cxa_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            cxa_values = []
            valid_M = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cxa_values.append(results[M]['alpha_results'][alpha_deg]['cxa_iz_p'])
                    valid_M.append(M)
            if cxa_values:  # Проверяем, что есть данные для построения
                axes[1, 0].plot(valid_M, cxa_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 0].set_xlabel('Число Маха, M')
        axes[1, 0].set_ylabel('cxa_iz_p')
        axes[1, 0].set_title('Коэффициент сопротивления от числа Маха')
        if alpha_deg_values:  # Добавляем легенду только если есть данные
            axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. cxa_iz_p от угла атаки для разных M
        for M in selected_M:
            cxa_values = []
            valid_alpha = []
            for alpha_deg in alpha_deg_values:
                if alpha_deg in results[M]['alpha_results']:
                    cxa_values.append(results[M]['alpha_results'][alpha_deg]['cxa_iz_p'])
                    valid_alpha.append(alpha_deg)
            if cxa_values:  # Проверяем, что есть данные для построения
                axes[1, 1].plot(valid_alpha, cxa_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[1, 1].set_xlabel('Угол атаки, °')
        axes[1, 1].set_ylabel('cxa_iz_p')
        axes[1, 1].set_title('Коэффициент сопротивления от угла атаки')
        if selected_M:  # Добавляем легенду только если есть данные
            axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_focus_moment_graphs(self, results, M_values, alpha_deg_values):
        """Графики фокуса и моментов"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Положение фокуса и моменты крыла', fontsize=16)
        
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
            valid_M = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    x_d_values.append(results[M]['alpha_results'][alpha_deg]['x_d_iz_p'])
                    valid_M.append(M)
            if x_d_values:  # Проверяем, что есть данные для построения
                axes[0, 1].plot(valid_M, x_d_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 1].set_xlabel('Число Маха, M')
        axes[0, 1].set_ylabel('x_d_iz_p')
        axes[0, 1].set_title('Центр давления от числа Маха')
        if alpha_deg_values:  # Добавляем легенду только если есть данные
            axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. mz_alpha_iz_p от M для разных углов атаки
        for alpha_deg in alpha_deg_values:
            mz_alpha_values = []
            valid_M = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    mz_alpha_values.append(results[M]['alpha_results'][alpha_deg]['mz_alpha_iz_p'])
                    valid_M.append(M)
            if mz_alpha_values:  # Проверяем, что есть данные для построения
                axes[1, 0].plot(valid_M, mz_alpha_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 0].set_xlabel('Число Маха, M')
        axes[1, 0].set_ylabel('mz_alpha_iz_p')
        axes[1, 0].set_title('Производная момента от числа Маха')
        if alpha_deg_values:  # Добавляем легенду только если есть данные
            axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. mz_iz_p от угла атаки для разных M
        # Автоматически выбираем несколько значений M для отображения
        if len(M_values) > 4:
            selected_M_indices = np.linspace(0, len(M_values)-1, 4, dtype=int)
            selected_M = [M_values[i] for i in selected_M_indices]
        else:
            selected_M = M_values
            
        for M in selected_M:
            mz_values = []
            valid_alpha = []
            for alpha_deg in alpha_deg_values:
                if alpha_deg in results[M]['alpha_results']:
                    mz_values.append(results[M]['alpha_results'][alpha_deg]['mz_iz_p'])
                    valid_alpha.append(alpha_deg)
            if mz_values:  # Проверяем, что есть данные для построения
                axes[1, 1].plot(valid_alpha, mz_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        axes[1, 1].set_xlabel('Угол атаки, °')
        axes[1, 1].set_ylabel('mz_iz_p')
        axes[1, 1].set_title('Коэффициент продольного момента от угла атаки')
        if selected_M:  # Добавляем легенду только если есть данные
            axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_polar_diagram(self, results, M_values, alpha_deg_values):
        """Построение поляры крыла"""
        plt.figure(figsize=(10, 8))
        
        # Автоматически выбираем несколько значений M для отображения
        if len(M_values) > 4:
            selected_M_indices = np.linspace(0, len(M_values)-1, 4, dtype=int)
            selected_M = [M_values[i] for i in selected_M_indices]
        else:
            selected_M = M_values
        
        for M in selected_M:
            cxa_values = []
            cya_values = []
            for alpha_deg in alpha_deg_values:
                if alpha_deg in results[M]['alpha_results']:
                    cxa_values.append(results[M]['alpha_results'][alpha_deg]['cxa_iz_p'])
                    cya_values.append(results[M]['alpha_results'][alpha_deg]['cya_iz_p'])
            
            if cxa_values and cya_values:  # Проверяем, что есть данные для построения
                plt.plot(cxa_values, cya_values, 'o-', label=f'M = {M:.2f}', linewidth=2)
        
        plt.xlabel('cxa_iz_p')
        plt.ylabel('cya_iz_p')
        plt.title('Поляра крыла')
        if selected_M:  # Добавляем легенду только если есть данные
            plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    
    def print_results(self, results):
        """Вывод результатов расчетов"""
        print('\033[1;32;40m РЕЗУЛЬТАТЫ РАСЧЕТА АДХ КРЫЛА \033[0m')
        print('=' * 60)
        
        # Вывод для нескольких характерных значений
        if not results:
            print("Нет результатов для вывода")
            return
            
        M_values = sorted(results.keys())
        
        # Автоматически выбираем значения для вывода
        if len(M_values) > 4:
            sample_M_indices = np.linspace(0, len(M_values)-1, 4, dtype=int)
            sample_M = [M_values[i] for i in sample_M_indices]
        else:
            sample_M = M_values
            
        # Берем первый доступный угол атаки
        first_M = M_values[0]
        alpha_deg_values = sorted(results[first_M]['alpha_results'].keys())
        if alpha_deg_values:
            sample_alpha = alpha_deg_values[len(alpha_deg_values)//2]  # берем средний угол
        else:
            sample_alpha = 0

        for M in sample_M:
            if M in results:
                m_results = results[M]
                if sample_alpha in m_results['alpha_results']:
                    alpha_results = m_results['alpha_results'][sample_alpha]
                    
                    print(f'\n\033[1;34;40m М = {M:.1f}, α = {sample_alpha}° \033[0m')
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
    geometry = WingGeometry()
    calculator = WingAerodynamicCalculator(geometry)
    
    # Выполнение расчетов
    print("Выполнение расчетов аэродинамических характеристик крыла...")
    results = calculator.run_calculations()
    
    # Вывод результатов
    calculator.print_results(results)
    
    # Построение графиков
    print("\nПостроение графиков...")
    calculator.plot_all_graphs(results)
