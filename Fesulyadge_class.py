from Krilo_class import WingGeometry, WingAerodynamicCalculator as WingCalculator
import numpy as np
import math
import AeroBDSM
from math import *
import matplotlib.pyplot as plt
from Atmosphere import atmo
from scipy.integrate import quad
"Fesulyadge_class"
class FuselageGeometry:
    """Класс для хранения геометрических параметров фюзеляжа"""
     
    def __init__(self):
        # self.R = 0.8
        # self.r_zatup = 0.02
        # self.d_m = 0.240
        # self.r_ = 2 * self.r_zatup / self.d_m
        # self.l_nos = 0.4

        # self.l_cil = 0.2
        # self.S_m = (pi * self.d_m**2) / 4
        
         
        # self.l_don = 0.8 # Cxa_0_p_kor появляется при всех прочих 0.06 (то есть угол наклона образующей конуса >20 и течение отрывное)
        # self.d_k = 0.030
        # self.d_a = 0.95 * self.d_k
        # self.lambda_kor = self.l_don / self.d_m
        # self.eta_kor = self.d_k / self.d_m
        # self.S_dn = (pi * self.d_k**2) / 4
        # self.L = self.l_nos+self.l_cil+self.l_don
        
        # self.a = a = self.l_nos - self.r_zatup
        # self.phi = np.arccos((np.square(self.a) - np.square((self.d_m/2) - self.r_zatup))/(np.square(self.a+-np.square((self.d_m/2) - self.r_zatup))))
        # self.x_1 = self.r_zatup * (1 - sin(self.phi))
        # self.l_f = self.l_nos + self.l_cil + self.l_don
        # self.lambda_nos = self.l_nos / self.d_m 
        # self.lambda_cil = self.l_cil / self.d_m
        # self.lambda_fesulage = self.l_f / self.d_m
        
        # self.lambda_nos_ogi = (2 * self.lambda_nos - self.r_) / (2 * math.sqrt(1 - self.r_))
        # self.Teta = math.atan(np.sqrt(1 - self.r_) / self.lambda_nos_ogi)
        
        # self.engine = 0 # (0 - двигатель не раб-ет, 1 - двиг.раб-ет) это важно для к-та донного сопроотивления      
        # self.wing_geo = WingGeometry()
        # self.wing_calc = WingCalculator(self.wing_geo)
        # self.H = self.wing_geo.H
        # self.dvigatel = 1  # 0-выкл, 1-вкл
        # self.xx_t = 0
        # # Параметры атмосферы для расчета сил
        # self.rho = atmo.rho(self.H)  # плотность воздуха на высоте H
        # self.a_sound = atmo.a(self.H)  # скорость звука на высоте H
        self.R = 1.579
        self.r_zatup = 0.003 #0.02
        self.d_m = 0.122 #0.240
        self.r_ = 2 * self.r_zatup / self.d_m
        self.l_nos = 0.411#0.4

        self.l_cil = 2.5818 # 0.2
        self.S_m = (pi * self.d_m**2) / 4
        
         
        self.l_don = 0.004 #0.8 # Cxa_0_p_kor появляется при всех прочих 0.06 (то есть угол наклона образующей конуса >20 и течение отрывное)
        self.d_k = 0.1219 #0.030
        self.d_a = 0.95 * self.d_k
        self.lambda_kor = self.l_don / self.d_m
        self.eta_kor = self.d_k / self.d_m
        self.S_dn = (pi * self.d_k**2) / 4
        self.L = self.l_nos+self.l_cil+self.l_don
        
        self.a = a = self.l_nos - self.r_zatup
        self.phi = np.arccos((np.square(self.a) - np.square((self.d_m/2) - self.r_zatup))/(np.square(self.a) + np.square((self.d_m/2) - self.r_zatup)))
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.x_1 = self.r_zatup * (1 - sin(self.phi))
        self.l_f = self.l_nos + self.l_cil + self.l_don
        self.lambda_nos = self.l_nos / self.d_m 
        self.lambda_cil = self.l_cil / self.d_m
        self.lambda_fesulage = self.l_f / self.d_m
        
        self.lambda_nos_ogi = (2 * self.lambda_nos - self.r_) / (2 * math.sqrt(1 - self.r_))
        self.Teta = math.atan(np.sqrt(1 - self.r_) / self.lambda_nos_ogi)
        
        self.engine = 1 # (0 - двигатель не раб-ет, 1 - двиг.раб-ет) это важно для к-та донного сопроотивления      
        self.wing_geo = WingGeometry()
        self.wing_calc = WingCalculator(self.wing_geo)
        self.H = self.wing_geo.H
        self.dvigatel = 1  # 0-выкл, 1-вкл
        self.xx_t = 0
        # Параметры атмосферы для расчета сил
        self.rho = atmo.rho(self.H)  # плотность воздуха на высоте H
        self.a_sound = atmo.a(self.H)  # скорость звука на высоте H
class AerodynamicCalculator:
    """Класс для расчета аэродинамических характеристик"""
     
    def __init__(self, geometry):
        self.geo = geometry
        # Диапазоны расчетов
        self.wing_geo = WingGeometry()
        self.wing_calc = WingCalculator(self.wing_geo)
        
        self.M_list = self.wing_geo.M_list
        self.alpha_list = self.wing_geo.alpha_list # Углы атаки в радианах 
    
    def detect_nose_type(self):
        """
        определяет тип носовой части
        """
        # Здесь должен быть код анализа вашего класса
        # Для примера - определяем по комментариям или структуре
        
        # conical_with_sphere - коническая со сферическим затуплением
        # ogive_with_sphere - оживальная со сферическим затуплением
        # simple_conical - простая коническая
        # simple_ogive - простая оживальная
        # spherical - сферическая
        # none - отсутствующая носовая часть
        
        
        nose_type = "ogive_with_sphere"  # или "ogive_with_sphere"
        
        return nose_type    
    
    def calculate_cy_alpha_nos_cil(self, M):
        """Расчет производной коэффициента нормальной силы нос+цил"""
        nose_type = self.detect_nose_type()
        if nose_type == "conical_with_sphere":
            # Существующий расчет для конической носовой части со сферическим затуплением
            cy_alpha_con = AeroBDSM.get_Cy_alpha_ConCil(M, self.geo.lambda_nos, self.geo.lambda_cil)
            cy_alpha_sph = AeroBDSM.get_Cy_alpha_SphCil(M, 0.5, self.geo.lambda_cil)
            
            return cy_alpha_con * (1 - self.geo.r_**2) + cy_alpha_sph * self.geo.r_**2
        
        elif nose_type == "ogive_with_sphere":
            """Расчет производной коэффициента нормальной силы нос+цил"""
            cy_alpha_ogi = AeroBDSM.get_Cy_alpha_OgiCil(M, self.geo.lambda_nos, self.geo.lambda_cil).Value
            cy_alpha_sph = AeroBDSM.get_Cy_alpha_SphCil(M, 0.5, self.geo.lambda_cil).Value
            
            return cy_alpha_ogi * (1 - self.geo.r_**2) + cy_alpha_sph * self.geo.r_**2
        
        elif nose_type == "simple_conical":
            # Простая коническая носовая часть
            return AeroBDSM.get_Cy_alpha_ConCil(M, self.geo.lambda_nos, self.geo.lambda_cil).Value
        
        elif nose_type == "simple_ogive":
            # Простая оживальная носовая часть
            return AeroBDSM.get_Cy_alpha_OgiCil(M, self.geo.lambda_nos, self.geo.lambda_cil).Value
        
        elif nose_type == "spherical":
            # Эллиптическая носовая часть
            return AeroBDSM.get_Cy_alpha_SphCil(M, self.geo.lambda_nos, self.geo.lambda_cil).Value
        
        elif nose_type == "none":
            # Отсутствующая носовая часть
            return AeroBDSM.get_Cy_alpha_SphCil(M, 0, self.geo.lambda_cil).Value
        
        else:
            raise ValueError(f"Неизвестный тип носовой части: {nose_type}")
            
    def calculate_cy_alpha_don(self):  
        """Расчет производной коэффициента нормальной силы дна"""
        return  - 0.4 * (1 - self.geo.eta_kor**2)
    def calculate_ka(self, M, alpha):
        """Расчет поправочного коэффициента K_a"""
        return 1 - 0.45 * (1 - 2.7**(-0.06 * M**2) * (1 - 2.7**(-0.12 * abs(alpha))))
    def calc_s_bok_nos(self):
        """
        Вычисление общей площади продольного сечения носовой части
        по формулам из изображения:
            
        """
        def safe_sqrt(radicand):
            """Безопасное вычисление квадратного корня"""
            if radicand < 0:
                return 0.0
            return np.sqrt(radicand)
        
        # 1. СФЕРИЧЕСКАЯ ЧАСТЬ: S_сф = 2∫√(r_сф² - (x - r_сф)²)dx от 0 до x_1
        def spherical_integrand(x):
            radicand = self.geo.r_zatup**2 - (x - self.geo.r_zatup)**2
            return 2 * safe_sqrt(radicand)
        
        # 2. ОЖИВАЛЬНАЯ ЧАСТЬ: S_ож = 2∫[√(R² - (x - L_нос)²) - R + d_M/2]dx от x_1 до L_нос
        def ogival_integrand(x):
            radicand = self.geo.R**2 - (x - self.geo.l_nos)**2
            sqrt_part = safe_sqrt(radicand)
            return 2 * (sqrt_part - self.geo.R + self.geo.d_m/2)
        
        # Вычисление интегралов
        S_sf, error_zatup = quad(spherical_integrand, 0, self.geo.x_1)
        S_og, error_og = quad(ogival_integrand, self.geo.x_1, self.geo.l_nos)
        # Общая площадь
        S_total = S_sf + S_og
        
        return S_total
    # def calculate_cy_f(self, M, alpha, cy_alpha):#!!!!!!!!!!!!!!!!!
    #     """Расчет коэффициента нормальной силы"""
    #     S_bok_nos = self.calc_s_bok_nos()
        
    #     S_bok_ratio = (S_bok_nos + pi * self.geo.d_m * self.geo.l_cil + pi * np.sqrt((self.geo.l_don**2) + ((self.geo.d_m - self.geo.d_k) / 2)**2) * ((self.geo.d_m + self.geo.d_k) / 2)) / (pi * self.geo.d_m**2)
        
    #     cx_cil_n = AeroBDSM.get_Cx_Cil_N(M, abs(alpha)).Value
        
    #     ka = self.calculate_ka(M, alpha)
        
    #     return (cy_alpha * ka * sin(alpha) * cos(alpha) + 4 * S_bok_ratio * cx_cil_n * sin(alpha)**2 * np.sign(alpha))
    
    def calculate_cy_f(self, M, alpha, cy_alpha):
        """Расчет коэффициента нормальной силы"""
        S_bok_nos = self.calc_s_bok_nos()
        
        # Полная площадь боковой проекции (м²)
        S_bok_full = (S_bok_nos + 
                      pi * self.geo.d_m * self.geo.l_cil + 
                      pi * np.sqrt((self.geo.l_don**2) + ((self.geo.d_m - self.geo.d_k) / 2)**2) * 
                      ((self.geo.d_m + self.geo.d_k) / 2))
        
        # Отношение площади боковой проекции к площади миделя
        S_bok_ratio = S_bok_full / self.geo.S_m
        
        cx_cil_n = AeroBDSM.get_Cx_Cil_N(M, abs(alpha)).Value
        ka = self.calculate_ka(M, alpha)
        
        # По методике: cy = cy_alpha * ka * sin(alpha) * cos(alpha) + 
        #               (4 * S_bok / (pi * d_m**2)) * cx_cil_n * sin^2(alpha) * sign(alpha)
        # Но S_bok_ratio у нас = S_bok_full / S_m = (4 * S_bok_full) / (pi * d_m**2)
        # Поэтому 4 * S_bok / (pi * d_m**2) = S_bok_ratio
        
        return (cy_alpha * ka * sin(alpha) * cos(alpha) + 
                S_bok_ratio * cx_cil_n * sin(alpha)**2 * np.sign(alpha))

    def calculate_cx_a_0_p_predvar(self, M):
        """Расчет коэффициента сопротивления давления (предварительный)"""

        cx_ogi = AeroBDSM.get_Cxnos_Par(M, self.geo.lambda_nos_ogi)
        cx_ell = AeroBDSM.get_Cxnos_Ell(M, 0.5)

        return (cx_ogi * (1 - (self.geo.r_ * math.cos(self.geo.Teta))**2 * 
         (3.1 - 1.4 * self.geo.r_ * math.cos(self.geo.Teta) - 0.7 * (self.geo.r_**2) * math.cos(self.geo.Teta)**2)) + 
         cx_ell * (self.geo.r_**2))
    
    def calculate_boundary_layer_correction(self, M):
        """Учет пограничного слоя методом эквивалентного конуса с использованием метода половинного деления"""
        
        # Предварительный расчет коэффициента сопротивления для исходной носовой части
        cx_nos_predvar = self.calculate_cx_a_0_p_predvar(M)
        
        # Функция для поиска корня (разность между коэффициентами сопротивления)
        def f(lambda_ekv):
            cx_ekv_con = AeroBDSM.get_Cxnos_Con(M, lambda_ekv)
            return cx_ekv_con - cx_nos_predvar
        
        # Метод половинного деления для нахождения lambda_ekv_con
        def bisection_method(func, a, b, tol=1e-6, max_iter=100):
            """Метод половинного деления для нахождения корня уравнения"""
            if func(a) * func(b) >= 0:
                # Если на границах интервала функция одного знака, расширяем интервал
                if abs(func(a)) < abs(func(b)):
                    return a
                else:
                    return b
            
            iter_count = 0
            while (b - a) / 2.0 > tol and iter_count < max_iter:
                c = (a + b) / 2.0
                if func(c) == 0:
                    return c
                elif func(a) * func(c) < 0:
                    b = c
                else:
                    a = c
                iter_count += 1
        
            return (a + b) / 2.0
        
        # Начальный интервал для поиска (удлинение конуса обычно в диапазоне 0.5-3.0)
        lambda_min = 0.5
        lambda_max = 3.0
        
        # Поиск эквивалентного удлинения конуса
        lambda_ekv_con = bisection_method(f, lambda_min, lambda_max,tol=1e-6, max_iter=100)
        
        # Дальнейшие расчеты по методике
        l_ekv_con = lambda_ekv_con * self.geo.d_m
        Teta_ekv_con = atan(self.geo.d_m / (2 * l_ekv_con))
        
        Ms_con = AeroBDSM.get_Ms_Con(M, Teta_ekv_con)
        Re_con = (l_ekv_con * M * atmo.a(self.geo.H)) / atmo.nu(self.geo.H)
        
        Teta_dop = (0.037 * (1 + 0.4 * Ms_con + 0.147 * Ms_con**2 - 0.006 * Ms_con**3) / 
                   Re_con**0.2)
        
        Teta_M_con = Teta_ekv_con + Teta_dop
        lambda_nos_M_con = 1 / (2 * tan(Teta_M_con))
        
        cx_nos_mcon = AeroBDSM.get_Cxnos_Con(Ms_con, lambda_nos_M_con)
        cx_ekv_con = AeroBDSM.get_Cxnos_Con(M, lambda_ekv_con)
        
        # Отладочная информация
#        print(f"M = {M:.2f}: lambda_ekv_con = {lambda_ekv_con:.6f}, correction = {cx_nos_mcon/cx_ekv_con:.6f}")
    
        return lambda_ekv_con, cx_nos_mcon / cx_ekv_con
    # конец расчёта эквивалентного конуса
    def calculate_cx_a_0_p_nos_cil(self, M):
        """Расчет коэффициента сопротивления давления с учетом пограничного слоя нос+цил"""
        cx_predvar = self.calculate_cx_a_0_p_predvar(M)
        lambda_ekv_con, correction = self.calculate_boundary_layer_correction(M)
        
        return cx_predvar * correction
    def calculate_cx_a_0_p_kor(self, M):
        """Расчет коэффициента сопротивления давления с учетом пограничного слоя конусной кормы"""
        Teta_kor = math.atan((self.geo.d_m - self.geo.d_k)/ (2 * self.geo.l_don)) # Угол между образующей и осью ракеты
        
        if Teta_kor >= np.deg2rad(20):
            cx_a_0_p_kor = AeroBDSM.get_Cxcor_Con(M, self.geo.lambda_kor, self.geo.eta_kor).Value  
        else:
            cx_a_0_p_kor = 0
           
        return cx_a_0_p_kor
    
    
    def calculate_cx_a_0_f(self, M):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """Расчет коэффициента сопротивления трения"""
        S_bok_nos = self.calc_s_bok_nos()
        # F_fuselage = (S_bok_nos + pi * self.geo.d_m * self.geo.l_cil + pi * np.sqrt((self.geo.l_don**2) + ((self.geo.d_m - self.geo.d_k) / 2)**2) * ((self.geo.d_m + self.geo.d_k) / 2)) / ((pi * self.geo.d_m**2))
        Re = ((self.geo.l_nos + self.geo.l_cil) * M * atmo.a(self.geo.H)) / atmo.nu(self.geo.H)
        
        Cf_M0 = AeroBDSM.get_Cf_M0(Re, self.geo.xx_t)
        Sigma_M = AeroBDSM.get_Sigma_M(M, self.geo.xx_t)
        Cf = Cf_M0 * Sigma_M
        
        S_full = (S_bok_nos +                     # площадь боковой проекции носовой части (м²)
          pi * self.geo.d_m * self.geo.l_cil +  # площадь цилиндра (πd * L)
          pi * sqrt((self.geo.l_don**2) + ((self.geo.d_m - self.geo.d_k)/2)**2) * 
          ((self.geo.d_m + self.geo.d_k)/2))    # площадь конуса (π * R_ср * образующая)

        S_m = pi * self.geo.d_m**2 / 4

        F_fuselage = S_full / S_m
        # return Cf * (F_fuselage / self.geo.S_m)
        return Cf * (F_fuselage)
    def calculate_cx_a_don(self, M):
        """Расчет коэффициента донного сопротивления"""
        Re = (self.geo.l_f * M * atmo.a(self.geo.H)) / atmo.nu(self.geo.H)
        Re_06 = (self.geo.l_f * 0.6 * atmo.a(self.geo.H)) / atmo.nu(self.geo.H)
        
        return AeroBDSM.get_Cxdon_Cil(M, 0, Re_06, Re, self.geo.xx_t, self.geo.lambda_fesulage).Value
    
    def calculate_coef_cx_a_don(self, M):
        Sigma_eta = AeroBDSM.get_Sigma_eta(M, self.geo.lambda_kor, self.geo.eta_kor).Value
        
        if self.geo.engine == 0:
            return Sigma_eta * self.geo.eta_kor**2
        else:
            return Sigma_eta * (self.geo.eta_kor**2 - (self.geo.d_a**2)/(self.geo.d_m**2))
    
 
    
    def calc_V_nos(self):
        """Расчёт объёма носовой части"""
    
        def safe_sqrt(radicand):
            """Безопасное вычисление квадратного корня"""
            if radicand < 0:
                return 0.0
            return np.sqrt(radicand)
        
        # 1. ОБЪЁМ СФЕРИЧЕСКОЙ ЧАСТИ: V_сф = π∫[r_сф² - (x - r_сф)²]dx от 0 до x_1
        def spherical_volume_integrand(x):
            return math.pi * (self.geo.r_zatup**2 - (x - self.geo.r_zatup)**2)
        
        # 2. ОБЪЁМ ОЖИВАЛЬНОЙ ЧАСТИ: 
        # V_ож = π∫[√(R² - (x - L_нос)²) - R + d_m/2]²dx от x_1 до L_нос
        def ogival_volume_integrand(x):
            sqrt_part = safe_sqrt(self.geo.R**2 - (x - self.geo.l_nos)**2)
            radius_at_x = sqrt_part - self.geo.R + self.geo.d_m / 2
            return math.pi * radius_at_x**2
        
        # Вычисление интегралов
        try:
            V_sf, error_zatup = quad(spherical_volume_integrand, 0, self.geo.x_1)
#            print(f"Объем сферической части: V_sf = {V_sf:.6f} м³")
        except Exception as e:
            V_sf = 0
        
        try:
            V_og, error_og = quad(ogival_volume_integrand, self.geo.x_1, self.geo.l_nos)
#            print(f"Объем оживальной части: V_og = {V_og:.6f} м³")
        except Exception as e:
            V_og = 0
        
        # 3. ОБЩИЙ ОБЪЁМ НОСОВОЙ ЧАСТИ
        V_nos_total = V_sf + V_og
        
                
        # Сохраняем промежуточные значения
        self.V_sf_calculated = V_sf
        self.V_og_calculated = V_og
        
        return V_nos_total
    
    def calc_V_don(self):
        """Расчёт объёма кормовой части"""
        R1 = self.geo.d_m / 2     #  (мидель)
        R2 = self.geo.d_k / 2     #  (дно)
        # Объем усеченного конуса
        V_kor = (math.pi * self.geo.l_don / 3) * (R1**2 + R1 * R2 + R2**2)
        
        return V_kor

# =============================================================================
# Расчёт x_alpha_F для всего фезюляжа (координаты фокуса)
# =============================================================================
    def calculate_x_alpha_F_noscil(self, M):
       """Расчет координаты фокуса нос+цил"""
       sigma_M = AeroBDSM.get_xi_M(M, self.geo.lambda_nos, self.geo.lambda_cil)
       delta_x_F = sigma_M * self.geo.l_nos
       V_nos = self.calc_V_nos()
       
       return self.geo.l_nos - 4 * V_nos / (pi * self.geo.d_m**2) + delta_x_F
    def calculate_x_alpha_F_don(self):
        """Расчет координаты фокуса кормы"""
        return self.geo.l_don - (self.geo.S_m * self.geo.l_don + self.calc_V_don()) / (self.geo.S_m - self.geo.S_dn)
    def calculate_x_alpha_F(self, M):
        """Расчет координаты фокуса всего ЛА (в расчёте на голый фезюляж)"""
        
        cy_alpha_nos_cil = self.calculate_cy_alpha_nos_cil(M)  # c_{y пос+цил}^{α}
        cy_alpha_kor = self.calculate_cy_alpha_don()          # c_{y кор}^{α}
        
        x_F_nos_cil_alpha = self.calculate_x_alpha_F_noscil(M) # x_{F пос+цил}^{α}
        x_F_kor_alpha = self.calculate_x_alpha_F_don()         # x_{F кор}^{α}
        
        # Суммарная производная коэффициента подъемной силы фюзеляжа
        cy_alpha_iz_f = cy_alpha_nos_cil + cy_alpha_kor        # c_{y из.ф}^{α}
        
        # Длина носовой+цилиндрической части
        l_nos_cil = self.geo.l_nos + self.geo.l_cil
        
        # Расчет по формуле
        numerator = (x_F_nos_cil_alpha * cy_alpha_nos_cil + 
                    (x_F_kor_alpha + l_nos_cil) * cy_alpha_kor)
        
        if cy_alpha_iz_f != 0:
            x_F_iz_f_alpha = numerator / cy_alpha_iz_f
        else:
            x_F_iz_f_alpha = 0
        
        return x_F_iz_f_alpha
    def calculate_delta_cx(self, M, alpha):
        """Расчет дополнительной продольной силы при alpha ≠ 0"""
        sigma_cx_i_nos = AeroBDSM.get_Sigma_Cxinos(M, self.geo.lambda_nos, key=1)
        return 2 * sigma_cx_i_nos * sin(alpha)**2
    
    def calculate_k_M(self, M):
        """Расчет коэффициента торможения потока"""
        lambda_ekv_con, corr  = self.calculate_boundary_layer_correction(M)
        return AeroBDSM.get_kM_Con(M, lambda_ekv_con).Value
    
    def calculate_cx_cor(self, M):
        """Расчет коэффициента сопротивления кормы"""
        return AeroBDSM.get_Cxcor_Par(M, self.geo.lambda_kor, self.geo.eta_kor).Value
    
    
    def calculate_x_d(self, M, alpha):
        """
        Расчет координаты центра давления изолированного фюзеляжа
        """
        
        def S_side_proekcia_nos(self):
            """Расчет площади боковой проекции носовой части"""
            
            def r_sphere(x):
                return math.sqrt(self.geo.r_zatup**2 - (x - self.geo.r_zatup)**2)
            
            def r_ogive(x):
                return self.geo.d_m/2 - self.geo.R + math.sqrt(self.geo.R**2 - (x - self.geo.l_nos)**2)
                 
            # 1. Площадь проекции сферической части
            def sphere_integrand(x):
                return 2 * r_sphere(x)
            
            S_sphere, error1 = quad(sphere_integrand, 0, self.geo.x_1)
#            print("S_sphere = ", S_sphere)
            
            # 2. Площадь проекции оживальной части
            def ogive_integrand(x):
                return 2 * r_ogive(x)
            
            S_ogive, error2 = quad(ogive_integrand, self.geo.x_1, self.geo.l_nos)
#           print("S_ogive = ", S_ogive)
            
            # 3. Общая площадь боковой проекции
            S_side_total = S_sphere + S_ogive
#           print("S_side_total = ", S_side_total)
            
            return S_side_total
        
        def calculate_center_of_mass_nose(self):
            """
            Расчет центра масс носовой части (сфера + оживало)
            """
            
            def r_sphere(x):
                return math.sqrt(self.geo.r_zatup**2 - (x - self.geo.r_zatup)**2)
            
            def r_ogive(x):
                return self.geo.d_m/2 - self.geo.R + math.sqrt(self.geo.R**2 - (x - self.geo.l_nos)**2)
            
            # 1. Числитель: ∫x * π * r(x)² dx (статический момент)
            def numerator_integrand(x):
                if x <= self.geo.x_1:
                    r = r_sphere(x)
                else:
                    r = r_ogive(x)
                return x * math.pi * r**2
            
            total_numerator, error_num = quad(numerator_integrand, 0, self.geo.l_nos)
            
            # 2. Знаменатель: ∫π * r(x)² dx (объем)
            def denominator_integrand(x):
                if x <= self.geo.x_1:
                    r = r_sphere(x)
                else:
                    r = r_ogive(x)
                return math.pi * r**2
            
            total_denominator, error_den = quad(denominator_integrand, 0, self.geo.l_nos)
            
            # 3. Центр масс
            if total_denominator > 0:
                x_cm = total_numerator / total_denominator
            else:
                x_cm = 0
            
#           print(f"Центр масс x_cm = {x_cm:.6f} м")
            
            return x_cm
        
        # Вычисляем необходимые параметры
        S_nos = S_side_proekcia_nos(self)
        x_c_pl_nos = calculate_center_of_mass_nose(self)
        
        # Получаем координату центра тяжести боковой проекции фюзеляжа
        x_c_pl_F = AeroBDSM.get_x_c_pl_F(
            self.geo.d_m, S_nos, x_c_pl_nos, 
            self.geo.lambda_nos, self.geo.lambda_cil, 
            self.geo.lambda_kor, self.geo.eta_kor, 
            0.0, 0.0, self.geo.l_don
        ).Value
        
        # Получаем площадь боковой проекции фюзеляжа
        S_bok = AeroBDSM.get_S_bok(
            self.geo.d_m, S_nos, 
            self.geo.lambda_cil, self.geo.lambda_kor, 
            self.geo.eta_kor, 0.0, 0.0, self.geo.l_don
        ).Value
    
        # Получаем коэффициент сопротивления цилиндра при нормальном обтекании
        cx_cil_N = AeroBDSM.get_Cx_Cil_N(M, abs(alpha)).Value
        
        # Расчет коэффициента нормальной силы фюзеляжа
        cy_alpha_iz_f = (self.calculate_cy_alpha_nos_cil(M) + 
                         self.calculate_cy_alpha_don())
        cy_iz_f = self.calculate_cy_f(M, alpha, cy_alpha_iz_f)
        
        # Расчет координаты фокуса фюзеляжа
        x_F_iz_f_alpha = self.calculate_x_alpha_F(M)
        
        # Расчет поправочного коэффициента k_a
        k_a = self.calculate_ka(M, alpha)
        
        # Основная формула для центра давления
        if cy_iz_f != 0:
            term1 = (cy_alpha_iz_f * k_a * math.sin(alpha) * math.cos(alpha) * 
                    x_F_iz_f_alpha)
            term2 = ((S_bok / self.geo.S_m) * cx_cil_N * 
                    (math.sin(alpha)**2) * np.sign(alpha) * x_c_pl_F)
            
            x_d_iz_f = (term1 + term2) / cy_iz_f
#           print(f"Центр давления x_d_iz_f = {x_d_iz_f:.6f} м")
        else:
            x_d_iz_f = 0
        
        return x_d_iz_f
    def calculate_forces(self, M, alpha, cy_f, cx_a_iz_f, cx_iz_f):
        """Расчет аэродинамических сил"""
        # Скорость полета
        V = M * self.geo.a_sound
        
        # Динамическое давление
        q = 0.5 * self.geo.rho * V**2
        
        # Нормальная и продольная силы в связанной СК
        Y_iz_f = cy_f * self.geo.S_m * q
        X_iz_f = cx_iz_f * self.geo.S_m * q
        
        # Подъемная сила и сила лобового сопротивления в скоростной СК
        Ya_iz_f = cy_f * cos(alpha) * self.geo.S_m * q - cx_iz_f * sin(alpha) * self.geo.S_m * q
        Xa_iz_f = cx_a_iz_f * self.geo.S_m * q
        
        return Y_iz_f, X_iz_f, Ya_iz_f, Xa_iz_f
    
    def calculate_cx_iz_f(self, cx_a_iz_f, cy_f, alpha):
        """Расчет коэффициента продольной силы в связанной СК"""
        if cos(alpha) != 0:
            return (cx_a_iz_f - cy_f * sin(alpha)) / cos(alpha)
        else:
            return 0
        
    def m_z_omega_z(self, x_cm):
        "Расчёт коэффициента производной момента тангажа по omega_z для корпуса (см.ЛиЧ стр 275)"
        l = self.geo.l_f
        term = x_cm / l
        numerator = 2 * (self.geo.lambda_fesulage**2) - (self.geo.lambda_nos**2)
        denumerator = 4 * self.geo.lambda_fesulage * (self.geo.lambda_fesulage - ((2 * self.geo.lambda_nos)/3))
        otn = numerator / denumerator
        return -2 * (1 - term  - otn + term**2)
        
    def run_calculations(self):
        """Основной метод выполнения расчетов"""
        results = {}
        
        # Расчет для разных чисел Маха
        for M in self.M_list:
            results[M] = {}
            
            # Базовые характеристики
            results[M]['cy_alpha'] = self.calculate_cy_alpha_nos_cil(M) + self.calculate_cy_alpha_don()
            results[M]['cx_a_0_p_predvar'] = self.calculate_cx_a_0_p_predvar(M)
            results[M]['cx_a_0_p'] = self.calculate_cx_a_0_p_nos_cil(M) + self.calculate_cx_a_0_p_kor(M)
            results[M]['cx_a_0_f'] = self.calculate_cx_a_0_f(M)
            results[M]['cx_a_don'] = self.calculate_cx_a_don(M) * self.calculate_coef_cx_a_don(M)
            results[M]['x_alpha_F'] = self.calculate_x_alpha_F(M) # Здесь учтён не только нос + цил НО и дно
            results[M]['k_M'] = self.calculate_k_M(M)
            
            # Коэффициент сопротивления при нулевом угле атаки
            results[M]['cx_a_0_iz_f'] = (results[M]['cx_a_0_p'] + results[M]['cx_a_0_f'] + 
                                       results[M]['cx_a_don'])
            
            # Расчет для разных углов атаки
            results[M]['alpha_results'] = {}
            
            # print(f"\n=== Отладка для M={M} ===")
            # print(f"cx_a_0_p = {results[M]['cx_a_0_p']:.6f}")
            # print(f"cx_a_0_f = {results[M]['cx_a_0_f']:.6f}")
            # print(f"cx_a_don = {results[M]['cx_a_don']:.6f}")
            # print(f"Сумма = {results[M]['cx_a_0_iz_f']:.6f}")
            # print(f"S_m = {self.geo.S_m:.6f} м²")
            
            
            
            
            for alpha in self.alpha_list:
                alpha_deg = round(np.rad2deg(alpha))
                results[M]['alpha_results'][alpha_deg] = {}
                
                # Коэффициент нормальной силы
                results[M]['alpha_results'][alpha_deg]['cy_f'] = self.calculate_cy_f(
                    M, alpha, results[M]['cy_alpha'])
                
                # Дополнительная продольная сила
                results[M]['alpha_results'][alpha_deg]['delta_cx'] = self.calculate_delta_cx(M, alpha)
                
                # Коэффициент сопротивления в скоростной СК
                if alpha != 0:
                    cx_a_i_iz_f = (results[M]['alpha_results'][alpha_deg]['cy_f'] * sin(alpha) + 
                                  results[M]['alpha_results'][alpha_deg]['delta_cx'] * cos(alpha))
                    results[M]['alpha_results'][alpha_deg]['cx_a_iz_f'] = (results[M]['cx_a_0_iz_f'] + 
                                                                          cx_a_i_iz_f)
                else:
                    results[M]['alpha_results'][alpha_deg]['cx_a_iz_f'] = results[M]['cx_a_0_iz_f']
                
                # Коэффициент продольной силы в связанной СК
                results[M]['alpha_results'][alpha_deg]['cx_iz_f'] = self.calculate_cx_iz_f(
                    results[M]['alpha_results'][alpha_deg]['cx_a_iz_f'],
                    results[M]['alpha_results'][alpha_deg]['cy_f'],
                    alpha)
                
                # Подъемная сила в скоростной СК
                results[M]['alpha_results'][alpha_deg]['cy_a_iz_f'] = (
                    results[M]['alpha_results'][alpha_deg]['cy_f'] * cos(alpha) - 
                    results[M]['alpha_results'][alpha_deg]['cx_iz_f'] * sin(alpha))
                
                # Расчет сил
                (results[M]['alpha_results'][alpha_deg]['Y_iz_f'],
                 results[M]['alpha_results'][alpha_deg]['X_iz_f'],
                 results[M]['alpha_results'][alpha_deg]['Ya_iz_f'],
                 results[M]['alpha_results'][alpha_deg]['Xa_iz_f']) = self.calculate_forces(
                     M, alpha,
                     results[M]['alpha_results'][alpha_deg]['cy_f'],
                     results[M]['alpha_results'][alpha_deg]['cx_a_iz_f'],
                     results[M]['alpha_results'][alpha_deg]['cx_iz_f'])
                results[M]['alpha_results'][alpha_deg]['x_d'] = self.calculate_x_d(M, alpha)
        return results
    
    def plot_all_graphs(self, results):
        """Построение графиков всех характеристик"""
        
        # Подготовка данных для графиков
        M_values = list(results.keys())
        
        # Получаем все уникальные углы атаки из результатов
        all_alpha_deg = set()
        for M in M_values:
            all_alpha_deg.update(results[M]['alpha_results'].keys())
        
        alpha_deg_values = sorted(all_alpha_deg)
        
        # Выбираем подмножество углов атаки для графиков (не более 5 для читаемости)
        if len(alpha_deg_values) > 15:
            # Выбираем равномерно распределенные углы атаки
            step = len(alpha_deg_values) // 5
            plot_alpha_deg = alpha_deg_values[::step][:5]
        else:
            plot_alpha_deg = alpha_deg_values
        
        # Выбираем подмножество чисел Маха для графиков (не более 4 для читаемости)
        if len(M_values) > 14:
            step = len(M_values) // 4
            selected_M = M_values[::step][:4]
        else:
            selected_M = M_values
        
        # 1. Графики cx_a_iz_f от числа Маха для разных углов атаки
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        for alpha_deg in plot_alpha_deg:
            cx_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cx_values.append(results[M]['alpha_results'][alpha_deg]['cx_a_iz_f'])
                else:
                    cx_values.append(np.nan)
            plt.plot(M_values, cx_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        plt.xlabel('Число Маха, M')
        plt.ylabel('cx_a_iz_f')
        plt.title('Коэффициент сопротивления (скоростная СК) от числа Маха')
        plt.legend()
        plt.grid(True)
        
        # 2. График cx_a_iz_f от угла атаки для разных чисел Маха
        plt.subplot(2, 3, 2)
        
        for M in selected_M:
            if M in results:
                cx_values = []
                alpha_deg_for_plot = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        cx_values.append(results[M]['alpha_results'][alpha_deg]['cx_a_iz_f'])
                        alpha_deg_for_plot.append(alpha_deg)
                plt.plot(alpha_deg_for_plot, cx_values, 'o-', label=f'M = {M}', linewidth=2)
        
        plt.xlabel('Угол атаки, °')
        plt.ylabel('cx_a_iz_f')
        plt.title('Коэффициент сопротивления от угла атаки')
        plt.legend()
        plt.grid(True)
        
        # 3. График cy_a_iz_f от числа Маха для разных углов атаки
        plt.subplot(2, 3, 3)
        for alpha_deg in plot_alpha_deg:
            cy_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cy_values.append(results[M]['alpha_results'][alpha_deg]['cy_a_iz_f'])
                else:
                    cy_values.append(np.nan)
            plt.plot(M_values, cy_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        plt.xlabel('Число Маха, M')
        plt.ylabel('cy_a_iz_f')
        plt.title('Коэффициент подъемной силы от числа Маха')
        plt.legend()
        plt.grid(True)
        
        # 4. График cy_a_iz_f от угла атаки для разных чисел Маха
        plt.subplot(2, 3, 4)
        for M in selected_M:
            if M in results:
                cy_values = []
                alpha_deg_for_plot = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        cy_values.append(results[M]['alpha_results'][alpha_deg]['cy_a_iz_f'])
                        alpha_deg_for_plot.append(alpha_deg)
                plt.plot(alpha_deg_for_plot, cy_values, 'o-', label=f'M = {M}', linewidth=2)
        
        plt.xlabel('Угол атаки, °')
        plt.ylabel('cy_a_iz_f')
        plt.title('Коэффициент подъемной силы от угла атаки')
        plt.legend()
        plt.grid(True)
        
        # 5. Графики сил от числа Маха (для первого доступного угла атаки)
        plt.subplot(2, 3, 5)
        if alpha_deg_values:
            alpha_ref = alpha_deg_values[0]  # Первый доступный угол атаки
            
            Y_values = []
            X_values = []
            Ya_values = []
            Xa_values = []
            for M in M_values:
                if alpha_ref in results[M]['alpha_results']:
                    Y_values.append(results[M]['alpha_results'][alpha_ref]['Y_iz_f'])
                    X_values.append(results[M]['alpha_results'][alpha_ref]['X_iz_f'])
                    Ya_values.append(results[M]['alpha_results'][alpha_ref]['Ya_iz_f'])
                    Xa_values.append(results[M]['alpha_results'][alpha_ref]['Xa_iz_f'])
                else:
                    Y_values.append(np.nan)
                    X_values.append(np.nan)
                    Ya_values.append(np.nan)
                    Xa_values.append(np.nan)
            
            plt.plot(M_values, Y_values, 'r-', label='Y_iz_f', linewidth=2)
            plt.plot(M_values, X_values, 'g-', label='X_iz_f', linewidth=2)
            plt.plot(M_values, Ya_values, 'b-', label='Ya_iz_f', linewidth=2)
            plt.plot(M_values, Xa_values, 'purple', label='Xa_iz_f', linewidth=2)
            
            plt.xlabel('Число Маха, M')
            plt.ylabel('Силы, Н')
            plt.title(f'Аэродинамические силы от числа Маха (α = {alpha_ref}°)')
            plt.legend()
            plt.grid(True)
        
        # 6. График производной коэффициента нормальной силы
        plt.subplot(2, 3, 6)  # Заменяем последний график
        for alpha_deg in alpha_deg_values:
            x_d_values = [results[M]['alpha_results'][alpha_deg]['x_d'] for M in M_values]
            plt.plot(M_values, x_d_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        plt.xlabel('Число Маха, M')
        plt.ylabel('x_d, м')
        plt.title('Центр давления от числа Маха')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительные графики для более детального анализа
        
        # Графики сил для разных углов атаки
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Y_iz_f от M для разных углов атаки
        for alpha_deg in plot_alpha_deg:
            Y_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    Y_values.append(results[M]['alpha_results'][alpha_deg]['Y_iz_f'])
                else:
                    Y_values.append(np.nan)
            axes[0, 0].plot(M_values, Y_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 0].set_xlabel('Число Маха, M')
        axes[0, 0].set_ylabel('Y_iz_f, Н')
        axes[0, 0].set_title('Нормальная сила от числа Маха')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # X_iz_f от M для разных углов атаки
        for alpha_deg in plot_alpha_deg:
            X_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    X_values.append(results[M]['alpha_results'][alpha_deg]['X_iz_f'])
                else:
                    X_values.append(np.nan)
            axes[0, 1].plot(M_values, X_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 1].set_xlabel('Число Маха, M')
        axes[0, 1].set_ylabel('X_iz_f, Н')
        axes[0, 1].set_title('Продольная сила от числа Маха')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Ya_iz_f от M для разных углов атаки
        for alpha_deg in plot_alpha_deg:
            Ya_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    Ya_values.append(results[M]['alpha_results'][alpha_deg]['Ya_iz_f'])
                else:
                    Ya_values.append(np.nan)
            axes[1, 0].plot(M_values, Ya_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 0].set_xlabel('Число Маха, M')
        axes[1, 0].set_ylabel('Ya_iz_f, Н')
        axes[1, 0].set_title('Подъемная сила от числа Маха')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Xa_iz_f от M для разных углов атаки
        for alpha_deg in plot_alpha_deg:
            Xa_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    Xa_values.append(results[M]['alpha_results'][alpha_deg]['Xa_iz_f'])
                else:
                    Xa_values.append(np.nan)
            axes[1, 1].plot(M_values, Xa_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 1].set_xlabel('Число Маха, M')
        axes[1, 1].set_ylabel('Xa_iz_f, Н')
        axes[1, 1].set_title('Сила лобового сопротивления от числа Маха')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # НОВЫЕ ГРАФИКИ: cx_iz_f и cy_iz_f от чисел Маха и углов атаки
        self.plot_cx_cy_iz_f_graphs(results)
    
    def plot_cx_cy_iz_f_graphs(self, results):
        """Построение графиков для cx_iz_f и cy_iz_f"""
        
        # Подготовка данных для графиков
        M_values = list(results.keys())
        
        # Получаем все уникальные углы атаки из результатов
        all_alpha_deg = set()
        for M in M_values:
            all_alpha_deg.update(results[M]['alpha_results'].keys())
        
        alpha_deg_values = sorted(all_alpha_deg)
        
        # Выбираем подмножество углов атаки для графиков (не более 5 для читаемости)
        if len(alpha_deg_values) > 15:
            step = len(alpha_deg_values) // 5
            plot_alpha_deg = alpha_deg_values[::step][:]
        else:
            plot_alpha_deg = alpha_deg_values
        
        # Выбираем подмножество чисел Маха для графиков (не более 4 для читаемости)
        if len(M_values) > 14:
            step = len(M_values) // 5
            selected_M = M_values[::step][:]
        else:
            selected_M = M_values
        
        # Создание фигуры для графиков cx_iz_f и cy_iz_f
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. График cx_iz_f от числа Маха для разных углов атаки
        for alpha_deg in plot_alpha_deg:
            cx_iz_f_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cx_iz_f_values.append(results[M]['alpha_results'][alpha_deg]['cx_iz_f'])
                else:
                    cx_iz_f_values.append(np.nan)
            axes[0, 0].plot(M_values, cx_iz_f_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[0, 0].set_xlabel('Число Маха, M')
        axes[0, 0].set_ylabel('cx_iz_f')
        axes[0, 0].set_title('Коэффициент продольной силы (связанная СК)\nот числа Маха')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. График cx_iz_f от угла атаки для разных чисел Маха
        for M in selected_M:
            if M in results:
                cx_iz_f_values = []
                alpha_deg_for_plot = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        cx_iz_f_values.append(results[M]['alpha_results'][alpha_deg]['cx_iz_f'])
                        alpha_deg_for_plot.append(alpha_deg)
                axes[0, 1].plot(alpha_deg_for_plot, cx_iz_f_values, 'o-', label=f'M = {M}', linewidth=2)
        
        axes[0, 1].set_xlabel('Угол атаки, °')
        axes[0, 1].set_ylabel('cx_iz_f')
        axes[0, 1].set_title('Коэффициент продольной силы (связанная СК)\nот угла атаки')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. График cy_iz_f от числа Маха для разных углов атаки
        for alpha_deg in plot_alpha_deg:
            cy_iz_f_values = []
            for M in M_values:
                if alpha_deg in results[M]['alpha_results']:
                    cy_iz_f_values.append(results[M]['alpha_results'][alpha_deg]['cy_f'])
                else:
                    cy_iz_f_values.append(np.nan)
            axes[1, 0].plot(M_values, cy_iz_f_values, 'o-', label=f'α = {alpha_deg}°', linewidth=2)
        
        axes[1, 0].set_xlabel('Число Маха, M')
        axes[1, 0].set_ylabel('cy_iz_f')
        axes[1, 0].set_title('Коэффициент нормальной силы (связанная СК)\nот числа Маха')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. График cy_iz_f от угла атаки для разных чисел Маха
        for M in selected_M:
            if M in results:
                cy_iz_f_values = []
                alpha_deg_for_plot = []
                for alpha_deg in alpha_deg_values:
                    if alpha_deg in results[M]['alpha_results']:
                        cy_iz_f_values.append(results[M]['alpha_results'][alpha_deg]['cy_f'])
                        alpha_deg_for_plot.append(alpha_deg)
                axes[1, 1].plot(alpha_deg_for_plot, cy_iz_f_values, 'o-', label=f'M = {M}', linewidth=2)
        
        axes[1, 1].set_xlabel('Угол атаки, °')
        axes[1, 1].set_ylabel('cy_iz_f')
        axes[1, 1].set_title('Коэффициент нормальной силы (связанная СК)\nот угла атаки')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self, results):
        """Вывод результатов расчетов"""
        print('\033[1;32;40m РЕЗУЛЬТАТЫ РАСЧЕТА АДХ ФЮЗЕЛЯЖА \033[0m')
        print('=' * 60)

        # Вывод для нескольких характерных значений M
        sample_M = self.M_list[:3]  # Берем первые 3 значения M для вывода
        sample_alpha = 5  # градусы
        
        for M in sample_M:
            if M in results:
                m_results = results[M]
                
                # Ищем ближайший доступный угол атаки
                available_alpha = list(m_results['alpha_results'].keys())
                if available_alpha:
                    closest_alpha = min(available_alpha, key=lambda x: abs(x - sample_alpha))
                    alpha_results = m_results['alpha_results'][closest_alpha]
                    
                    print(f'\n\033[1;34;40m М = {M:.1f}, α = {sample_alpha}° \033[0m')
                    print(f'Cy_alpha = {m_results["cy_alpha"]:.6f}')
                    print(f'Cx_a_0_iz_f = {m_results["cx_a_0_iz_f"]:.6f}')
                    print(f'cy_f = {alpha_results["cy_f"]:.6f}')
                    print(f'cx_a_iz_f = {alpha_results["cx_a_iz_f"]:.6f}')
                    print(f'cy_a_iz_f = {alpha_results["cy_a_iz_f"]:.6f}')
                    print(f'cx_iz_f = {alpha_results["cx_iz_f"]:.6f}')  # Добавлен вывод cx_iz_f
                    print(f'Y_iz_f = {alpha_results["Y_iz_f"]:.2f} Н')
                    print(f'X_iz_f = {alpha_results["X_iz_f"]:.2f} Н')
                    print(f'Ya_iz_f = {alpha_results["Ya_iz_f"]:.2f} Н')
                    print(f'Xa_iz_f = {alpha_results["Xa_iz_f"]:.2f} Н')
                    print(f'x_d = {alpha_results["x_d"]:.4f} м')


# Основная программа
if __name__ == "__main__":
    # Создание объектов
    geometry = FuselageGeometry()
    calculator = AerodynamicCalculator(geometry)

    # Выполнение расчетов
    print("Выполнение расчетов...")
    results = calculator.run_calculations()
    
    # Вывод результатов
    calculator.print_results(results)
    
    # Построение графиков
    print("\nПостроение графиков...")
    calculator.plot_all_graphs(results)
    
    # Вывод значений донного сопротивления
    print('\n\033[1;33;40m Значения донного сопротивления: \033[0m')
    sample_M = [0.5,1,1.5,2]  # Берем первые 4 значения M
    for M in sample_M:
        if M in results:
            cx_don = results[M]['cx_a_don']
            print(f'М = {M}: Cx_don = {cx_don:.6f}')
