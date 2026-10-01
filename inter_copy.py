import numpy as np
import math
import AeroBDSM
from math import *
import matplotlib.pyplot as plt
from Atmosphere import atmo
from Fesulyadge_class import FuselageGeometry, AerodynamicCalculator as FuselageCalculator
from ruli_class import RudderGeometry, RudderAerodynamicCalculator as RudderCalculator
from Krilo_class import WingGeometry, WingAerodynamicCalculator as WingCalculator
 
#Krylo
class AircraftAerodynamics:
    """Класс для расчета суммарных АДХ летательного аппарата с учетом интерференции"""
    
    def __init__(self, H, x0_rel, config_scheme, num_front_consoles, num_rear_consoles):
         
        self.H =  H  # высота полета
        self.x0_rel = x0_rel  # координата центра масс
        self.rho = atmo.rho(H)
        self.a_sound = atmo.a(H)
        self.config_scheme = config_scheme  # схема конфигурации
        self.num_front_consoles = num_front_consoles  # количество консолей в переднем поясе
        self.num_rear_consoles = num_rear_consoles    # количество консолей в заднем поясе
        
        # Инициализация геометрии компонентов
        self._initialize_components()
        
        # Характерные параметры ЛА
        self.L_ref = self.fuselage_geo.l_f  # характерная длина (длина фюзеляжа)
        self.S_ref = self.fuselage_geo.S_m  # характерная площадь
        
        # Инициализация конфигурации консолей
        self._initialize_consoles_configuration()
         
        # УВЕЛИЧЕННОЕ количество точек для расчетов
        self.M_list = self.wing_geo.M_list
        self.alpha_list = self.wing_geo.alpha_list
        self.phi_alpha_list = [0]
        
        #self.c_xa_0_iz_profil= self.WingCalculator.c_xa_0_iz_profil_wing
         
        # Добавляем варьирование по delta (угол отклонения рулей)
        self.n = 5
        self.delta_list = np.deg2rad(np.linspace(0, 0, self.n))  # 5 значений delta от -10° до +10°
        
        self.num_kons = len(self.consoles_config['II'])
                
        if self.consoles_config['II']:
            self.psi = np.rad2deg(self.consoles_config['II'][0]['psi'])
        else:
            self.psi = 45

    def _initialize_components(self):
        """Инициализация всех компонентов ЛА"""
        # Инициализация геометрии фюзеляжа и определение типа носовой части
        self.fuselage_geo = FuselageGeometry()
        self.fuselage_calc = FuselageCalculator(self.fuselage_geo)
        self.nose_type = self.fuselage_calc.detect_nose_type()
        
        self.wing_geo = WingGeometry()
        self.wing_calc = WingCalculator(self.wing_geo)
        
        self.rudder_geo = RudderGeometry()
        self.rudder_calc = RudderCalculator(self.rudder_geo)
        
        # Общие параметры
        self.l_op = self.wing_geo.l
        self.b_bort = self.wing_geo.b_b
        self.chi_pk = math.degrees(self.wing_geo.X_0) 
        self.b_k = self.wing_geo.b1
        self.d_op = self.wing_geo.d_op
        self.H = self.wing_geo.H  # высота полета
        
        self.r_zatup = self.fuselage_geo.r_zatup
        self.R = self.fuselage_geo.R  # радиус оживала
        self.l_nos = self.fuselage_geo.l_nos
        self.a = self.l_nos - self.r_zatup
        self.d_m = self.fuselage_geo.d_m
        self.l_cyl = self.fuselage_geo.l_cil
        self.d_k = self.fuselage_geo.d_k
        self.l_don = self.fuselage_geo.l_don
        
        
        # Блок расчёта высотных поправок для AD_tabl2
        print("\nРасчет таблицы AD_tabl2...")
        self.AD_tabl2 = self.calculate_height_corrections_silent()

        # Вывод результата
        print("\nСформированная таблица AD_tabl2:")
        print("[H, M, delta_Cxa_0h, Cxa_0_дн]")
        for row in self.AD_tabl2:
            print(f"  [{row[0]}, {row[1]}, {row[2]:.6f}, {row[3]:.6f}]")
        # Параметры в зависимости от типа носовой части
        if self.nose_type == "conical_with_sphere":
            # Коническая со сферическим затуплением
            self.R = None  # Не используется для конической
            self.a = self.l_nos - self.r_zatup
            self.theta_kon = np.arctan((self.d_m/2) / (self.l_nos - self.r_zatup))
            self.x_1 = self.r_zatup * (1 - np.sin(self.theta_kon))
            self.nose_name = "Коническая носовая часть со сферическим затуплением"
            
        elif self.nose_type == "ogive_with_sphere":
            # Оживальная со сферическим затуплением  
            self.a = self.l_nos - self.r_zatup
            r_sf_otn = 2 * self.r_zatup / self.d_m
            phi = np.arccos((self.a**2 - (self.d_m/2 - self.r_zatup)**2) / (self.a**2 + (self.d_m/2 - self.r_zatup)**2))
            self.x_1 = self.r_zatup * (1 - np.sin(phi))
            self.nose_name = "Оживальная носовая часть со сферическим затуплением"

        elif self.nose_type == "simple_conical":
            # Простая коническая
            self.r_zatup = 0
            self.theta_kon = np.arctan((self.d_m/2) / self.l_nos)
            self.x_1 = 0
            self.nose_name = "Коническая носовая часть"
            
        elif self.nose_type == "simple_ogive":
            # Простая оживальная
            self.r_zatup = 0
            self.R = 1.3
            self.x_1 = 0
            self.nose_name = "Оживальная носовая часть"
            
        elif self.nose_type == "spherical":
            # Сферическая
            self.r_zatup = self.d_m/2
            self.x_1 = self.l_nos
            self.nose_name = "Сферическая носовая часть"

        self.x = np.linspace(0, self.l_nos + self.l_cyl + self.l_don, 1000)
        
    def calculate_height_corrections_silent(self):
        """Тихий расчет высотных поправок и донных сопротивлений для AD_tabl2"""
        
        heights = [0, 5000, 10000, 20000]
        target_M_list = [0.5, 1.0, 1.5, 2.0]
        
        height_results = {H: {'cx_a_0_total': [], 'cx_don_total': []} for H in heights}
        
        # Сохраняем исходные параметры
        original_H = self.H
        original_wing_H = self.wing_geo.H
        original_fuselage_H = self.fuselage_geo.H
        original_rudder_H = self.rudder_geo.H
        
        # Сохраняем исходные списки M
        original_M_list_wing = self.wing_geo.M_list.copy()
        original_M_list_fuselage = self.fuselage_calc.M_list.copy() if hasattr(self.fuselage_calc, 'M_list') else []
        original_M_list_rudder = self.rudder_geo.M_list.copy()
        
        for H in heights:
            try:
                # Устанавливаем новую высоту
                self.H = H
                self.wing_geo.H = H
                self.fuselage_geo.H = H
                self.rudder_geo.H = H
                
                # Обновляем параметры атмосферы
                self.wing_geo.rho = atmo.rho(H)
                self.wing_geo.a_sound = atmo.a(H)
                self.fuselage_geo.rho = atmo.rho(H)
                self.fuselage_geo.a_sound = atmo.a(H)
                self.rudder_geo.rho = atmo.rho(H)
                self.rudder_geo.a_sound = atmo.a(H)
                
                # Временно изменяем списки M только на target_M_list
                self.wing_geo.M_list = target_M_list
                if hasattr(self.wing_calc, 'M_list'):
                    self.wing_calc.M_list = target_M_list
                if hasattr(self.fuselage_calc, 'M_list'):
                    self.fuselage_calc.M_list = target_M_list
                self.rudder_geo.M_list = target_M_list
                
                # Расчеты для текущей высоты
                fuselage_results_H = self.fuselage_calc.run_calculations()
                wing_results_H = self.wing_calc.run_calculations()
                rudder_results_H = self.rudder_calc.run_calculations()
                
                # Собираем Cx_a_0 для всех компонентов
                cx_a_0_fuselage = []
                cx_a_0_wing = []
                cx_a_0_rudder = []
                
                # Для фюзеляжа
                for M in target_M_list:
                    if M in fuselage_results_H:
                        cx_a_0_fuselage.append(fuselage_results_H[M]['cx_a_0_iz_f'])
                    else:
                        cx_a_0_fuselage.append(0)
                
                # Для крыла
                for M in target_M_list:
                    if M in wing_results_H and wing_results_H[M]['alpha_results']:
                        first_alpha_key = list(wing_results_H[M]['alpha_results'].keys())[0]
                        cx_a_0_wing.append(wing_results_H[M]['alpha_results'][first_alpha_key]['c_xa_0_iz_profil_wing'])
                    else:
                        cx_a_0_wing.append(0)
                
                # Для руля
                for M in target_M_list:
                    if M in rudder_results_H:
                        cx_a_0_rudder.append(rudder_results_H[M]['c_xa_0_iz_profil_rul'])
                    else:
                        cx_a_0_rudder.append(0)
                
                # Суммирование Cx_a_0 в зависимости от конфигурации
                if self.config_scheme == "default":
                    cx_a_0_total = [x + y + z for x, y, z in zip(cx_a_0_fuselage, cx_a_0_wing, cx_a_0_rudder)]
                elif self.config_scheme == "no_rudders":
                    cx_a_0_total = [x + y for x, y in zip(cx_a_0_fuselage, cx_a_0_wing)]
                elif self.config_scheme == "no_rudders_wings":
                    cx_a_0_total = cx_a_0_fuselage
                else:
                    cx_a_0_total = [x + y + z for x, y, z in zip(cx_a_0_fuselage, cx_a_0_wing, cx_a_0_rudder)]
                
                # Собираем донные сопротивления
                cx_don_fuselage = []
                cx_don_wing = []
                cx_don_rudder = []
                
                # Для фюзеляжа
                for M in target_M_list:
                    if M in fuselage_results_H:
                        cx_don_fuselage.append(fuselage_results_H[M]['cx_a_don'])
                    else:
                        cx_don_fuselage.append(0)
                
                # Для крыла (cxa_prof + cxa_don)
                for M in target_M_list:
                    if M in wing_results_H:
                        # Используем только профильное сопротивление крыла (без донного)
                        # Для AD_tabl2 нужен только дон фюзеляжа
                        cx_don_wing.append(wing_results_H[M]['cxa_prof'])
                    else:
                        cx_don_wing.append(0)
                
                # Для руля
                for M in target_M_list:
                    if M in rudder_results_H:
                        # Используем только профильное сопротивление руля (без донного)
                        cx_don_rudder.append(rudder_results_H[M]['cxa_prof'])
                    else:
                        cx_don_rudder.append(0)
                
                # Суммирование донных сопротивлений
                # ВНИМАНИЕ: Для AD_tabl2 в c_xa0_дн нужен ТОЛЬКО дон фюзеляжа!
                # Остальные компоненты не имеют донного сопротивления в этом смысле
                if self.config_scheme == "default":
                    # Для c_xa0_дн в AD_tabl2 берем только дон фюзеляжа
                    cx_don_total = cx_don_fuselage
                    # Но если нужно профильное сопротивление всех компонентов для высотной поправки:
                    # cx_prof_total = [x + y + z for x, y, z in zip(cx_don_fuselage, cx_don_wing, cx_don_rudder)]
                elif self.config_scheme == "no_rudders":
                    cx_don_total = cx_don_fuselage
                    # cx_prof_total = [x + y for x, y in zip(cx_don_fuselage, cx_don_wing)]
                elif self.config_scheme == "no_rudders_wings":
                    cx_don_total = cx_don_fuselage
                else:
                    cx_don_total = cx_don_fuselage
                
                # Сохраняем результаты
                height_results[H]['cx_a_0_total'] = cx_a_0_total
                height_results[H]['cx_don_total'] = cx_don_total
                
                # Для отладки
                print(f"\nH={H}:")
                print(f"  Cx_a_0 фюзеляж: {[f'{x:.6f}' for x in cx_a_0_fuselage]}")
                print(f"  Cx_a_0 крыло: {[f'{x:.6f}' for x in cx_a_0_wing]}")
                print(f"  Cx_a_0 руль: {[f'{x:.6f}' for x in cx_a_0_rudder]}")
                print(f"  Cx_a_0 суммарное: {[f'{x:.6f}' for x in cx_a_0_total]}")
                print(f"  Cx_don фюзеляж: {[f'{x:.6f}' for x in cx_don_fuselage]}")
                
            except Exception as e:
                print(f"Ошибка при расчете для высоты {H}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Восстанавливаем исходные параметры
        self.H = original_H
        self.wing_geo.H = original_wing_H
        self.fuselage_geo.H = original_fuselage_H
        self.rudder_geo.H = original_rudder_H
        
        # Восстанавливаем параметры атмосферы
        self.wing_geo.rho = atmo.rho(original_H)
        self.wing_geo.a_sound = atmo.a(original_H)
        self.fuselage_geo.rho = atmo.rho(original_H)
        self.fuselage_geo.a_sound = atmo.a(original_H)
        self.rudder_geo.rho = atmo.rho(original_H)
        self.rudder_geo.a_sound = atmo.a(original_H)
        
        # Восстанавливаем исходные списки M
        self.wing_geo.M_list = original_M_list_wing
        if hasattr(self.wing_calc, 'M_list'):
            self.wing_calc.M_list = original_M_list_wing
        if hasattr(self.fuselage_calc, 'M_list'):
            self.fuselage_calc.M_list = original_M_list_fuselage
        self.rudder_geo.M_list = original_M_list_rudder
        
        # Формируем таблицу AD_tabl2
        AD_tabl2 = []
        
        for H in heights:
            if H in height_results and height_results[H]['cx_a_0_total']:
                for i, M in enumerate(target_M_list):
                    if i < len(height_results[H]['cx_a_0_total']):
                        # Высотная поправка: Cx_a_0_h(M,H) - Cx_a_0_h(M,H=0)
                        height_correction = 0
                        if H > 0 and 0 in height_results and i < len(height_results[0]['cx_a_0_total']):
                            height_correction = (height_results[H]['cx_a_0_total'][i] - 
                                                height_results[0]['cx_a_0_total'][i])
                        
                        # Донное сопротивление - только фюзеляжа!
                        cx_don_value = height_results[H]['cx_don_total'][i] if i < len(height_results[H]['cx_don_total']) else 0
                        
                        AD_tabl2.append([H, M, height_correction, cx_don_value])

        
        return AD_tabl2

    def save_table_to_python_format_2(self, results, filename="aircraft_data.py"):
        """Сохранение таблицы в формате Python списка списков"""
        table_data = self.create_data_table(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Заголовок файла с комментариями
            f.write("# АЭРОДИНАМИЧЕСКИЕ ХАРАКТЕРИСТИКИ ЛЕТАТЕЛЬНОГО АППАРАТА\n")
            f.write(f"# Конфигурация: {self.config_scheme}\n")
            f.write("# Формат данных: [M, alpha_rad, delta, c_xa, c_ya, c_x, c_y, m_z]\n")
            f.write("# M - число Маха\n")
            f.write("# alpha_rad - угол атаки в радианах\n")
            f.write("# delta - угол отклонения рулей в радианах\n")
            f.write("# c_xa, c_ya - коэффициенты по формулам (2.15) и (2.16)\n")
            f.write("# c_x, c_y - коэффициенты продольной и нормальной силы\n")
            f.write("# m_z - коэффициент момента тангажа\n")
            f.write("aerodynamics_data = [\n")
            
            # Данные в формате Python списков
            for i, row in enumerate(table_data):
                line = f"    [{row['M']:.6f}, {row['alpha']:.8f}, {row['delta']:.8f}, "
                line += f"{row['c_xa']:.8f}, {row['c_ya']:.8f}, "
                line += f"{row['c_x']:.8f}, {row['c_y']:.8f}, {row['m_z']:.8f}]"
                
                # Добавляем запятую для всех строк кроме последней
                if i < len(table_data) - 1:
                    line += ","
                else:
                    line += ""
                
                f.write(line + "\n")
            
            f.write("]\n\n")
            
            # Дополнительная информация о размерах данных
            f.write(f"# Всего точек расчета: {len(table_data)}\n")
            f.write(f"# Диапазон чисел Маха: {min(row['M'] for row in table_data):.2f} - {max(row['M'] for row in table_data):.2f}\n")
            f.write(f"# Диапазон углов атаки: {np.rad2deg(min(row['alpha'] for row in table_data)):.1f}° - {np.rad2deg(max(row['alpha'] for row in table_data)):.1f}°\n")
            f.write(f"# Диапазон углов отклонения рулей: {np.rad2deg(min(row['delta'] for row in table_data)):.8f}° - {np.rad2deg(max(row['delta'] for row in table_data)):.1f}°\n\n")
            
            # ДОБАВЛЯЕМ ТАБЛИЦУ AD_tabl2
            f.write("#" * 60 + "\n")
            f.write("# ТАБЛИЦА ВЫСОТНЫХ ПОПРАВОК И ДОННЫХ СОПРОТИВЛЕНИЙ (AD_tabl2)\n")
            f.write("# Формат: [H, M, delta_Cxa_0h, Cxa_0_дн]\n")
            f.write("# H - высота, м\n")
            f.write("# M - число Маха\n")
            f.write("# delta_Cxa_0h - высотная поправка\n")
            f.write("# Cxa_0_дн - коэффициент донного сопротивления фюзеляжа\n")
            f.write("AD_tabl2 = [\n")
            
            for i, row in enumerate(self.AD_tabl2):
                # Убеждаемся, что значения - это числа, а не кортежи
                delta_val = row[2]
                cx_don_val = row[3]
                
                # Если значения - кортежи, берем первый элемент
                if isinstance(delta_val, tuple):
                    delta_val = delta_val[0] if delta_val else 0
                if isinstance(cx_don_val, tuple):
                    cx_don_val = cx_don_val[0] if cx_don_val else 0
                
                # Преобразуем в float для безопасности
                delta_val = float(delta_val)
                cx_don_val = float(cx_don_val)
                
                line = f"    [{row[0]}, {row[1]}, {delta_val:.6f}, {cx_don_val:.6f}]"
                if i < len(self.AD_tabl2) - 1:
                    line += ","
                f.write(line + "\n")
            
            f.write("]\n\n")
            
            # Информация о AD_tabl2
            f.write(f"# Всего записей в AD_tabl2: {len(self.AD_tabl2)}\n")
            f.write(f"# Высоты: {sorted(set([row[0] for row in self.AD_tabl2]))}\n")
            f.write(f"# Числа Маха: {sorted(set([row[1] for row in self.AD_tabl2]))}\n")
        
        print(f"Таблица данных сохранена в Python-файл: {filename}")
        return table_data
    def _create_consoles_array(self, num_consoles, console_type, x6_position, is_front_belt=True):
        """Создание массива консолей с заданным количеством и равномерным распределением по окружности"""
        consoles = []
        
        if num_consoles <= 0:
            return consoles
            
        if num_consoles == 4:
            # X-образная схема для 4 консолей
            angles = [np.deg2rad(45), np.deg2rad(135), np.deg2rad(225), np.deg2rad(315)]
        else:
            # Для другого количества - равномерное распределение
            angles = np.linspace(0, 2*np.pi, num_consoles, endpoint=False)
        
        for i, angle in enumerate(angles):
            if console_type == 'rudder':
                # Параметры для руля
                console_params = {
                    'psi': angle, 
                    'x6': x6_position, 
                    'IsReflected': 1 if np.pi/2 <= angle <= 3*np.pi/2 else 0,  # Отражение для нижней полусферы
                    'type': 'rudder',
                    'l1k': self.rudder_geo.l, 
                    'zeta': self.rudder_geo.zeta_p,
                    'b6': self.rudder_geo.b_b, 
                    'S1k': self.rudder_geo.S/2,
                    'bA': self.rudder_geo.bA, 
                    'xA': self.rudder_geo.xA,
                    'lambda_k': self.rudder_geo.lambda_p, 
                    'X_05': self.rudder_geo.X_05
                }
            else:  # console_type == 'wing'
                # Параметры для крыла
                console_params = {
                    'psi': angle, 
                    'x6': x6_position, 
                    'IsReflected': 1 if np.pi/2 <= angle <= 3*np.pi/2 else 0,  # Отражение для нижней полусферы
                    'type': 'wing',
                    'l1k': self.wing_geo.l, 
                    'zeta': self.wing_geo.zeta_k,
                    'b6': self.wing_geo.b_b, 
                    'S1k': self.wing_geo.S/2,
                    'bA': self.wing_geo.bA, 
                    'xA': self.wing_geo.xA,
                    'lambda_k': self.wing_geo.lambda_k, 
                    'X_05': self.wing_geo.X_05
                }
            
            consoles.append(console_params)
        
        return consoles

    def _initialize_consoles_configuration(self):
        """Инициализация конфигурации консолей в зависимости от выбранной схемы"""
        if self.config_scheme == "default":
            self._setup_default_configuration()
        elif self.config_scheme == "no_rudders":
            self._setup_no_rudders_configuration()
        elif self.config_scheme == "no_rudders_wings":
            self._setup_no_rudders_wings_configuration()
        elif self.config_scheme == "wings_front":
            self._setup_wings_front_configuration()
        elif self.config_scheme == "rudders_front":
            self._setup_rudders_front_configuration()
        elif self.config_scheme == "custom":
            self._setup_custom_configuration()
        else:
            print(f"Неизвестная схема конфигурации: {self.config_scheme}. Используется схема по умолчанию.")
            self._setup_default_configuration()

    def _setup_default_configuration(self):
        """Стандартная конфигурация: передний пояс - рули, задний пояс - крылья"""
        self.consoles_config = {
            'I': self._create_consoles_array(self.num_front_consoles, 'rudder', 1.5, True),
            'II': self._create_consoles_array(self.num_rear_consoles, 'wing', 1.0, False)
        }

    def _setup_no_rudders_configuration(self):
        """Конфигурация с отключенными рулями (только крылья)"""
        self.consoles_config = {
            'I': [],  # Передний пояс пустой
            'II': self._create_consoles_array(self.num_rear_consoles, 'wing', 1.0, False)
        }

    def _setup_no_rudders_wings_configuration(self):
        """Конфигурация с отключенными рулями и крыльями (только фюзеляж)"""
        self.consoles_config = {
            'I': [],  # Передний пояс пустой
            'II': []   # Задний пояс пустой
        }

    def _setup_wings_front_configuration(self):
        """Конфигурация с крыльями впереди (передний пояс) и рулями сзади (задний пояс)"""
        self.consoles_config = {
            'I': self._create_consoles_array(self.num_front_consoles, 'wing', 1.5, True),
            'II': self._create_consoles_array(self.num_rear_consoles, 'rudder', 1.0, False)
        }

    def _setup_rudders_front_configuration(self):
        """Конфигурация с рулями впереди (передний пояс) и крыльями сзади (задний пояс) - стандартная"""
        self._setup_default_configuration()

    def _setup_custom_configuration(self):
        """Пользовательская конфигурация с произвольным количеством консолей"""
        # Здесь можно реализовать любую специфическую конфигурацию
        self.consoles_config = {
            'I': self._create_consoles_array(self.num_front_consoles, 'rudder', 1.5, True),
            'II': self._create_consoles_array(self.num_rear_consoles, 'wing', 1.0, False)
        }

    def change_configuration_scheme(self, new_scheme, num_front_consoles=None, num_rear_consoles=None):
        """
        Функция смены конфигурации (схемы расположения консолей)
        
        Параметры:
        new_scheme (str): новая схема конфигурации:
            - "default": стандартная (рули спереди, крылья сзади)
            - "no_rudders": отключение рулей (только крылья)
            - "no_rudders_wings": отключение рулей и крыльев (только фюзеляж)
            - "wings_front": крылья становятся передним поясом, рули - задним
            - "rudders_front": рули становятся передним поясом, крылья - задним
            - "custom": пользовательская конфигурация
        num_front_consoles (int): количество консолей в переднем поясе (опционально)
        num_rear_consoles (int): количество консолей в заднем поясе (опционально)
        """
        valid_schemes = ["default", "no_rudders", "no_rudders_wings", "wings_front", "rudders_front", "custom"]
        
        if new_scheme not in valid_schemes:
            print(f"Ошибка: недопустимая схема конфигурации '{new_scheme}'")
            print(f"Допустимые схемы: {valid_schemes}")
            return False
        
        old_scheme = self.config_scheme
        self.config_scheme = new_scheme
        
        # Обновляем количество консолей если указано
        if num_front_consoles is not None:
            self.num_front_consoles = num_front_consoles
        if num_rear_consoles is not None:
            self.num_rear_consoles = num_rear_consoles
        
        # Обновляем конфигурацию консолей
        self._initialize_consoles_configuration()
        
        # Обновляем количество консолей для отображения
        self.num_kons = len(self.consoles_config['II'])
        
        # Обновляем угол psi для отображения
        if self.consoles_config['II']:
            self.psi = np.rad2deg(self.consoles_config['II'][0]['psi'])
        else:
            self.psi = 45
            
        print(f"Конфигурация изменена: {old_scheme} -> {new_scheme}")
        print(f"Количество консолей в переднем поясе: {len(self.consoles_config['I'])}")
        print(f"Количество консолей в заднем поясе: {len(self.consoles_config['II'])}")
        
        return True

    def set_consoles_count(self, front_count, rear_count):
        """
        Установка количества консолей в поясах
        
        Параметры:
        front_count (int): количество консолей в переднем поясе
        rear_count (int): количество консолей в заднем поясе
        """
        self.num_front_consoles = front_count
        self.num_rear_consoles = rear_count
        
        # Переинициализируем конфигурацию
        self._initialize_consoles_configuration()
        
        # Обновляем количество консолей для отображения
        self.num_kons = len(self.consoles_config['II'])
        
        print(f"Установлено количество консолей: передний пояс={front_count}, задний пояс={rear_count}")

    def get_current_configuration_info(self):
        """Получить информацию о текущей конфигурации"""
        front_angles = [f"{np.rad2deg(console['psi']):.1f}°" for console in self.consoles_config['I']]
        rear_angles = [f"{np.rad2deg(console['psi']):.1f}°" for console in self.consoles_config['II']]
        
        # Определяем схему расположения
        if len(front_angles + rear_angles) == 4:
            arrangement = "X-образная"
        elif len(front_angles + rear_angles) == 6:
            arrangement = "шестилучевая" 
        elif len(front_angles + rear_angles) == 8:
            arrangement = "восьмилучевая"
        else:
            arrangement = f"{len(front_angles + rear_angles)}-лучевая"
        
        info = {
            'scheme': self.config_scheme,
            'arrangement': arrangement,
            'front_belt_count': len(self.consoles_config['I']),
            'rear_belt_count': len(self.consoles_config['II']),
            'front_belt_type': 'рули' if self.consoles_config['I'] and self.consoles_config['I'][0]['type'] == 'rudder' else 
                              'крылья' if self.consoles_config['I'] and self.consoles_config['I'][0]['type'] == 'wing' else 'отсутствует',
            'rear_belt_type': 'крылья' if self.consoles_config['II'] and self.consoles_config['II'][0]['type'] == 'wing' else 
                             'рули' if self.consoles_config['II'] and self.consoles_config['II'][0]['type'] == 'rudder' else 'отсутствует',
            'front_angles': front_angles,
            'rear_angles': rear_angles
        }
        return info

    def Y_yyy(self, x, nose_type):
        """Функция образующей в зависимости от типа носовой части"""
        Y = []
        for i in x:
            if i <= self.x_1 and nose_type in ["conical_with_sphere", "ogive_with_sphere"]:
                # Сферическая часть (только для носовых частей со сферическим затуплением)
                y = np.sqrt(self.r_zatup**2 - (i - self.r_zatup)**2)
                Y.append(y)
                
            elif i <= self.l_nos:
                # Основная часть носового обтекателя
                if nose_type in ["conical_with_sphere", "simple_conical"]:
                    # Коническая форма
                    if nose_type == "conical_with_sphere" and i > self.x_1:
                        y = (self.d_m/2) - (self.l_nos - i) * np.tan(self.theta_kon)
                    elif nose_type == "simple_conical":
                        y = (self.d_m/2) * (i / self.l_nos)
                    else:
                        y = 0
                        
                elif nose_type in ["ogive_with_sphere", "simple_ogive"]:
                    # Оживальная форма
                    if nose_type == "ogive_with_sphere" and i > self.x_1:
                        y = (self.d_m/2) - self.R + np.sqrt(self.R**2 - (i - self.l_nos)**2)
                    elif nose_type == "simple_ogive":
                        y = (self.d_m/2) - self.R + np.sqrt(self.R**2 - (i - self.l_nos)**2)
                    else:
                        y = 0
                        
                elif nose_type == "spherical":
                    # Сферическая форма
                    y = np.sqrt((self.d_m/2)**2 - (i - self.l_nos)**2)
                    
                Y.append(y)
                
            elif self.l_nos < i <= self.l_nos + self.l_cyl:
                # Цилиндрическая часть
                y = self.d_m/2
                Y.append(y)
                
            elif self.l_nos + self.l_cyl < i < self.l_nos + self.l_cyl + self.l_don:
                # Коническая кормовая часть
                y = (self.d_m/2) + (((self.d_k/2) - (self.d_m/2))/self.l_don) * (i - (self.l_nos + self.l_cyl))
                Y.append(y)
                
            elif i == self.l_nos + self.l_cyl + self.l_don:
                y = 0
                Y.append(y)
                
        return Y

    def Krylo(self, x):
        """Функция для построения крыла"""
        Kry = []
        l_all = self.l_nos + self.l_cyl + self.l_don
        x_nach = l_all - self.b_bort + ((np.tan(np.deg2rad(self.chi_pk)) * (self.d_op - self.d_k) / 2))
        x_nach_index = np.argmax(x >= x_nach)
        x_partial = x[x_nach_index:]
        
        for i in x_partial:
            if i == x_nach:
                y = (self.d_m/2) + (((self.d_k/2)-(self.d_m/2))/self.l_don)*(i - (self.l_nos + self.l_cyl))
                Kry.append(y)
            elif (l_all - self.b_k - x_nach) <= i <= l_all - self.b_k:    
                d_nach = (self.d_m/2) + (((self.d_k/2)-(self.d_m/2))/self.l_don)*(x_nach - (self.l_nos + self.l_cyl))
                y = d_nach - ((d_nach - (self.l_op/2)) / (l_all - self.b_k - x_nach)) * (i - x_nach)
                Kry.append(y)
            elif l_all - self.b_k <= i < l_all:
                y = (self.d_k/2) + (self.l_op - self.d_k) / 2
                Kry.append(y)
            elif i == l_all:
                y = self.d_k / 2
                Kry.append(y)
        return Kry
    
    def View_front(self, r, r_zatup, num_kons, psi):
        """Функция для построения вида спереди"""
        # psi больше не используется как единый угол, так как углы берутся из конфигурации
        theta = np.linspace(0, 2*np.pi, 100)
        z = r * np.cos(theta)
        y = r * np.sin(theta)
        
        z_small = r_zatup * np.cos(theta)
        y_small = r_zatup * np.sin(theta)
        
        plt.figure(figsize=(8, 8))
        plt.plot(z, y, 'b-', linewidth=2)
        if r_zatup > 0:
            plt.plot(z_small, y_small, 'b-', linewidth=2)
        plt.fill(z, y, "lightblue")
        plt.xlabel('Z')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        
        # Получаем реальные консоли из конфигурации
        all_consoles = self.consoles_config['I'] + self.consoles_config['II']
        
        if all_consoles:
            len_kons = (self.d_m / 2) + ((self.l_op / 2) - (self.d_m / 2))
            
            for console in all_consoles:
                angle = console['psi']
                start_z = r * np.cos(angle)
                start_y = r * np.sin(angle)
                end_z = len_kons * np.cos(angle)
                end_y = len_kons * np.sin(angle)
                
                # Определяем цвет в зависимости от типа консоли
                color = 'green' if console['type'] == 'rudder' else 'red'
                
                plt.plot([start_z, end_z], [start_y, end_y], color=color, linewidth=5, alpha=0.7,
                        label=f"{'Руль' if console['type'] == 'rudder' else 'Крыло'} ({np.rad2deg(angle):.0f}°)")
    
        plt.axhline(y=0, color='orange', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='orange', linestyle='-', alpha=0.3)
        plt.plot(0, 0, 'ko', markersize=6)
        
        # Добавляем легенду только если есть консоли
        if all_consoles:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f"Вид спереди ")
        plt.tight_layout()
        plt.show()

    def plot_fuselage_geometry(self):
        """Построение графиков геометрии фюзеляжа"""
        plt.figure(figsize=(15, 15))

        y_values = self.Y_yyy(self.x, self.nose_type)
        plt.plot(self.x, y_values, 'b-', linewidth=2)
        
        x_nach = self.l_nos + self.l_cyl + self.l_don - self.b_bort + ((np.tan(np.deg2rad(self.chi_pk)) * (self.d_op - self.d_k) / 2))
        x_nach_index = np.argmax(self.x >= x_nach)
        x_partial = self.x[x_nach_index:]
        
        if self.config_scheme == "no_rudders_wings":
            # Для случая только фюзеляжа используем исходные коэффициенты без масштабирования
            plt.fill(self.x, y_values, color='lightblue', alpha=0.7)
            plt.ylim(0, self.fuselage_geo.l_nos+self.fuselage_geo.l_cil+self.fuselage_geo.l_don)
            plt.xlabel('Длина, м')
            plt.ylabel('Радиус, м')
            plt.title(f'Геометрия фюзеляжа: {self.nose_name}')
            plt.legend()
            plt.xticks([self.x_1, self.l_nos,self.l_nos + self.l_cyl, self.l_nos + self.l_cyl + self.l_don - 
                        self.b_bort + ((np.tan(np.deg2rad(self.chi_pk)) * (self.d_op - self.d_k)) / 2),self.l_nos + 
                        self.l_cyl + self.l_don - self.b_k, self.l_nos + self.l_cyl + self.l_don],fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            plt.plot(x_partial, self.Krylo(self.x), 'green', linewidth=2)
    
            plt.fill(self.x, y_values, color='lightblue', alpha=0.7)
            plt.fill(x_partial, self.Krylo(self.x), color='red', alpha=0.8)

        # Вертикальные линии для ключевых точек
        # if self.x_1 > 0:
        #     plt.axvline(x=self.x_1, color='red', linestyle='--', alpha=0.7, label=f'x₁ = {self.x_1:.3f} м')
        # plt.axvline(x=self.l_nos, color='green', linestyle='--', alpha=0.7, label=f'Lнос = {self.l_nos:.3f} м')
        # plt.axvline(x=self.l_nos + self.l_cyl, color='blue', linestyle='--', alpha=0.7, label=f'Начало кормы')
        # plt.axvline(x=self.l_nos + self.l_cyl + self.l_don, color='purple', linestyle='--', alpha=0.7, label=f'Конец кормы')

            plt.ylim(0, self.fuselage_geo.l_nos+self.fuselage_geo.l_cil+self.fuselage_geo.l_don)
            plt.xlabel('Длина, м')
            plt.ylabel('Радиус, м')
            plt.title(f'Геометрия фюзеляжа: {self.nose_name}')
            plt.legend()
            plt.xticks([self.x_1, self.l_nos,self.l_nos + self.l_cyl, self.l_nos + self.l_cyl + self.l_don - 
                        self.b_bort + ((np.tan(np.deg2rad(self.chi_pk)) * (self.d_op - self.d_k)) / 2),
                        #self.l_nos + self.l_cyl + self.l_don - self.b_k, 
                        self.l_nos + self.l_cyl + self.l_don],fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Вид спереди
        self.View_front(self.d_m/2, self.r_zatup, self.num_kons, self.psi)

        # Вывод информации
        print("\n" + "="*60)
        print(f"ОПРЕДЕЛЕН ТИП НОСОВОЙ ЧАСТИ: {self.nose_name}")
        print("="*60)
        print(f"Диаметр миделя (d_m): {self.d_m:.3f} м")
        print(f"Длина носовой части (l_nos): {self.l_nos:.3f} м")
        print(f"Радиус сферического затупления (r_сф): {self.r_zatup:.3f} м")

        if self.nose_type in ["conical_with_sphere", "simple_conical"]:
            print(f"Угол полураствора конуса (θ_кон): {np.rad2deg(self.theta_kon):.1f}°")
        elif self.nose_type in ["ogive_with_sphere", "simple_ogive"]:
            print(f"Радиус оживала (R): {self.R:.3f} м")

        if self.x_1 > 0:
            print(f"Координата перехода сфера-основная часть (x₁): {self.x_1:.3f} м")
        print(f"Удлинение носовой части (λ_нос): {self.l_nos/self.d_m:.2f}")

    def phi_function(self, x):
        """Функция Лапласа-Гаусса согласно формуле (3.16)"""
        if isinstance(x, (int, float)) and x == 0:
            return 0
        t = np.linspace(0, x, 1000)
        integrand = np.exp(-t**2/2)
        integral = np.trapz(integrand, t)
        return (2/np.sqrt(2*np.pi)) * integral

    def calculate_isolated_console_cy(self, M, alpha, delta, console_params):
        """Расчет коэффициента нормальной силы изолированной консоли с использованием методов из классов"""
        if console_params['type'] == 'wing':
            # Используем метод из WingCalculator
            cy_alpha = self.wing_calc.calculate_cy_alpha(M)
            return self.wing_calc.calculate_cy_iz_p(M, alpha, cy_alpha)
        else:
            # Используем метод из RudderCalculator
            alpha_eff = self.rudder_calc.calculate_effective_angle(alpha)
            return self.rudder_calc.calculate_cn_iz_p(M, alpha_eff)
        
        
    
    def calculate_isolated_console_cx_interpolated(self, M, alpha, delta, console_params):
        """
        Расчет коэффициента продольной силы изолированной консоли с интерполяцией
        
        Параметры:
        M - число Маха
        alpha - угол атаки в радианах
        delta - угол отклонения рулей в радианах  
        console_params - параметры консоли
        
        Возвращает:
        cx - коэффициент продольной силы
        """
        
        if console_params['type'] == 'wing':
            calculator = self.wing_calc
            results = self.wing_calc.run_calculations()
        else:
            calculator = self.rudder_calc
            results = self.rudder_calc.run_calculations()
        
        # Преобразуем угол атаки в градусы для поиска в таблице
        alpha_deg = np.rad2deg(alpha)
        
        # Получаем доступные значения M и alpha из результатов
        available_M = sorted(results.keys())

        # Для первого доступного M получаем доступные углы атаки
        first_M = available_M[0]
        available_alpha_deg = sorted(results[first_M]['alpha_results'].keys())

        # Интерполяция по числу Маха и углу атаки
        return self._interpolate_2d_cx(M, alpha_deg, available_M, available_alpha_deg, results)
    
    def _interpolate_2d_cx(self, M_target, alpha_deg_target, M_list, alpha_list, results):
        """
        2D интерполяция коэффициента cx по числу Маха и углу атаки
        """
        
        # Находим ближайшие значения M для интерполяции
        M_lower, M_upper = self._find_bounding_values(M_target, M_list)
        
        # Если целевое M за пределами диапазона, используем экстраполяцию
        if M_lower is None:
            return self._interpolate_1d_alpha(alpha_deg_target, alpha_list, results[M_upper]['alpha_results'], 'cx_iz_p')
        elif M_upper is None:
            return self._interpolate_1d_alpha(alpha_deg_target, alpha_list, results[M_lower]['alpha_results'], 'cx_iz_p')
        
        # Интерполяция по alpha для нижнего M
        cx_lower = self._interpolate_1d_alpha(alpha_deg_target, alpha_list, results[M_lower]['alpha_results'], 'cx_iz_p')
        
        # Интерполяция по alpha для верхнего M  
        cx_upper = self._interpolate_1d_alpha(alpha_deg_target, alpha_list, results[M_upper]['alpha_results'], 'cx_iz_p')
        
        # Линейная интерполяция по M
        if M_upper != M_lower:
            weight_M = (M_target - M_lower) / (M_upper - M_lower)
            cx_interpolated = cx_lower + weight_M * (cx_upper - cx_lower)
        else:
            cx_interpolated = cx_lower
        
        return cx_interpolated
    
    def _interpolate_1d_alpha(self, alpha_target, alpha_list, alpha_results, param_name):
        """
        1D интерполяция по углу атаки
        """
        # Находим ближайшие значения alpha
        alpha_lower, alpha_upper = self._find_bounding_values(alpha_target, alpha_list)
        
        # Если целевой угол за пределами диапазона, используем экстраполяцию
        if alpha_lower is None:
            return alpha_results[alpha_upper][param_name]
        elif alpha_upper is None:
            return alpha_results[alpha_lower][param_name]
        
        # Получаем значения параметра для bounding angles
        value_lower = alpha_results[alpha_lower][param_name]
        value_upper = alpha_results[alpha_upper][param_name]
        
        # Линейная интерполяция
        if alpha_upper != alpha_lower:
            weight_alpha = (alpha_target - alpha_lower) / (alpha_upper - alpha_lower)
            value_interpolated = value_lower + weight_alpha * (value_upper - value_lower)
        else:
            value_interpolated = value_lower
        
        return value_interpolated
    
    def _find_bounding_values(self, target_value, value_list):
        """
        Находит ближайшие нижнее и верхнее значения для интерполяции
        """
        if not value_list:
            return None, None
        
        sorted_values = sorted(value_list)
        
        # Если target меньше минимального значения
        if target_value <= sorted_values[0]:
            return None, sorted_values[0]
        
        # Если target больше максимального значения  
        if target_value >= sorted_values[-1]:
            return sorted_values[-1], None
        
        # Находим bounding values
        for i in range(len(sorted_values) - 1):
            if sorted_values[i] <= target_value <= sorted_values[i + 1]:
                return sorted_values[i], sorted_values[i + 1]
        
        return sorted_values[0], sorted_values[-1]

    
    # Аналогичная функция для координаты центра давления
    def calculate_isolated_console_xd_interpolated(self, M, alpha, delta, console_params):
        """
        Расчет координаты центра давления с интерполяцией
        """
        
        if console_params['type'] == 'wing':
            results = self.wing_calc.run_calculations()
        else:
            results = self.rudder_calc.run_calculations()
        
        alpha_deg = np.rad2deg(alpha)
        available_M = sorted(results.keys())
        
        if not available_M:
            return self._estimate_xd_fallback(console_params, M, alpha_deg)
        
        first_M = available_M[0]
        available_alpha_deg = sorted(results[first_M]['alpha_results'].keys())
        
        if not available_alpha_deg:
            return self._estimate_xd_fallback(console_params, M, alpha_deg)
        
        return self._interpolate_2d_xd(M, alpha_deg, available_M, available_alpha_deg, results)
    
    def _interpolate_2d_xd(self, M_target, alpha_deg_target, M_list, alpha_list, results):
        """
        2D интерполяция координаты центра давления
        """
        M_lower, M_upper = self._find_bounding_values(M_target, M_list)
        
        if M_lower is None:
            return self._interpolate_1d_alpha(alpha_deg_target, alpha_list, results[M_upper]['alpha_results'], 'x_d_iz_p')
        elif M_upper is None:
            return self._interpolate_1d_alpha(alpha_deg_target, alpha_list, results[M_lower]['alpha_results'], 'x_d_iz_p')
        
        xd_lower = self._interpolate_1d_alpha(alpha_deg_target, alpha_list, results[M_lower]['alpha_results'], 'x_d_iz_p')
        xd_upper = self._interpolate_1d_alpha(alpha_deg_target, alpha_list, results[M_upper]['alpha_results'], 'x_d_iz_p')
        
        if M_upper != M_lower:
            weight_M = (M_target - M_lower) / (M_upper - M_lower)
            xd_interpolated = xd_lower + weight_M * (xd_upper - xd_lower)
        else:
            xd_interpolated = xd_lower
        
        return xd_interpolated
      
    # Обновленные версии оригинальных функций с интерполяцией
    def calculate_isolated_console_cx(self, M, alpha, delta, console_params):
        """Расчет коэффициента продольной силы с интерполяцией"""
        return self.calculate_isolated_console_cx_interpolated(M, alpha, delta, console_params)
    
    def calculate_isolated_console_xd(self, M, alpha, delta, console_params):
        """Расчет координаты центра давления с интерполяцией"""
        return self.calculate_isolated_console_xd_interpolated(M, alpha, delta, console_params)
        
    """
    def calculate_isolated_console_cx(self, M, alpha, delta, console_params):
        
        if console_params['type'] == 'wing':
            # Для крыла используем расчет из WingCalculator
            wing_results = self.wing_calc.run_calculations()
            if M in wing_results and round(np.rad2deg(alpha)) in wing_results[M]['alpha_results']:
                return wing_results[M]['alpha_results'][round(np.rad2deg(alpha))]['cx_iz_p']
            else:
                return 0.01  # значение по умолчанию
        else:
            # Для руля используем расчет из RudderCalculator
            rudder_results = self.rudder_calc.run_calculations()
            if M in rudder_results and round(np.rad2deg(alpha)) in rudder_results[M]['alpha_results']:
                return rudder_results[M]['alpha_results'][round(np.rad2deg(alpha))]['cx_iz_p']
            else:
                return 0.005  # значение по умолчанию

    def calculate_isolated_console_xd(self, M, alpha, delta, console_params):
       
        if console_params['type'] == 'wing':
            # Для крыла
            wing_results = self.wing_calc.run_calculations()
            if M in wing_results and round(np.rad2deg(alpha)) in wing_results[M]['alpha_results']:
                return wing_results[M]['alpha_results'][round(np.rad2deg(alpha))]['x_d_iz_p']
            else:
                return console_params['xA'] + console_params['bA']/2  # приблизительное значение
        else:
            # Для руля
            rudder_results = self.rudder_calc.run_calculations()
            if M in rudder_results and round(np.rad2deg(alpha)) in rudder_results[M]['alpha_results']:
                return rudder_results[M]['alpha_results'][round(np.rad2deg(alpha))]['x_d_iz_p']
            else:
                return console_params['xA'] + console_params['bA']/2  # приблизительное значение
    """
    def calculate_isolated_console_x_alpha_F(self, M, console_params):
        """Расчет координаты фокуса по углу атаки изолированной консоли"""
        if console_params['type'] == 'wing':
            return self.wing_calc.calculate_focus_position(M)
        else:
            return self.rudder_calc.calculate_focus_position(M)
    
    def calculate_interference_coefficients(self, M, D, l1k, zeta, x6, b6, L_phi):
        """Расчет коэффициентов интерференции по методике (раздел 3.1.2)"""
        
        # Относительный диаметр фюзеляжа (формула 3.6)
        D_rel = D / (2 * l1k + D)
        
        # Теоретические коэффициенты (формулы 3.4, 3.5)
        k_aa_teor = (1 + 0.41 * D_rel)**2
        K_aa_teor = (1 + D_rel)**2
        
        # Поправка на сужение (формула 3.7)
        k_zeta = (1 + 3*D_rel - zeta*D_rel*(1-D_rel)) / (1 + D_rel)**2
        
        # Поправка на пограничный слой (формулы 3.8, 3.9)
        Re_ps = (self.a_sound * M * (x6 + 0.5*b6)) / atmo.nu(self.H)
        delta_ps = ((x6 + 0.5*b6) / (D_rel * Re_ps**0.2) * 
                   0.093*(1 + 0.4*M + 0.147*M**2 - 0.006*M**3))
        
        k_ps_aa = ((1 - (2*D_rel**2)/(1-D_rel**2)*delta_ps) * 
                  (1 - (D_rel/(1-D_rel)) * ((1-zeta)/(1+zeta))*delta_ps))
        
        # Поправка на число Маха (упрощенная замена AeroBDSM)
        k_M = 1.0
        if isinstance(M, (int, float)) and M > 1.0:
            k_M = 0.9 + 0.1 * (2.0 - M)
        
        # Поправка на длину предшествующей части (формула 3.10)
        k_pre = 0.6 + 0.4*(1 - np.exp(-0.5*(x6 + 0.5*b6)/D))
        
        # Коэффициент k_aa с поправками (формула 3.11)
        k_aa = k_aa_teor * k_zeta * k_ps_aa * k_M * k_pre
        
        # Поправка на последующую часть (формулы 3.12-3.16)
        if isinstance(M, (int, float)) and M <= 1:
            k_pos = 1.0
            k_pos1 = 1.0
        else:
            # Расчет для сверхзвука
            c_q = (4 + zeta) * (1 + 8*D**2)
            L_M = 0.5 * np.pi * D * np.sqrt(M**2 - 1)
            b6_rel = b6 / L_M
            L_pos_rel = (L_phi - x6 - b6) / L_M
            
            k_pos = 1 - (np.sqrt(np.pi)/(2*b6_rel*np.sqrt(c_q)) * 
                        (self.phi_function((b6_rel + L_pos_rel)*np.sqrt(2*c_q)) - 
                         self.phi_function(L_pos_rel*np.sqrt(2*c_q))))
            
            k_pos1 = 1 - (np.exp(-c_q * L_pos_rel**2)/(b6_rel**2 * c_q) + 
                         np.exp(-c_q * (b6_rel + L_pos_rel)**2)/(b6_rel**2 * c_q) + 
                         np.sqrt(np.pi)/(b6_rel * np.sqrt(c_q)) * 
                         self.phi_function(L_pos_rel * np.sqrt(2*c_q)))
        
        # Коэффициент K_aa с поправками (формула 3.17)
        K_aa = (k_aa_teor + (K_aa_teor - k_aa_teor)*k_pos) * k_zeta * k_ps_aa * k_M * k_pre
        
        # Коэффициенты для случая "80" (формулы 3.18, 3.19)
        k_delta_teor = (k_aa_teor**2) / K_aa_teor
        K_delta_teor = k_aa_teor
        
        # Поправка на пограничный слой для случая "80" (формула 3.20)
        k_ps_80 = ((1 - D_rel*(1 + delta_ps)) * 
                  (1 - ((1-zeta)/(1+zeta-2*zeta*D_rel))*D_rel*(1+delta_ps)) / 
                  (1 - ((1-zeta)/(1+zeta-2*zeta*D_rel))*D_rel*(1-D_rel)))
        
        # Коэффициенты с поправками (формулы 3.21, 3.22)
        k_delta = k_delta_teor * k_zeta * k_ps_80 * k_M
        K_delta = (k_delta_teor + (K_delta_teor - k_delta_teor)*k_pos) * k_zeta * k_ps_80 * k_M
        
        return {
            'k_aa': k_aa,
            'K_aa': K_aa,
            'k_delta': k_delta,
            'K_delta': K_delta,
            'k_pos': k_pos,
            'k_pos1': k_pos1
        }
    
    def calculate_console_normal_force(self, M, alpha, delta, console_params, interference_coeffs):
        """Расчет нормальной силы консоли с учетом интерференции (раздел 3.1.3)"""
        
        # Учет отражения
        alpha_eff = alpha
        delta_eff = delta
        if console_params['IsReflected'] == 1:
            alpha_eff = -alpha
            delta_eff = -delta
        
        # Коэффициент нормальной силы изолированной консоли
        cy_iz_p = self.calculate_isolated_console_cy(M, alpha_eff, delta_eff, console_params)
        
        # Эффективные углы с учетом интерференции
        alpha_eff_int = interference_coeffs['k_aa'] * alpha_eff
        delta_eff_int = interference_coeffs['k_delta'] * delta_eff
        
        # Коэффициент нормальной силы с учетом интерференции на консоли
        cy_p = self.calculate_isolated_console_cy(M, alpha_eff_int, delta_eff_int, console_params)
        
        # Дополнительная сила на консоли (формулы 3.26, 3.29)
        cy_in = cy_p - cy_iz_p
        
        # Дополнительная сила на фюзеляже (формула 3.30)
        cy_if = (interference_coeffs['K_aa']/interference_coeffs['k_aa'] - 1) * cy_p
        
        # Суммарный коэффициент нормальной силы (формула 3.31)
        cy_k = cy_iz_p + cy_in + cy_if
        
        # Учет отражения для итогового коэффициента
        if console_params['IsReflected'] == 1:
            cy_k = -cy_k
        
        return cy_k, cy_p, cy_iz_p
    
    def calculate_console_longitudinal_force(self, M, alpha, delta, console_params):
        """Расчет продольной силы консоли (раздел 3.1.4)"""
        # Используем метод из соответствующего класса
        return self.calculate_isolated_console_cx(M, alpha, delta, console_params)
    
    def calculate_console_pressure_center(self, M, alpha, delta, console_params, 
                                        cy_k, cy_p, cy_iz_p, interference_coeffs):
        """Расчет координат центра давления консоли (разделы 3.1.5, 3.1.6)"""
        
        # Продольная координата ЦД изолированной консоли (формулы 3.36, 3.38)
        xd_iz_p = console_params['x6'] + self.calculate_isolated_console_xd(M, alpha, delta, console_params)
        
        # Координата приложения дополнительной силы на консоли (формула 3.39)
        # Корректный расчет согласно методичке
        try:
            xi_D = AeroBDSM.get_xi_D(console_params['zeta'])
            if hasattr(xi_D, 'Value'):
                xi_D = xi_D.Value
        except:
            xi_D = 0.15
            
        # Используем x_alpha_F_iz_p из соответствующего класса
        x_Fa_iz_p = self.calculate_isolated_console_x_alpha_F(M, console_params)
        
        xd_in = console_params['x6'] + x_Fa_iz_p - xi_D * console_params['l1k'] * np.tan(console_params['X_05'])
        
        # Координата приложения дополнительной силы на фюзеляже (формула 3.41)
        x_F6 = x_Fa_iz_p / console_params['bA'] + 0.02 * console_params['lambda_k'] * np.tan(console_params['X_05'])
        xd_if = console_params['x6'] + console_params['b6'] * x_F6 * interference_coeffs['k_pos'] * interference_coeffs['k_pos1']
        
        # Продольная координата ЦД (формула 3.34, 3.45)
        if abs(cy_k) > 1e-10:
            xd_k = (cy_iz_p*xd_iz_p + (cy_p - cy_iz_p)*xd_in + 
                   ((interference_coeffs['K_aa']/interference_coeffs['k_aa'] - 1)*cy_p)*xd_if) / cy_k
        else:
            xd_k = (xd_iz_p + (interference_coeffs['k_aa']-1)*xd_in + 
                   (interference_coeffs['K_aa']-interference_coeffs['k_aa'])*xd_if) / interference_coeffs['K_aa']
        
        # Поперечная координата ЦД (формулы 3.48, 3.50, 3.51, 3.53)
        zd_iz_p = self.fuselage_geo.d_m/2 + console_params['l1k']/3
        
        try:
            xi_D = AeroBDSM.get_xi_D(console_params['zeta'])
            if hasattr(xi_D, 'Value'):
                xi_D = xi_D.Value
        except:
            xi_D = 0.15
            
        zd_in = zd_iz_p - xi_D * console_params['l1k']
        zd_if = self.fuselage_geo.d_m/4
        
        if abs(cy_k) > 1e-10:
            zd_k = (cy_iz_p*zd_iz_p + (cy_p - cy_iz_p)*zd_in + 
                   ((interference_coeffs['K_aa']/interference_coeffs['k_aa'] - 1)*cy_p)*zd_if) / cy_k
        else:
            zd_k = (zd_iz_p + (interference_coeffs['k_aa']-1)*zd_in + 
                   (interference_coeffs['K_aa']-interference_coeffs['k_aa'])*zd_if) / interference_coeffs['K_aa']
        
        return xd_k, zd_k
    
    def calculate_console_aerodynamics(self, M, alpha, delta, console_params, belt_type, interference_coeffs):
        """Расчет АДХ одной консоли с учетом интерференции"""
        
        # Нормальная сила
        cy_k, cy_p, cy_iz_p = self.calculate_console_normal_force(
            M, alpha, delta, console_params, interference_coeffs)
        
        # Продольная сила
        cx_k = self.calculate_console_longitudinal_force(M, alpha, delta, console_params)
        
        # Центр давления
        xd_k, zd_k = self.calculate_console_pressure_center(
            M, alpha, delta, console_params, cy_k, cy_p, cy_iz_p, interference_coeffs)
        
        # Сила, нормальная к консоли (раздел 3.1.7)
        cn_n = cy_p
        if console_params['IsReflected'] == 1:
            cn_n = -cn_n
        
        # Приведение к характерной площади ЛА
        S_ratio = console_params['S1k'] / self.S_ref
        
        # Учет торможения потока
        if belt_type == 'I':
            k_M = 0.95
        else:
            k_M = 0.90
        
        cy_k *= k_M * S_ratio
        cx_k *= k_M * S_ratio
        cn_n *= k_M * S_ratio
        
        return {
            'cy_k': cy_k,
            'cx_k': cx_k,
            'xd_k': xd_k,
            'zd_k': zd_k,
            'cn_n': cn_n
        }
    
    def calculate_flow_perturbation(self, M, front_console, rear_console, front_cn_n):
        """Расчет возмущения потока от передней консоли (раздел 3.2)"""
        # Упрощенный расчет скоса потока
        # Расстояние между консолями
        x_distance = rear_console['x6'] - (front_console['x6'] + front_console['b6'])
        
        if x_distance <= 0:
            return 0, 1.0  # нет возмущения
        
        # Угол скоса потока (упрощенная формула)
        epsilon = front_cn_n * front_console['S1k'] / (2 * np.pi * x_distance * self.S_ref)
        
        # Коэффициент торможения
        k_M_perturbation = 0.95 if isinstance(M, (int, float)) and M > 1.0 else 1.0
        
        return epsilon, k_M_perturbation
    
    def calculate_cxa_cya_from_cx_cy(self, cx, cy, alpha):
        """Расчет коэффициентов Cxa и Cya по формулам (2.15) и (2.16)"""
        if abs(np.cos(alpha)) < 1e-10:
            return cx, cy
        
        # Формула (2.15): c_{x,alpha} = (c_x - c_y * sin(alpha)) / cos(alpha)
        cxa = (cx - cy * np.sin(alpha)) / np.cos(alpha)
        
        # Формула (2.16): c_{y,alpha} = (c_y - c_x * sin(alpha)) / cos(alpha)
        cya = (cy - cx * np.sin(alpha)) / np.cos(alpha)
        
        return cxa, cya
    
    def calculate_aircraft_aerodynamics(self, M, alpha_n, phi_alpha, delta_I, delta_II):
        """Расчет суммарных АДХ летательного аппарата (раздел 3.2)"""
        
        # АДХ фюзеляжа (формулы 3.59-3.68)
        fuselage_results = self.fuselage_calc.run_calculations()
        
        # Находим ближайшее число Маха из результатов фюзеляжа
        available_M = list(fuselage_results.keys())
        M_fuselage = min(available_M, key=lambda x: abs(x - M))
        
        # Находим ближайший угол атаки
        alpha_deg = round(np.rad2deg(alpha_n))
        available_alpha = list(fuselage_results[M_fuselage]['alpha_results'].keys())
        alpha_deg = min(available_alpha, key=lambda x: abs(x - alpha_deg))
        
        fuselage_data = fuselage_results[M_fuselage]['alpha_results'][alpha_deg]
        
        if self.config_scheme == "no_rudders_wings":
            # Для случая только фюзеляжа используем исходные коэффициенты без масштабирования
            cx_f = fuselage_data['cx_iz_f']
            cn_f = fuselage_data['cy_f']
        else:
            # Для случаев с консолями масштабируем на характерную площадь
            cx_f = fuselage_data['cx_iz_f'] * self.fuselage_geo.S_m / self.S_ref
            cn_f = fuselage_data['cy_f'] * self.fuselage_geo.S_m / self.S_ref
        
        # Проекция на оси ЛА (формулы 3.61, 3.62)
        cy_f = cn_f * np.cos(phi_alpha)
        cz_f = -cn_f * np.sin(phi_alpha)
        
        # Центр давления фюзеляжа
        xd_f = self.fuselage_calc.calculate_x_d(M_fuselage,alpha_n)
        # Моменты от фюзеляжа
        x0 = self.x0_rel
        mx_f = (cy_f * 0 + cz_f * 0) / self.L_ref  # yd_f = zd_f = 0
        my_f = -cz_f * (x0 - xd_f) / self.L_ref
        mz_f = cy_f * (x0 - xd_f) / self.L_ref
        
        # Если конфигурация "no_rudders_wings", возвращаем только АДХ фюзеляжа
        if self.config_scheme == "no_rudders_wings":
            # Расчет коэффициентов Cxa и Cya по формулам (2.15) и (2.16)
            cxa_total, cya_total = self.calculate_cxa_cya_from_cx_cy(cx_f, cy_f, alpha_n)
         
            return {
                'cx': cx_f,
                'cy': cy_f,
                'cz': cz_f,
                'cxa': cxa_total,
                'cya': cya_total,
                'mx': mx_f,
                'my': my_f,
                'mz': mz_f,
                'xd': xd_f
            }
        
        # Остальной код для случаев с консолями...
        cx_total, cy_total, cz_total = cx_f, cy_f, cz_f
        mx_total, my_total, mz_total = mx_f, my_f, mz_f
        
 
        # Хранение cn_n для передних консолей (для расчета скоса потока)
        front_console_cn = {}
        
        # Расчет для переднего пояса консолей - ТОЛЬКО ЕСЛИ ЕСТЬ КОНСОЛИ
        if self.consoles_config['I']:
            for i, console in enumerate(self.consoles_config['I']):
                # Угол установки консоли (формула 3.72)
                psi_alpha = console['psi'] - phi_alpha
                alpha_k = alpha_n * np.cos(psi_alpha)  # формула 3.73
                
                # Интерференционные коэффициенты
                D = self.fuselage_geo.d_m
                interference_coeffs = self.calculate_interference_coefficients(
                    M, D, console['l1k'], console['zeta'], console['x6'], 
                    console['b6'], self.fuselage_geo.l_f)
                
                # Расчет АДХ консоли
                console_results = self.calculate_console_aerodynamics(
                    M, alpha_k, delta_I, console, 'I', interference_coeffs)
                
                # Сохраняем cn_n для расчета скоса потока
                front_console_cn[i] = console_results['cn_n']
                
                # Проекция на оси ЛА (формулы 3.79-3.81)
                cx_console = console_results['cx_k']
                cy_console = console_results['cy_k'] * np.cos(console['psi'])
                cz_console = console_results['cy_k'] * np.sin(console['psi'])
                
                # Координаты ЦД в СК ЛА (формулы 3.82-3.84)
                xd_console = console_results['xd_k']
                yd_console = console_results['zd_k'] * np.sin(console['psi'])
                zd_console = console_results['zd_k'] * np.cos(console['psi'])
                
                # Моменты от консоли (формулы 3.85-3.87)
                x0_k =   self.wing_geo.xA + self.wing_geo.bA/2#!!!!!!
                mx_console = (-cy_console * zd_console + cz_console * yd_console) / self.L_ref
                my_console = -cz_console * (x0_k - xd_console) / self.L_ref
                mz_console = cy_console * (x0_k - xd_console) / self.L_ref
                
                # Суммирование
                cx_total += cx_console
                cy_total += cy_console
                cz_total += cz_console
                mx_total += mx_console
                my_total += my_console
                mz_total += mz_console
         
        # Расчет для заднего пояса консолей с учетом скоса потока - ТОЛЬКО ЕСЛИ ЕСТЬ КОНСОЛИ
        if self.consoles_config['II']:
            for i, console in enumerate(self.consoles_config['II']):
                # Угол установки консоли
                psi_alpha = console['psi'] - phi_alpha
                alpha_k_geom = alpha_n * np.cos(psi_alpha)
                
                # Расчет скоса потока только если есть передние консоли
                epsilon = 0
                k_M_perturbation = 1.0
                
                if self.consoles_config['I']:
                    # Находим ближайшую переднюю консоль
                    min_angle_diff = float('inf')
                    nearest_front_idx = 0
                    for j, front_console in enumerate(self.consoles_config['I']):
                        angle_diff = abs(front_console['psi'] - console['psi'])
                        if angle_diff < min_angle_diff:
                            min_angle_diff = angle_diff
                            nearest_front_idx = j
                    
                    # Расчет скоса потока
                    epsilon, k_M_perturbation = self.calculate_flow_perturbation(
                        M, self.consoles_config['I'][nearest_front_idx], console, 
                        front_console_cn[nearest_front_idx])
                
                # Эффективный угол атаки с учетом скоса потока (формула 3.104)
                alpha_k = alpha_k_geom - epsilon
                
                # Интерференционные коэффициенты
                D = self.fuselage_geo.d_m
                interference_coeffs = self.calculate_interference_coefficients(
                    M * k_M_perturbation, D, console['l1k'], console['zeta'], 
                    console['x6'], console['b6'], self.fuselage_geo.l_f)
                
                # Расчет АДХ консоли
                console_results = self.calculate_console_aerodynamics(
                    M * k_M_perturbation, alpha_k, delta_II, console, 'II', interference_coeffs)
                
                # Проекция на оси ЛА
                cx_console = console_results['cx_k']
                cy_console = console_results['cy_k'] * np.cos(console['psi'])
                cz_console = console_results['cy_k'] * np.sin(console['psi'])
                
                # Координаты ЦД в СК ЛА
                xd_console = console_results['xd_k']
                yd_console = console_results['zd_k'] * np.sin(console['psi'])
                zd_console = console_results['zd_k'] * np.cos(console['psi'])
                
                # Моменты от консоли
                x0_k =   self.wing_geo.xA + self.wing_geo.bA/2#!!!!!!
                mx_console = (-cy_console * zd_console + cz_console * yd_console) / self.L_ref
                my_console = -cz_console * (x0_k - xd_console+1.295) / self.L_ref
                mz_console = cy_console * (x0_k - (xd_console+1.295)) / self.L_ref
                
                # Суммирование
                cx_total += cx_console
                cy_total += cy_console
                cz_total += cz_console
                mx_total += mx_console
                my_total += my_console
                mz_total += mz_console
        
        # Координата ЦД ЛА (формула 3.130)
        force_magnitude = np.sqrt(cy_total**2 + cz_total**2)
        moment_magnitude = np.sqrt(mz_total**2 + my_total**2)
        
        if force_magnitude > 0:#1e-10:
            xd_LA = x0 + moment_magnitude * self.L_ref / force_magnitude
        else:
            xd_LA =  x0#!!!!!!!!!!!!! 
        # Расчет коэффициентов Cxa и Cya по формулам (2.15) и (2.16)
        cxa_total, cya_total = self.calculate_cxa_cya_from_cx_cy(cx_total, cy_total, alpha_n)
        return {
            'cx': cx_total,
            'cy': cy_total,
            'cz': cz_total,
            'cxa': cxa_total,
            'cya': cya_total,
            'mx': mx_total,
            'my': my_total,
            'mz': mz_total,
            'xd': xd_LA
        }
        
    def run_complete_analysis(self):
        """Полный анализ АДХ летательного аппарата"""
        results = {}
        
        print("Расчет АДХ летательного аппарата...")
        print(f"Конфигурация: {self.get_current_configuration_info()}")
        print(f"Количество точек по числу Маха: {len(self.M_list)}")
        print(f"Количество точек по углу атаки: {len(self.alpha_list)}")
        print(f"Количество точек по углу отклонения рулей: {len(self.delta_list)}")
        print(f"Общее количество расчетных точек: {len(self.M_list) * len(self.alpha_list) * len(self.delta_list)}")
        
        for M in self.M_list:
            results[M] = {}
            
            for alpha in self.alpha_list:
                alpha_deg = round(np.rad2deg(alpha))
                results[M][alpha_deg] = {}
                
                for delta in self.delta_list:
                    delta_deg = round(np.rad2deg(delta))
                    results[M][alpha_deg][delta_deg] = {}
                    
                    for phi_alpha in self.phi_alpha_list:
                        try:
                            # Используем одинаковый угол отклонения для обоих поясов
                            delta_I = delta
                            delta_II = delta
                            
                            aircraft_results = self.calculate_aircraft_aerodynamics(
                                M, alpha, phi_alpha, delta_I, delta_II)
                            
                            results[M][alpha_deg][delta_deg][phi_alpha] = aircraft_results
                            
                        except Exception as e:
                            print(f"Ошибка при M={M:.2f}, α={alpha_deg}°, δ={delta_deg}°: {e}")
                            # Заполняем нулевыми значениями при ошибке
                            results[M][alpha_deg][delta_deg][phi_alpha] = {
                                'cx': 0, 'cy': 0, 'cz': 0, 'cxa': 0, 'cya': 0,
                                'mx': 0, 'my': 0, 'mz': 0, 'xd': self.x0_rel 
                            }
        
        return results

    def plot_aircraft_characteristics(self, results):
        """Построение графиков суммарных АДХ ЛА с улучшенной организацией"""
        
        if not results:
            print("Нет данных для построения графиков")
            return
        
        # 1. Графики от числа Маха для разных углов атаки (при delta=0)
        self._plot_vs_mach_for_different_alphas(results)
        
        # 2. Графики от угла атаки для разных чисел Маха (при delta=0)
        self._plot_vs_alpha_for_different_mach(results)
        
        # 3. Поляра (Cya от Cxa) (при delta=0)
        self._plot_polar_diagram_improved(results)
        
        return results

    def _plot_vs_mach_for_different_alphas(self, results):
        """Графики от числа Маха для разных углов атаки (при delta=0)"""
        M_values = sorted(results.keys())
        
        # Выбираем несколько характерных углов атаки
        all_alphas = sorted(set(alpha for M in results for alpha in results[M].keys()))
        selected_alphas = self._select_representative_values(all_alphas, 5)
        
        if not selected_alphas:
            return
        
        # Создаем графики для каждого параметра
        parameters = [
            ('cx', 'Cx', 'Коэффициент продольной силы', 'red'),
            ('cy', 'Cy', 'Коэффициент нормальной силы', 'blue'),
            ('cz', 'Cz', 'Коэффициент поперечной силы', 'green'),
            ('cxa', 'Cxa', 'Коэффициент Cxa', 'orange'),
            ('cya', 'Cya', 'Коэффициент Cya', 'purple'),
            ('mz', 'Mz', 'Момент тангажа', 'pink'),
            ('my', 'My', 'Момент рыскания', 'brown'),
            ('mx', 'Mx', 'Момент крена', 'gray'),
            ('xd', 'xd', 'Координата центра давления, м', 'black')
        ]
        
        # Создаем subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Аэродинамические характеристики ЛА от числа Маха для разных углов атаки (δ=0°) ', fontsize=16)
        
        axes = axes.flatten()
        
        for idx, (param, ylabel, title, base_color) in enumerate(parameters[:9]):
            ax = axes[idx]
            
            # Создаем цветовую карту для углов атаки
            colors = plt.cm.viridis(np.linspace(0, 1, len(selected_alphas)))
            
            for alpha, color in zip(selected_alphas, colors):
                M_data = []
                param_data = []
                
                for M in M_values:
                    if alpha in results[M] and 0 in results[M][alpha]:  # delta=0
                        data = results[M][alpha][0][0]  # delta=0, phi_alpha=0
                        M_data.append(M)
                        param_data.append(data[param])
                
                if M_data:
                    ax.plot(M_data, param_data, color=color, marker='o', 
                           linewidth=2, markersize=4, label=f'α={alpha}°')
            
            ax.set_xlabel('Число Маха')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Убираем лишние subplots
        for idx in range(len(parameters), 9):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def _plot_vs_alpha_for_different_mach(self, results):
        """Графики от угла атаки для разных чисел Маха (при delta=0)"""
        # Находим все доступные углы атаки
        first_M = next(iter(results.keys()))
        alpha_values = sorted(results[first_M].keys())
        
        # Выбираем несколько характерных чисел Маха
        M_values = sorted(results.keys())
        selected_M = self._select_representative_values(M_values, 5)
        
        if not selected_M:
            return
        
        # Создаем графики для каждого параметра
        parameters = [
            ('cx', 'Cx', 'Коэффициент продольной силы', 'red'),
            ('cy', 'Cy', 'Коэффициент нормальной силы', 'blue'),
            ('cz', 'Cz', 'Коэффициент поперечной силы', 'green'),
            ('cxa', 'Cxa', 'Коэффициент Cxa', 'orange'),
            ('cya', 'Cya', 'Коэффициент Cya', 'purple'),
            ('mz', 'Mz', 'Момент тангажа', 'pink'),
            ('my', 'My', 'Момент рыскания', 'brown'),
            ('mx', 'Mx', 'Момент крена', 'gray'),
            ('xd', 'xd', 'Координата центра давления, м', 'black')
        ]
         
        # Создаем subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Аэродинамические характеристики ЛА от угла атаки для разных чисел Маха (δ=0°) ', fontsize=16)
        
        axes = axes.flatten()
        
        for idx, (param, ylabel, title, base_color) in enumerate(parameters[:9]):
            ax = axes[idx]
            
            # Создаем цветовую карту для чисел Маха
            colors = plt.cm.plasma(np.linspace(0, 1, len(selected_M)))
            
            for M, color in zip(selected_M, colors):
                alpha_data = []
                param_data = []
                
                for alpha in alpha_values:
                    if alpha in results[M] and 0 in results[M][alpha]:  # delta=0
                        data = results[M][alpha][0][0]  # delta=0, phi_alpha=0
                        alpha_data.append(alpha)
                        param_data.append(data[param])
                
                if alpha_data:
                    ax.plot(alpha_data, param_data, color=color, marker='s', 
                           linewidth=2, markersize=4, label=f'M={M:.2f}')
            
            ax.set_xlabel('Угол атаки, °')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Убираем лишние subplots
        for idx in range(len(parameters), 9):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def _plot_polar_diagram_improved(self, results):
        """Улучшенный график поляры (Cya от Cxa) - только для разных углов атаки (при delta=0)"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # График: Поляра для разных углов атаки (фиксированное число Маха, delta=0)
        self._plot_polar_by_alpha(results, ax)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_polar_by_alpha(self, results, ax):
        """Поляра для разных углов атаки (при delta=0)"""
        # Фиксируем число Маха (ближайшее к 1.0)
        target_M = 2.0
        available_M = sorted(results.keys())
        if not available_M:
            return
            
        M_ref = min(available_M, key=lambda x: abs(x - target_M))
        
        alpha_values = sorted(results[M_ref].keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))
        
        cxa_values = []
        cya_values = []
        
        for alpha, color in zip(alpha_values, colors):
            if alpha in results[M_ref] and 0 in results[M_ref][alpha]:  # delta=0
                data = results[M_ref][alpha][0][0]  # delta=0, phi_alpha=0
                cxa_values.append(data['cxa'])
                cya_values.append(data['cya'])
                ax.scatter(data['cxa'], data['cya'], color=color, s=60, 
                          label=f'α={alpha}°', alpha=0.7)
        
        # Соединяем точки линиями в порядке возрастания угла атаки
        if len(cxa_values) > 1:
            points = sorted(zip(cxa_values, cya_values, alpha_values), key=lambda x: x[2])
            cxa_sorted, cya_sorted, alphas_sorted = zip(*points)
            ax.plot(cxa_sorted, cya_sorted, 'k-', alpha=0.3, linewidth=1)
            
            # Аннотации для крайних точек
            if len(alphas_sorted) >= 2:
                ax.annotate(f'α={alphas_sorted[0]}°', (cxa_sorted[0], cya_sorted[0]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.annotate(f'α={alphas_sorted[-1]}°', (cxa_sorted[-1], cya_sorted[-1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Cxa')
        ax.set_ylabel('Cya')
        ax.set_title(f'Поляра ЛА (M = {M_ref:.2f}, δ=0°) для разных углов атаки\nКонфигурация: {self.config_scheme}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Добавляем цветовую шкалу для углов атаки
        if alpha_values:
            sc = ax.scatter([], [], c=[], cmap='viridis')
            sc.set_clim(vmin=min(alpha_values), vmax=max(alpha_values))
            plt.colorbar(sc, ax=ax, label='Угол атаки, °')

    def _select_representative_values(self, values, max_count):
        """Выбор представительных значений из списка"""
        if len(values) <= max_count:
            return values
        
        # Выбираем равномерно распределенные значения
        step = len(values) // max_count
        selected = [values[i] for i in range(0, len(values), step)][:max_count]
        
        # Добавляем минимальное и максимальное значения, если их нет
        if min(values) not in selected:
            selected.insert(0, min(values))
        if max(values) not in selected:
            selected.append(max(values))
        
        return sorted(selected)
 
    def print_summary_results(self, results):
        """Вывод сводных результатов расчета"""
        print("\n" + "="*80)
        print("СВОДНЫЕ РЕЗУЛЬТАТЫ РАСЧЕТА АДХ ЛЕТАТЕЛЬНОГО АППАРАТА")
        print("="*80)
        
        # Результаты для типичных условий (M=1.0, α=5°, δ=0°)
        M_ref = 1.0
        alpha_ref = 5
        delta_ref = 0
        
        if M_ref in results and alpha_ref in results[M_ref] and delta_ref in results[M_ref][alpha_ref]:
            data = results[M_ref][alpha_ref][delta_ref][0]  # phi_alpha = 0
            
            print(f"\nУсловия: M = {M_ref}, α = {alpha_ref}°, δ = {delta_ref}°, φ = 0°")
            print(f"Конфигурация: {self.config_scheme}")
            print(f"Характерная площадь: S_ref = {self.S_ref:.3f} м²")
            print(f"Характерная длина: L_ref = {self.L_ref:.3f} м")
            print(f"Координата ЦМ: x0 = {self.x0_rel:.2f}L_ref = {self.x0_rel:.3f} м")
            
            print(f"\nСИЛЫ И МОМЕНТЫ:")
            print(f"Cx = {data['cx']:.6f}")
            print(f"Cy = {data['cy']:.6f}")
            print(f"Cz = {data['cz']:.6f}")
            print(f"Cxa = {data['cxa']:.6f} (по формуле 2.15)")
            print(f"Cya = {data['cya']:.6f} (по формуле 2.16)")
            print(f"Mx = {data['mx']:.6f}")
            print(f"My = {data['my']:.6f}")
            print(f"Mz = {data['mz']:.6f}")
            print(f"Центр давления: xd = {data['xd']:.3f} м")
            
            # Расчет производных устойчивости (при delta=0)
            alphas_deg = sorted(results[M_ref].keys())
            if len(alphas_deg) >= 2:
                cy_values = [results[M_ref][alpha][0][0]['cy'] for alpha in alphas_deg if 0 in results[M_ref][alpha]]
                cya_values = [results[M_ref][alpha][0][0]['cya'] for alpha in alphas_deg if 0 in results[M_ref][alpha]]
                mz_values = [results[M_ref][alpha][0][0]['mz'] for alpha in alphas_deg if 0 in results[M_ref][alpha]]
                
                # Производная Cy по α
                if len(cy_values) > 1:
                    dcy_dalpha = (cy_values[1] - cy_values[0]) / (alphas_deg[1] - alphas_deg[0])
                    print(f"Cy_α = {dcy_dalpha:.4f} 1/град")
                
                # Производная Cya по α
                if len(cya_values) > 1:
                    dcya_dalpha = (cya_values[1] - cya_values[0]) / (alphas_deg[1] - alphas_deg[0])
                    print(f"Cya_α = {dcya_dalpha:.4f} 1/град")
                
                # Производная Mz по α (статическая устойчивость)
                if len(mz_values) > 1:
                    dmz_dalpha = (mz_values[1] - mz_values[0]) / (alphas_deg[1] - alphas_deg[0])
                    stability = "устойчив" if dmz_dalpha < 0 else "неустойчив"
                    print(f"Mz_α = {dmz_dalpha:.4f} 1/град ({stability})")
    
    def save_results_to_file(self, results, filename="aircraft_aerodynamics_results.txt"):
        """Сохранение результатов в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Результаты расчета аэродинамических характеристик ЛА\n")
            f.write("="*60 + "\n\n")
            f.write(f"Конфигурация: {self.config_scheme}\n")
            f.write(f"Информация о конфигурации: {self.get_current_configuration_info()}\n\n")
            
            for M in sorted(results.keys()):
                f.write(f"Число Маха M = {M:.2f}\n")
                f.write("-"*40 + "\n")
                
                for alpha in sorted(results[M].keys()):
                    for delta in sorted(results[M][alpha].keys()):
                        for phi_alpha in results[M][alpha][delta].keys():
                            data = results[M][alpha][delta][phi_alpha]
                            
                            f.write(f"α = {alpha}°, δ = {delta}°, φ = {np.rad2deg(phi_alpha):.1f}°:\n")
                            f.write(f"  Cx = {data['cx']:.6f}\n")
                            f.write(f"  Cy = {data['cy']:.6f}\n")
                            f.write(f"  Cz = {data['cz']:.6f}\n")
                            f.write(f"  Cxa = {data['cxa']:.6f}\n")
                            f.write(f"  Cya = {data['cya']:.6f}\n")
                            f.write(f"  Mx = {data['mx']:.6f}\n")
                            f.write(f"  My = {data['my']:.6f}\n")
                            f.write(f"  Mz = {data['mz']:.6f}\n")
                            f.write(f"  xd = {data['xd']:.4f} м\n\n")
        
        print(f"Результаты сохранены в файл: {filename}")
    
    def create_data_table(self, results):
        """Создание таблицы данных в требуемом формате"""
        if np.array_equal(self.delta_list, np.deg2rad(np.linspace(0, 0, self.n))):
            table_data = []
            
            for M in sorted(results.keys()):
                for alpha_deg in sorted(results[M].keys()):
                    for delta_deg in sorted(results[M][alpha_deg].keys())*self.n:
                        for phi_alpha in results[M][alpha_deg][delta_deg].keys():
                            data = results[M][alpha_deg][delta_deg][phi_alpha]
                            
                            # Преобразуем углы обратно в радианы для consistency
                            alpha_rad = np.deg2rad(alpha_deg)
                            delta_rad = np.deg2rad(delta_deg)
                            
                            table_data.append({
                                'M': M,
                                'alpha': alpha_rad,
                                'delta': delta_rad,
                                'c_xa': data['cxa'],
                                'c_ya': data['cya'],
                                'c_x': data['cx'],
                                'c_y': data['cy'],
                                'm_z': data['mz']
                            })
        else:
            table_data = []
            
            for M in sorted(results.keys()):
                for alpha_deg in sorted(results[M].keys()):
                    for delta_deg in sorted(results[M][alpha_deg].keys()):
                        for phi_alpha in results[M][alpha_deg][delta_deg].keys():
                            data = results[M][alpha_deg][delta_deg][phi_alpha]
                            
                            # Преобразуем углы обратно в радианы для consistency
                            alpha_rad = np.deg2rad(alpha_deg)
                            delta_rad = np.deg2rad(delta_deg)
                            
                            table_data.append({
                                'M': M,
                                'alpha': alpha_rad,
                                'delta': delta_rad,
                                'c_xa': data['cxa'],
                                'c_ya': data['cya'],
                                'c_x': data['cx'],
                                'c_y': data['cy'],
                                'm_z': data['mz']
                            })
        return table_data
    
    def save_table_to_python_format(self, results, filename="aircraft_data.py"):
        """Сохранение таблицы в формате Python списка списков"""
        table_data = self.create_data_table(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Заголовок файла с комментариями
            f.write("# АЭРОДИНАМИЧЕСКИЕ ХАРАКТЕРИСТИКИ ЛЕТАТЕЛЬНОГО АППАРАТА\n")
            f.write(f"# Конфигурация: {self.config_scheme}\n")
            f.write("# Формат данных: [M, alpha_rad, delta, c_xa, c_ya, c_x, c_y, m_z]\n")
            f.write("# M - число Маха\n")
            f.write("# alpha_rad - угол атаки в радианах\n")
            f.write("# delta - угол отклонения рулей в радианах\n")
            f.write("# c_xa, c_ya - коэффициенты по формулам (2.15) и (2.16)\n")
            f.write("# c_x, c_y - коэффициенты продольной и нормальной силы\n")
            f.write("# m_z - коэффициент момента тангажа\n")
            f.write("AD_tabl = [\n")
            
            # Данные в формате Python списков
            for i, row in enumerate(table_data):
                line = f"    [{row['M']:.6f}, {row['alpha']:.8f}, {row['delta']:.8f}, "
                line += f"{row['c_xa']:.8f}, {row['c_ya']:.8f}, "
                line += f"{row['c_x']:.8f}, {row['c_y']:.8f}, {row['m_z']:.8f}]"
                
                # Добавляем запятую для всех строк кроме последней
                if i < len(table_data) - 1:
                    line += ","
                else:
                    line += ""
                
                f.write(line + "\n")
            
            f.write("]\n\n")
            
            # Дополнительная информация о размерах данных
            f.write(f"# Всего точек расчета: {len(table_data)}\n")
            f.write(f"# Диапазон чисел Маха: {min(row['M'] for row in table_data):.2f} - {max(row['M'] for row in table_data):.2f}\n")
            f.write(f"# Диапазон углов атаки: {np.rad2deg(min(row['alpha'] for row in table_data)):.1f}° - {np.rad2deg(max(row['alpha'] for row in table_data)):.1f}°\n")
            f.write(f"# Диапазон углов отклонения рулей: {np.rad2deg(min(row['delta'] for row in table_data)):.8f}° - {np.rad2deg(max(row['delta'] for row in table_data)):.1f}°\n")
        
        print(f"Таблица данных сохранена в Python-файл: {filename}")
        return table_data
    
    def print_python_format_preview(self, table_data, num_rows=5):
        """Печать предварительного просмотра в Python формате"""
        print("\nПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР ТАБЛИЦЫ В PYTHON ФОРМАТЕ:")
        print("[" + "M        alpha[рад]    delta[рад] c_xa              c_ya              c_x               c_y               m_z".ljust(100) + "]")
        print("-" * 120)
        
        for i, row in enumerate(table_data[:num_rows]):
            line = f"[{row['M']:.2f}, {row['alpha']:.6f}, {row['delta']:.8f}, "
            line += f"{row['c_xa']:.8f}, {row['c_ya']:.8f}, "
            line += f"{row['c_x']:.8f}, {row['c_y']:.8f}, {row['m_z']:.8f}]"
            print("    " + line + ",")
        
        if len(table_data) > num_rows:
            print("    ...")
            # Показываем также последние строки
            for i, row in enumerate(table_data[-2:], start=len(table_data)-2):
                line = f"[{row['M']:.2f}, {row['alpha']:.6f}, {row['delta']:.6f}, "
                line += f"{row['c_xa']:.8f}, {row['c_ya']:.8f}, "
                line += f"{row['c_x']:.8f}, {row['c_y']:.8f}, {row['m_z']:.8f}]"
                comma = "," if i < len(table_data) - 1 else ""
                print("    " + line + comma)
                
                
              
#_setup_no_rudders_configuration 
# Основной исполняемый блок с примерами использования разных конфигураций
if __name__ == "__main__":
    print("=== РАСЧЕТ АЭРОДИНАМИЧЕСКИХ ХАРАКТЕРИСТИК ЛА ===")
    print("Демонстрация работы с разным количеством консолей\n")
    
    # Запуск расчета для выбранной конфигурации
    print("\n" + "="*60)
    print("ЗАПУСК РАСЧЕТА ДЛЯ ВЫБРАННОЙ КОНФИГУРАЦИИ")
    print("="*60)
    
    
    selected_config = "no_rudders"  
    # selected_config = "no_rudders"  # Только 6 крыльев  
    # selected_config = "wings_front"  # 4 крыла спереди + 4 руля сзади
    wing_geo = WingGeometry()
    if selected_config == "default":
        aircraft_final = AircraftAerodynamics(wing_geo.H, x0_rel=0.659, config_scheme="default", 
                                            num_front_consoles=4, num_rear_consoles=4)
    elif selected_config == "no_rudders":
        aircraft_final = AircraftAerodynamics(wing_geo.H, x0_rel=0.6, config_scheme="no_rudders", 
                                            num_front_consoles=0, num_rear_consoles=4)
    elif selected_config == "no_rudders_wings":
        aircraft_final = AircraftAerodynamics(wing_geo.H, x0_rel=0.2, config_scheme="no_rudders_wings", 
                                            num_front_consoles=0, num_rear_consoles=0)    
    else:  # wings_front
        aircraft_final = AircraftAerodynamics(wing_geo.H, x0_rel=0.659, config_scheme="wings_front", 
                                            num_front_consoles=4, num_rear_consoles=4)
    
    # Вывод информации о финальной конфигурации
    final_info = aircraft_final.get_current_configuration_info()
    print(f"\nФИНАЛЬНАЯ КОНФИГУРАЦИЯ ДЛЯ РАСЧЕТА:")
    print(f"Схема: {final_info['scheme']}")
    print(f"Передний пояс: {final_info['front_belt_count']} консолей ({final_info['front_belt_type']})")
    print(f"Задний пояс: {final_info['rear_belt_count']} консолей ({final_info['rear_belt_type']})")
    print(f"Углы передних консолей: {final_info['front_angles']}")
    print(f"Углы задних консолей: {final_info['rear_angles']}")
    
    # Построение геометрии
    print("\nПостроение геометрии фюзеляжа...")
    aircraft_final.plot_fuselage_geometry()
    
    # Запуск расчета (можно раскомментировать для реального расчета)
    
    print("Запуск расчета аэродинамических характеристик...")
    results = aircraft_final.run_complete_analysis()
    
    print("Построение графиков аэродинамических характеристик...")
    aircraft_final.plot_aircraft_characteristics(results)
     
    # Вывод сводных результатов
    aircraft_final.print_summary_results(results)
      
    # Сохранение результатов
    filename = f"aircraft_data.py"
    aircraft_final.save_table_to_python_format(results, filename)
    aircraft_final.save_table_to_python_format_2(results, filename)
    
    print(f"\nГотово! Код поддерживает конфигурации с любым количеством консолей.")
    print("Используйте методы set_consoles_count() и change_configuration_scheme()")
    print("для настройки нужного количества консолей в каждом поясе.")