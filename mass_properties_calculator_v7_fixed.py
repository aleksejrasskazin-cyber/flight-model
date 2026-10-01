# [file name]: mass_properties_calculator_v6_disk_method_final.py
"""
Модуль для расчета массовых характеристик методом дискретизации (тонких дисков)
Интеграция с inter.py для расчета центра масс ракеты
Версия с заданием объема СКТ, объема БЧ и длины ДУ
"""

import numpy as np
import math
from scipy.integrate import quad
from Fesulyadge_class import FuselageGeometry
from Krilo_class import WingGeometry, WingAerodynamicCalculator
from ruli_class import RudderGeometry, RudderAerodynamicCalculator
import matplotlib.pyplot as plt

class DiscreteMassCalculator:
    """
    Калькулятор массовых характеристик методом дискретизации (тонких дисков)
    Аналогично подходу в MathCAD
    """
    
    def __init__(self, dx: float = 0.001):
        """
        Инициализация с шагом дискретизации
        
        Параметры:
        dx - шаг дискретизации по оси X (от носа к корме, м)
        """
        self.dx = dx
        
        # Массивы для накопления характеристик
        self.elements = []
        
    class ElementProperties:
        """Свойства элементарного диска/кольца"""
        def __init__(self, x: float, r_inner: float, r_outer: float, 
                    density: float, dx: float, element_type: str = "disk", component: str = None):
            self.x = x  # Координата центра элемента по оси X (от носа к корме)
            self.r_inner = r_inner  # Внутренний радиус (0 для диска)
            self.r_outer = r_outer  # Наружный радиус
            self.density = density  # Плотность материала (кг/м³)
            self.dx = dx  # Толщина элемента (шаг дискретизации)
            self.type = element_type  # "disk" или "ring"
            self.component = component  # К какому компоненту относится
            
            # Вычисляемые свойства
            self.volume = 0
            self.mass = 0
            self.static_moment = 0
            self.Ixx = 0  # Момент инерции относительно оси X (ось ракеты)
            self.Iyy = 0  # Момент инерции относительно оси Y (поперечная)
            self.Izz = 0  # Момент инерции относительно оси Z (поперечная)
    
            self._calculate_properties()
        
        def _calculate_properties(self):
            """Вычисление свойств элементарного диска/кольца"""
            # Площадь поперечного сечения
            area = math.pi * (self.r_outer**2 - self.r_inner**2)
            
            # Объем элемента
            self.volume = area * self.dx
            
            # Масса
            self.mass = self.volume * self.density
            
            # Статический момент относительно носа (x=0)
            self.static_moment = self.mass * self.x
            
            # Моменты инерции относительно начала координат (нос)
            # Для тонкого диска/кольца толщиной dx
            if self.mass > 0:
                # Локальные моменты инерции (относительно собственного центра)
                # Для вращения вокруг оси X (продольной):
                Ixx_local = (self.mass/2.0) * (self.r_outer**2 + self.r_inner**2)
                
                # Для вращения вокруг поперечных осей Y/Z:
                Iyy_local = (self.mass/12.0) * (3*(self.r_outer**2 + self.r_inner**2) + self.dx**2)
                Izz_local = Iyy_local  # Симметрия
                
                # Моменты относительно начала координат (ось X)
                # По теореме Штейнера: I = I_local + m*d²
                d = self.x  # Расстояние от носа до центра элемента
                
                self.Ixx = Ixx_local
                self.Iyy = Iyy_local + self.mass * d**2
                self.Izz = Izz_local + self.mass * d**2
        
        def __repr__(self):
            return f"Element(x={self.x:.4f}, r_in={self.r_inner:.4f}, r_out={self.r_outer:.4f}, m={self.mass:.6f}, comp={self.component})"
    
    def add_element(self, x: float, r_inner: float, r_outer: float, 
                   density: float, element_type: str = "disk", component: str = None) -> None:
        """
        Добавление элементарного диска/кольца
        
        Параметры:
        x - координата центра элемента по оси X (от носа ракеты)
        r_inner - внутренний радиус (0 для сплошного диска)
        r_outer - наружный радиус
        density - плотность материала (кг/м³)
        element_type - тип элемента: "disk" (диск) или "ring" (кольцо)
        component - название компонента (СКТ, БЧ, ДУ и т.д.)
        """
        element = self.ElementProperties(x, r_inner, r_outer, density, self.dx, element_type, component)
        self.elements.append(element)
    
    def add_cylinder_segment(self, x_start: float, x_end: float, 
                            r_inner: float, r_outer: float, 
                            density: float, component: str = None) -> None:
        """
        Добавление цилиндрического сегмента (разбивается на диски)
        """
        length = abs(x_end - x_start)
        num_segments = max(1, int(length / self.dx))
        
        if num_segments < 1:
            num_segments = 1
        
        step = length / num_segments
        
        for i in range(num_segments):
            x_center = x_start + (i + 0.5) * step
            self.add_element(x_center, r_inner, r_outer, density, 
                           "ring" if r_inner > 0 else "disk", component)
    
    def add_cylinder_segment_variable(self, x_start: float, x_end: float, 
                                     r_inner_func, r_outer_func,
                                     density: float, component: str = None) -> None:
        """
        Добавление сегмента с переменным радиусом
        """
        length = abs(x_end - x_start)
        num_segments = max(1, int(length / self.dx))
        
        if num_segments < 1:
            num_segments = 1
        
        step = length / num_segments
        
        for i in range(num_segments):
            x_center = x_start + (i + 0.5) * step
            r_inner = r_inner_func(x_center) if callable(r_inner_func) else r_inner_func
            r_outer = r_outer_func(x_center) if callable(r_outer_func) else r_outer_func
            
            element_type = "ring" if r_inner > 0 else "disk"
            self.add_element(x_center, r_inner, r_outer, density, element_type, component)
    
    def calculate_total_properties(self) -> dict:
        """
        Расчет суммарных массовых характеристик
        
        Возвращает:
        Словарь с полными характеристиками
        """
        if not self.elements:
            return {
                'total_mass': 0,
                'total_volume': 0,
                'center_of_mass': 0,
                'Ixx': 0, 'Iyy': 0, 'Izz': 0,
                'elements_count': 0,
                'components': {}
            }
        
        # Суммирование характеристик
        total_mass = 0
        total_volume = 0
        total_static_moment = 0
        total_Ixx = 0
        total_Iyy = 0
        total_Izz = 0
        
        # Характеристики по компонентам
        components = {}
        
        for element in self.elements:
            total_mass += element.mass
            total_volume += element.volume
            total_static_moment += element.static_moment
            total_Ixx += element.Ixx
            total_Iyy += element.Iyy
            total_Izz += element.Izz
            
            # Суммируем по компонентам
            if element.component:
                if element.component not in components:
                    components[element.component] = {
                        'mass': 0,
                        'volume': 0,
                        'static_moment': 0,
                        'elements': []
                    }
                
                components[element.component]['mass'] += element.mass
                components[element.component]['volume'] += element.volume
                components[element.component]['static_moment'] += element.static_moment
                components[element.component]['elements'].append(element)
        
        # Центр масс
        center_of_mass = total_static_moment / total_mass if total_mass > 0 else 0
        
        # Пересчет моментов инерции относительно центра масс
        d_cm = center_of_mass
        
        Ixx_cm = total_Ixx  # Продольный момент не меняется при переносе оси
        Iyy_cm = total_Iyy - total_mass * d_cm**2
        Izz_cm = total_Izz - total_mass * d_cm**2
        
        # Расчет центров масс компонентов
        for comp_name, comp_data in components.items():
            if comp_data['mass'] > 0:
                comp_data['center_of_mass'] = comp_data['static_moment'] / comp_data['mass']
                comp_data['mass_fraction'] = comp_data['mass'] / total_mass
            else:
                comp_data['center_of_mass'] = 0
                comp_data['mass_fraction'] = 0
        
        return {
            'total_mass': total_mass,
            'total_volume': total_volume,
            'center_of_mass': center_of_mass,
            'Ixx': Ixx_cm,  # Относительно ЦМ
            'Iyy': Iyy_cm,  # Относительно ЦМ
            'Izz': Izz_cm,  # Относительно ЦМ
            'Ixx_origin': total_Ixx,  # Относительно носа
            'Iyy_origin': total_Iyy,  # Относительно носа
            'Izz_origin': total_Izz,  # Относительно носа
            'elements_count': len(self.elements),
            'mass_density': total_mass / total_volume if total_volume > 0 else 0,
            'components': components
        }


class MassPropertiesCalculatorV6:
    """Класс для расчета массовых характеристик ракеты с методом дискретизации"""
    
    def __init__(self, fuselage_geometry, config_scheme="default", 
                 num_front_consoles=0, num_rear_consoles=0,
                 skt_volume=None, bc_volume=None, du_length=None, fuel_mass=18.52, fuel_density=1650):
        """
        Инициализация с геометрией и заданными параметрами СКТ, БЧ, ДУ
        
        Параметры:
        fuselage_geometry - геометрия фюзеляжа
        config_scheme - схема расположения консолей
        num_front_consoles - количество передних консолей
        num_rear_consoles - количество задних консолей
        
        skt_volume - объем СКТ (м³) [ОБЯЗАТЕЛЬНЫЙ параметр]
        bc_volume - объем БЧ (м³) [ОБЯЗАТЕЛЬНЫЙ параметр]
        du_length - длина ДУ (м) [ОБЯЗАТЕЛЬНЫЙ параметр]
        """
        self.geo = fuselage_geometry
        self.config_scheme = config_scheme
        self.num_front_consoles = num_front_consoles
        self.num_rear_consoles = num_rear_consoles
        
        # Параметры компонентов (заданные пользователем)
        self.skt_volume = skt_volume
        self.bc_volume = bc_volume
        self.du_length = du_length
        
        # Проверка наличия обязательных параметров
        if skt_volume is None:
            raise ValueError("Необходимо задать объем СКТ (skt_volume)")
        if bc_volume is None:
            raise ValueError("Необходимо задать объем БЧ (bc_volume)")
        if du_length is None:
            raise ValueError("Необходимо задать длину ДУ (du_length)")
        
        # Создаем объекты геометрии крыльев и рулей
        self.wing_geo = WingGeometry()
        self.rudder_geo = RudderGeometry()
        
        # Параметры фюзеляжа - ТОЛЬКО ИЗ FUSELAGEGЕOMETRY
        self.l_nos = self.geo.l_nos
        self.l_cyl = self.geo.l_cil
        self.l_don = self.geo.l_don
        self.d_m = self.geo.d_m
        self.d_k = self.geo.d_k  # Диаметр кормы из фюзеляжа
        self.L_total = self.l_nos + self.l_cyl + self.l_don
        
        # Стандартные плотности компонентов (кг/м³)
        self.densities = {
            'СКТ': 1100.0,     # Система коррекции траектории
            'БЧ': 4400.0,      # Боевая часть
            'ДУ': 1000.0,      # Двигательная установка
            'обшивка': 600.0,#2800.0, # Материал обшивки
            'консоли': 2500.0, # Средняя плотность консолей
        }
        
        # Толщины конструкций (м)
        self.thickness = {
            'обшивка': 0.002,  # Толщина обшивки фюзеляжа (2 мм)
        }
        
        # Удельные массы консолей (кг/м²)
        self.specific_mass = {
            'крыло': 2.0,      # Легкие композитные крылья
            'руль': 8.0,       # Рули с приводами немного тяжелее
        }
        
        self.fuel_mass = fuel_mass
        self.fuel_density = fuel_density
        
        # Плотность ДУ теперь будет средней (конструкция + топливо)
        if fuel_mass > 0:
            # Пересчитываем эффективную плотность ДУ
            du_volume = du_length * math.pi * (self.geo.d_m/2)**2
            structure_mass = du_volume * 1000  # масса пустой ДУ
            total_du_mass = structure_mass + fuel_mass
            self.densities['ДУ'] = total_du_mass / du_volume
        else:
            self.densities['ДУ'] = 1000.0  # пустая ДУ
            
        # Массивы для хранения координат консолей
        self.console_positions = []
        
        # Дискретный калькулятор
        self.disc_calc = DiscreteMassCalculator(dx=0.001)  # Шаг 1 мм
        
        # Настройка параметров консолей
        self._setup_console_parameters()
        
        # Инициализация геометрии
        self._initialize_geometry()
        
        # Расчет геометрии фюзеляжа
        self._calculate_fuselage_profile()
    
    def _setup_console_parameters(self):
        """Настройка параметров консолей в зависимости от схемы"""
        if self.config_scheme == "default":
            self.console_params = {
                'front': {'type': 'руль', 'count': self.num_front_consoles, 'geometry': self.rudder_geo},
                'rear': {'type': 'крыло', 'count': self.num_rear_consoles, 'geometry': self.wing_geo}
            }
        elif self.config_scheme == "no_rudders":
            self.console_params = {
                'front': {'type': 'крыло', 'count': self.num_front_consoles, 'geometry': self.wing_geo},
                'rear': {'type': 'крыло', 'count': self.num_rear_consoles, 'geometry': self.wing_geo}
            }
        elif self.config_scheme == "no_rudders_wings":
            self.console_params = {
                'front': {'type': 'none', 'count': 0, 'geometry': None},
                'rear': {'type': 'none', 'count': 0, 'geometry': None}
            }
        elif self.config_scheme == "wings_front":
            self.console_params = {
                'front': {'type': 'крыло', 'count': self.num_front_consoles, 'geometry': self.wing_geo},
                'rear': {'type': 'руль', 'count': self.num_rear_consoles, 'geometry': self.rudder_geo}
            }
        else:
            self.console_params = {
                'front': {'type': 'руль', 'count': self.num_front_consoles, 'geometry': self.rudder_geo},
                'rear': {'type': 'крыло', 'count': self.num_rear_consoles, 'geometry': self.wing_geo}
            }
    
    def _initialize_geometry(self):
        """Инициализация геометрических параметров"""
        self.x_points = np.linspace(0, self.L_total, 1000)
        self.radii = self._calculate_radii(self.x_points)
        
        # Расчет координат установки консолей
        self._calculate_console_positions()
    
    def _calculate_fuselage_profile(self):
        """Расчет профиля фюзеляжа для дискретизации"""
        # Точки профиля фюзеляжа (внешний контур)
        self.profile_x = []
        self.profile_r_outer = []
        
        n_points = 1000
        for i in range(n_points + 1):
            x = i * self.L_total / n_points
            r = self._calculate_radius_at_x(x)
            self.profile_x.append(x)
            self.profile_r_outer.append(r)
    
    def _calculate_radii(self, x_points):
        """Расчет радиуса фюзеляжа в заданных точках"""
        radii = []
        
        for x in x_points:
            radii.append(self._calculate_radius_at_x(x))
            
        return np.array(radii)
    
    def _calculate_radius_at_x(self, x):
        """Расчет радиуса в конкретной точке x"""
        if x <= self.geo.x_1:
            # Сферическая часть
            return math.sqrt(self.geo.r_zatup**2 - (x - self.geo.r_zatup)**2)
        elif x <= self.geo.l_nos:
            # Оживальная часть
            sqrt_part = math.sqrt(self.geo.R**2 - (x - self.geo.l_nos)**2)
            return sqrt_part - self.geo.R + self.geo.d_m / 2
        elif x <= self.geo.l_nos + self.geo.l_cil:
            # Цилиндрическая часть
            return self.geo.d_m / 2
        elif x <= self.L_total:
            # Коническая кормовая часть
            x_kor = x - (self.geo.l_nos + self.geo.l_cil)
            return (self.geo.d_m / 2) - (x_kor / self.geo.l_don) * (self.geo.d_m - self.geo.d_k) / 2
        else:
            return 0
    
    def _calculate_console_positions(self):
        """Расчет координат установки консолей"""
        # Упрощенный расчет позиций консолей
        # Для задних консолей (обычно крылья) - ближе к корме
        rear_position = 0.85 * self.L_total  # 85% длины от носа
        
        self.console_positions = []
        
        # Координаты установки
        if self.num_front_consoles > 0:
            # Передние консоли устанавливаются на 75% длины от носа
            front_position = 0.75 * self.L_total
            self.console_positions.append({
                'position': 'front',
                'x_installation': front_position,
                'type': self.console_params['front']['type'],
                'count': self.num_front_consoles
            })
        
        if self.num_rear_consoles > 0:
            # Задние консоли
            self.console_positions.append({
                'position': 'rear',
                'x_installation': rear_position,
                'type': self.console_params['rear']['type'],
                'count': self.num_rear_consoles
            })
    
    def calculate_total_volume_disk_method(self):
        """Расчет полного объема методом дискретизации"""
        volume = 0
        n_disks = int(self.L_total / self.disc_calc.dx) + 1
        
        for i in range(n_disks):
            x = i * self.disc_calc.dx
            if x > self.L_total:
                break
                
            r_outer = self._calculate_radius_at_x(x)
            area = math.pi * r_outer**2
            volume += area * self.disc_calc.dx
        
        return volume
    
    def calculate_component_boundaries_by_volume_and_length(self):
        """
        Определение границ отсеков с учетом реальной геометрии
        """
        boundaries = {}
        current_x = 0
        
        # 1. Расчет для СКТ с учетом оживальной геометрии
        print(f"\n   Расчет СКТ в оживальной части...")
        
        # Находим длину оживальной части
        ogive_end = self.geo.l_nos
        
        # Решаем уравнение: ∫(π*r²(x)dx) от 0 до L = V_СКТ
        from scipy.optimize import bisect
        
        def volume_up_to_x(x_target):
            """Объем от носа до x_target"""
            vol = 0
            n_points = 1000
            for i in range(n_points):
                x = (i / n_points) * x_target
                if x > x_target:
                    break
                r = self._calculate_radius_at_x(x)
                vol += math.pi * r**2 * (x_target / n_points)
            return vol
        
        # Если объем СКТ меньше общего объема оживальной части
        total_ogive_volume = volume_up_to_x(ogive_end)
        
        if self.skt_volume <= total_ogive_volume:
            # СКТ полностью в оживальной части
            def find_length(L):
                return volume_up_to_x(L) - self.skt_volume
            
            try:
                skt_length = bisect(find_length, 0, ogive_end)
                skt_end = current_x + skt_length
                location_status = "полностью в оживальной части"
                ogive_portion = self.skt_volume
                cyl_portion = 0
            except:
                # Если не нашли корень, используем приближение
                skt_length = self.skt_volume / (math.pi * (self.geo.d_m/2)**2) * 1.5
                skt_end = current_x + skt_length
                location_status = "приблизительно в оживальной части"
                ogive_portion = self.skt_volume
                cyl_portion = 0
        else:
            # СКТ занимает всю оживальную часть + часть цилиндра
            remaining_volume = self.skt_volume - total_ogive_volume
            area_cyl = math.pi * (self.geo.d_m/2)**2
            cyl_length = remaining_volume / area_cyl
            skt_length = ogive_end + cyl_length
            skt_end = current_x + skt_length
            location_status = f"оживальная часть ({ogive_end:.3f} м) + цилиндрическая часть ({cyl_length:.3f} м)"
            ogive_portion = total_ogive_volume
            cyl_portion = remaining_volume
        
        boundaries['СКТ'] = {
            'start': current_x,
            'end': skt_end,
            'target_volume': self.skt_volume,
            'calculated_length': skt_length,
            'actual_volume': volume_up_to_x(skt_end),
            'location_status': location_status,
            'ogive_portion_volume': ogive_portion,
            'cyl_portion_volume': cyl_portion,
            'ogive_portion_fraction': (ogive_portion / self.skt_volume * 100) if self.skt_volume > 0 else 0,
            'fits_completely_in_ogive': self.skt_volume <= total_ogive_volume
        }
        current_x = skt_end
        
        # Информативный вывод для СКТ
        print(f"     Длина СКТ: {skt_length:.4f} м")
        print(f"     Расположение: {location_status}")
        if not boundaries['СКТ']['fits_completely_in_ogive']:
            print(f"     Объем в оживале: {ogive_portion:.6f} м³ ({boundaries['СКТ']['ogive_portion_fraction']:.1f}%)")
            print(f"     Объем в цилиндре: {cyl_portion:.6f} м³ ({100 - boundaries['СКТ']['ogive_portion_fraction']:.1f}%)")
        
        # 2. Расчет для БЧ (в цилиндрической части)
        print(f"\n   Расчет БЧ в цилиндрической части...")
        area_cyl = math.pi * (self.geo.d_m/2)**2
        bc_length = self.bc_volume / area_cyl
        bc_end = current_x + bc_length
        
        boundaries['БЧ'] = {
            'start': current_x,
            'end': bc_end,
            'target_volume': self.bc_volume,
            'calculated_length': bc_length,
            'actual_volume': self._calculate_volume_between(current_x, bc_end),
            'location_status': "цилиндрическая часть",
            'fits_completely_in_cyl': True
        }
        current_x = bc_end
        
        print(f"     Длина БЧ: {bc_length:.4f} м")
        print(f"     Расположение: цилиндрическая часть")
        
        # 3. ДУ
        print(f"\n   Расчет ДУ...")
        du_end = current_x + self.du_length
        du_volume = self._calculate_volume_between(current_x, du_end)
        
        boundaries['ДУ'] = {
            'start': current_x,
            'end': du_end,
            'target_length': self.du_length,
            'calculated_volume': du_volume,
            'actual_volume': du_volume,
            'location_status': "цилиндрическая и кормовая части",
            'fits_completely': True
        }
        
        print(f"     Длина ДУ: {self.du_length:.4f} м")
        print(f"     Расположение: от {current_x:.3f} до {du_end:.3f} м")
        
        # Сводный вывод
        print(f"\n   РЕЗУЛЬТАТЫ РАССТАНОВКИ КОМПОНЕНТОВ:")
        print(f"   {'─'*50}")
        print(f"   СКТ:      длина {skt_length:6.3f} м | {boundaries['СКТ']['location_status']}")
        print(f"   БЧ:       длина {bc_length:6.3f} м | {boundaries['БЧ']['location_status']}")
        print(f"   ДУ:       длина {self.du_length:6.3f} м | {boundaries['ДУ']['location_status']}")
        print(f"   {'─'*50}")
        print(f"   Суммарная длина компонентов: {current_x + self.du_length - boundaries['СКТ']['start']:.3f} м")
        print(f"   Длина фюзеляжа: {self.L_total:.3f} м")
        print(f"   Свободное место: {self.L_total - (current_x + self.du_length):.3f} м")
        
        return boundaries
    
    def _calculate_volume_between(self, x_start, x_end):
        """Расчет объема между двумя координатами"""
        volume = 0
        n_disks = int((x_end - x_start) / self.disc_calc.dx) + 1
        
        for i in range(n_disks):
            x = x_start + i * self.disc_calc.dx
            if x > x_end:
                break
                
            r_outer = self._calculate_radius_at_x(x)
            area = math.pi * r_outer**2
            volume += area * self.disc_calc.dx
        
        return volume
    
    def build_fuselage_disk_model(self, boundaries):
        """Построение модели фюзеляжа методом дискретизации"""
        
        # Очищаем предыдущие элементы
        self.disc_calc.elements = []
        
        # Добавляем компоненты фюзеляжа
        for component in ['СКТ', 'БЧ', 'ДУ']:
            if component in boundaries:
                bounds = boundaries[component]
                start_x = bounds['start']
                end_x = bounds['end']
                density = self.densities[component]
                
                # Добавляем внутренний объем компонента как сегмент с переменным радиусом
                self.disc_calc.add_cylinder_segment_variable(
                    start_x, end_x,
                    r_inner_func=0,  # Сплошной диск
                    r_outer_func=lambda x: self._calculate_radius_at_x(x),
                    density=density,
                    component=component
                )
        
        # Добавляем обшивку фюзеляжа (тонкая стенка)
        self._add_fuselage_shell()
        
        return True
    
    def _add_fuselage_shell(self):
        """Добавление обшивки фюзеляжа как тонкого кольца"""
        n_segments = int(self.L_total / self.disc_calc.dx) + 1
        
        for i in range(n_segments):
            x = i * self.disc_calc.dx
            if x > self.L_total:
                break
                
            r_outer = self._calculate_radius_at_x(x)
            r_inner = max(0, r_outer - self.thickness['обшивка'])
            
            if r_inner > 0:  # Только если есть толщина
                self.disc_calc.add_element(
                    x=x,
                    r_inner=r_inner,
                    r_outer=r_outer,
                    density=self.densities['обшивка'],
                    element_type="ring",
                    component="обшивка"
                )
    
    def calculate_console_mass_disk_method(self, geometry, console_type='крыло'):
        """
        Расчет массы с использованием всех параметров WingGeometry
        """
        if console_type == 'крыло':
            # Удельная масса для разных типов крыльев
            if hasattr(geometry, 'type') and geometry.type == 'стреловидное':
                specific_mass = 2.5  # стреловидные тяжелее
            else:
                specific_mass = self.specific_mass['крыло']
        else:
            specific_mass = self.specific_mass['руль']
        
        # Площадь крыла
        S = geometry.S
        
        # Коэффициент сложности конструкции
        complexity_factor = 1.0
        
        # Учитываем размах (большой размах - легче)
        if hasattr(geometry, 'b'):
            complexity_factor *= (1.0 - 0.1 * (geometry.b - 1.0))
        
        # Учитываем сужение
        if hasattr(geometry, 'ny'):
            if geometry.ny < 0.5:  # сильное сужение
                complexity_factor *= 1.2
        
        # Учитываем стреловидность
        if hasattr(geometry, 'chi') and geometry.chi > 0:
            complexity_factor *= (1.0 + geometry.chi / 90.0)
        
        # Базовая масса
        mass = S * specific_mass * complexity_factor
        
        # Масса механизации (элероны, закрылки)
        if hasattr(geometry, 'has_flaps') and geometry.has_flaps:
            mass *= 1.15
        
        # Центр масс с учетом реальной геометрии
        if hasattr(geometry, 'xA') and hasattr(geometry, 'bA'):
            # Положение САХ
            cm_local = geometry.xA + 0.4 * geometry.bA
            
            # Для стреловидных крыльев ЦМ смещен назад
            if hasattr(geometry, 'chi') and geometry.chi > 0:
                cm_local += 0.05 * geometry.bA
        else:
            # Если нет данных - усредненное значение
            cm_local = 0.25 * geometry.bA if hasattr(geometry, 'bA') else 0.3
        
        # Масса пилонов и узлов крепления
        pylon_mass = 0.1 * mass
        mass += pylon_mass
        
        return mass, cm_local
    
    def calculate_mass_properties_disk_method(self):
        """Расчет полных массовых характеристик методом дискретизации"""
        print("\n" + "="*80)
        print("РАСЧЕТ МАССОВЫХ ХАРАКТЕРИСТИК МЕТОДОМ ДИСКРЕТИЗАЦИИ")
        print(f"Шаг дискретизации: {self.disc_calc.dx:.4f} м")
        print("="*80)
        
        # 1. Определение границ отсеков по объему и длине
        print("\n1. Определение границ отсеков по заданным параметрам...")
        print("   Исходные данные:")
        print(f"   - Объем СКТ: {self.skt_volume:.6f} м³")
        print(f"   - Объем БЧ: {self.bc_volume:.6f} м³")
        print(f"   - Длина ДУ: {self.du_length:.4f} м")
        
        boundaries = self.calculate_component_boundaries_by_volume_and_length()
        
        # 2. Построение модели фюзеляжа
        print("\n2. Построение модели фюзеляжа методом дискретизации...")
        self.build_fuselage_disk_model(boundaries)
        
        # 3. Добавление консолей
        print("\n3. Добавление консолей...")
        console_properties = self._add_consoles_to_model()
        
        # 4. Расчет суммарных характеристик
        print("\n4. Расчет суммарных характеристик...")
        total_props = self.disc_calc.calculate_total_properties()
        
        # 5. Формирование результатов
        results = {
            'component_properties': {},
            'total_mass': total_props['total_mass'],
            'overall_center_of_mass': total_props['center_of_mass'],
            'mass_fractions': {},
            'boundaries': boundaries,
            'console_properties': console_properties,
            'total_volume': total_props['total_volume'],
            'total_length': self.L_total,
            'moments_of_inertia': {
                'Ixx': total_props['Ixx'],
                'Iyy': total_props['Iyy'],
                'Izz': total_props['Izz']
            },
            'disk_method_details': {
                'elements_count': total_props['elements_count'],
                'dx': self.disc_calc.dx
            },
            'input_parameters': {
                'skt_volume': self.skt_volume,
                'bc_volume': self.bc_volume,
                'du_length': self.du_length
            }
        }
        
        # Информация о компонентах
        for comp_name, comp_data in total_props['components'].items():
            results['component_properties'][comp_name] = {
                'mass': comp_data['mass'],
                'volume': comp_data['volume'],
                'center_of_mass': comp_data.get('center_of_mass', 0),
                'mass_fraction': comp_data.get('mass_fraction', 0)
            }
            
            if comp_data['mass'] > 0:
                results['mass_fractions'][comp_name] = comp_data['mass'] / total_props['total_mass']
        
        # Информация о консолях
        console_total_mass = 0
        for pos in ['front', 'rear']:
            if console_properties[pos]['count'] > 0:
                console_total_mass += console_properties[pos]['total_mass']
        
        if console_total_mass > 0:
            results['mass_fractions']['консоли'] = console_total_mass / total_props['total_mass']
        
        # Вывод сводной информации
        print(f"\n5. РЕЗУЛЬТАТЫ:")
        print(f"   Общая масса: {results['total_mass']:.2f} кг")
        print(f"   Общий объем: {results['total_volume']:.6f} м³")
        print(f"   Центр масс: {results['overall_center_of_mass']:.3f} м от носа")
        print(f"   Относительная координата ЦМ: {results['overall_center_of_mass']/self.L_total:.3f} L")
        print(f"   Количество дискретных элементов: {total_props['elements_count']}")
        
        return results
    
    def _add_consoles_to_model(self):
        """Добавление консолей в дискретную модель"""
        console_properties = {
            'front': {'mass': 0, 'center_of_mass': 0, 'static_moment': 0, 'count': 0},
            'rear': {'mass': 0, 'center_of_mass': 0, 'static_moment': 0, 'count': 0}
        }
        
        # Находим координаты установки
        front_install = None
        rear_install = None
        
        for pos_info in self.console_positions:
            if pos_info['position'] == 'front':
                front_install = pos_info['x_installation']
            elif pos_info['position'] == 'rear':
                rear_install = pos_info['x_installation']
        
        # Расчет для передних консолей
        if self.num_front_consoles > 0 and front_install is not None:
            console_type = self.console_params['front']['type']
            geometry = self.console_params['front']['geometry']
            count = self.num_front_consoles
            
            single_mass, local_cm = self.calculate_console_mass_disk_method(geometry, console_type)
            global_cm = front_install + local_cm
            
            console_properties['front'] = {
                'type': console_type,
                'count': count,
                'single_mass': single_mass,
                'total_mass': single_mass * count,
                'center_of_mass_local': local_cm,
                'installation_position': front_install,
                'center_of_mass_global': global_cm,
                'static_moment': (single_mass * count) * global_cm
            }
            
            # Добавляем массу консолей как точечные массы
            for _ in range(count):
                self.disc_calc.add_element(
                    x=global_cm,
                    r_inner=0,
                    r_outer=0.9,  # Маленький радиус для точечной массы
                    density=single_mass / (math.pi * 0.01**2 * self.disc_calc.dx),
                    element_type="disk",
                    component="консоли"
                )
        
        # Расчет для задних консолей
        if self.num_rear_consoles > 0 and rear_install is not None:
            console_type = self.console_params['rear']['type']
            geometry = self.console_params['rear']['geometry']
            count = self.num_rear_consoles
            
            single_mass, local_cm = self.calculate_console_mass_disk_method(geometry, console_type)
            global_cm = rear_install + local_cm
            
            console_properties['rear'] = {
                'type': console_type,
                'count': count,
                'single_mass': single_mass,
                'total_mass': single_mass * count,
                'center_of_mass_local': local_cm,
                'installation_position': rear_install,
                'center_of_mass_global': global_cm,
                'static_moment': (single_mass * count) * global_cm
            }
            
            for _ in range(count):
                self.disc_calc.add_element(
                    x=global_cm,
                    r_inner=0,
                    r_outer=0.01,
                    density=single_mass / (math.pi * 0.01**2 * self.disc_calc.dx),
                    element_type="disk",
                    component="консоли"
                )
        
        return console_properties
    
    def plot_disk_model(self, results=None):
        """Визуализация дискретной модели"""
        if results is None:
            results = self.calculate_mass_properties_disk_method()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Модель дискретизации ракеты (схема: {self.config_scheme})', fontsize=14)
        
        # 1. Геометрия с отсеками
        ax1 = axes[0, 0]
        
        # Контур фюзеляжа
        ax1.plot(self.x_points, self.radii, 'b-', linewidth=2, label='Верхний контур')
        ax1.plot(self.x_points, -self.radii, 'b-', linewidth=2)
        ax1.fill_between(self.x_points, self.radii, -self.radii, alpha=0.2, color='lightblue')
        
        # Отсеки
        boundaries = results['boundaries']
        colors_fuselage = {'СКТ': 'red', 'БЧ': 'orange', 'ДУ': 'green'}
        
        for component, color in colors_fuselage.items():
            if component in boundaries:
                bounds = boundaries[component]
                x_start = bounds['start']
                x_end = bounds['end']
                
                ax1.axvline(x=x_start, color=color, linestyle='--', alpha=0.7)
                ax1.axvline(x=x_end, color=color, linestyle='--', alpha=0.7)
                
                # Подпись для отсека
                if component == 'СКТ':
                    label = f'{component} (V={bounds["target_volume"]:.3f} м³, L={bounds["calculated_length"]:.2f} м)'
                elif component == 'БЧ':
                    label = f'{component} (V={bounds["target_volume"]:.3f} м³, L={bounds["calculated_length"]:.2f} м)'
                else:  # ДУ
                    label = f'{component} (L={bounds["target_length"]:.2f} м, V={bounds["calculated_volume"]:.3f} м³)'
                
                mask = (self.x_points >= x_start) & (self.x_points <= x_end)
                if np.any(mask):
                    ax1.fill_between(self.x_points[mask], self.radii[mask], 
                                    -self.radii[mask], alpha=0.3, color=color, 
                                    label=label)
        
        # Центр масс
        cm = results['overall_center_of_mass']
        ax1.axvline(x=cm, color='blue', linewidth=2, linestyle='-', 
                   label=f'ЦМ (x={cm:.3f} м)')
        ax1.plot(cm, 0, 'bo', markersize=10, markerfacecolor='red')
        
        ax1.set_xlabel('Длина ракеты, м')
        ax1.set_ylabel('Радиус, м')
        ax1.set_title('Геометрия с отсеками')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal') # auto
        # 2. Элементы дискретизации
        ax2 = axes[0, 1]
        
        # Показываем выборку дисков
        elements = self.disc_calc.elements
        if elements:
            # Берем каждый 20-й элемент для визуализации
            sample_elements = elements[::max(1, len(elements)//100)]
            
            for elem in sample_elements:
                color_map = {
                    'СКТ': 'red', 'БЧ': 'orange', 'ДУ': 'green',
                    'обшивка': 'gray', 'консоли': 'purple'
                }
                color = color_map.get(elem.component, 'blue')
                
                # Упрощенное представление диска
                ax2.plot(elem.x, elem.r_outer, 'o', color=color, markersize=2, alpha=0.5)
                if elem.r_inner > 0:
                    ax2.plot(elem.x, elem.r_inner, 'o', color=color, markersize=1, alpha=0.5)
        
        ax2.set_xlabel('Длина ракеты, м')
        ax2.set_ylabel('Радиус, м')
        ax2.set_title('Дискретные элементы (выборка)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.L_total)
        ax2.set_ylim(-self.d_m/2, self.d_m/2)
        
        # 3. Распределение массы
        ax3 = axes[1, 0]
        
        components_all = list(results['component_properties'].keys())
        masses_all = [results['component_properties'][comp]['mass'] for comp in components_all]
        labels_all = [f"{comp}\n{mass:.1f} кг" for comp, mass in zip(components_all, masses_all)]
        
        colors_all = ['red', 'orange', 'green', 'gray', 'purple'][:len(components_all)]
        
        if masses_all:
            wedges, texts, autotexts = ax3.pie(masses_all, labels=labels_all, 
                                              colors=colors_all,
                                              autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
        
        ax3.set_title('Распределение массы')
        
        # 4. Сводные характеристики
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary = "СВОДНЫЕ ХАРАКТЕРИСТИКИ:\n\n"
        summary += f"Схема: {self.config_scheme}\n"
        summary += f"Длина ракеты: {self.L_total:.3f} м\n"
        summary += f"Диаметр миделя: {self.d_m:.3f} м\n"
        summary += f"Диаметр кормы: {self.d_k:.3f} м\n"
        summary += f"Общий объем: {results['total_volume']:.6f} м³\n"
        summary += f"Общая масса: {results['total_mass']:.2f} кг\n"
        summary += f"Средняя плотность: {results['total_mass']/results['total_volume']:.1f} кг/м³\n"
        summary += f"Центр масс: {cm:.3f} м от носа\n"
        summary += f"Относительно: {cm/self.L_total:.3f} L\n\n"
        
        summary += "ИСХОДНЫЕ ПАРАМЕТРЫ:\n"
        summary += f"Объем СКТ: {results['input_parameters']['skt_volume']:.4f} м³\n"
        summary += f"Объем БЧ: {results['input_parameters']['bc_volume']:.4f} м³\n"
        summary += f"Длина ДУ: {results['input_parameters']['du_length']:.3f} м\n\n"
        
        summary += "МОМЕНТЫ ИНЕРЦИИ:\n"
        summary += f"Ixx = {results['moments_of_inertia']['Ixx']:.4f} кг·м²\n"
        summary += f"Iyy = {results['moments_of_inertia']['Iyy']:.4f} кг·м²\n"
        summary += f"Izz = {results['moments_of_inertia']['Izz']:.4f} кг·м²\n"
        
        ax4.text(0.1, 0.95, summary, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_disk_method_results(self, results, filename="rocket_mass_disk_method.txt"):
        """Сохранение результатов метода дискретизации в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"РАСЧЕТ МАССОВЫХ ХАРАКТЕРИСТИК РАКЕТЫ (МЕТОД ДИСКРЕТИЗАЦИИ)\n")
            f.write(f"Схема: {self.config_scheme}\n")
            f.write(f"Шаг дискретизации: {self.disc_calc.dx:.4f} м\n")
            f.write(f"Количество элементов: {results['disk_method_details']['elements_count']}\n")
            f.write("="*80 + "\n\n")
            
            f.write("ИСХОДНЫЕ ДАННЫЕ:\n")
            f.write(f"Объем СКТ: {results['input_parameters']['skt_volume']:.6f} м³\n")
            f.write(f"Объем БЧ: {results['input_parameters']['bc_volume']:.6f} м³\n")
            f.write(f"Длина ДУ: {results['input_parameters']['du_length']:.4f} м\n\n")
            
            f.write("ОБЩИЕ ХАРАКТЕРИСТИКИ:\n")
            f.write(f"Длина ракеты: {self.L_total:.3f} м\n")
            f.write(f"Общий объем: {results['total_volume']:.6f} м³\n")
            f.write(f"Общая масса: {results['total_mass']:.2f} кг\n")
            f.write(f"Средняя плотность: {results['total_mass']/results['total_volume']:.1f} кг/м³\n")
            f.write(f"Центр масс: {results['overall_center_of_mass']:.3f} м от носа\n")
            f.write(f"Относительно: {results['overall_center_of_mass']/self.L_total:.3f} L\n\n")
            
            f.write("МОМЕНТЫ ИНЕРЦИИ (относительно ЦМ):\n")
            f.write(f"Ixx = {results['moments_of_inertia']['Ixx']:.4f} кг·м²\n")
            f.write(f"Iyy = {results['moments_of_inertia']['Iyy']:.4f} кг·м²\n")
            f.write(f"Izz = {results['moments_of_inertia']['Izz']:.4f} кг·м²\n\n")
            
            f.write("ХАРАКТЕРИСТИКИ КОМПОНЕНТОВ:\n")
            for comp, props in results['component_properties'].items():
                f.write(f"\n{comp}:\n")
                f.write(f"  Масса: {props['mass']:.2f} кг\n")
                f.write(f"  Объем: {props['volume']:.6f} м³\n")
                f.write(f"  Плотность: {props['mass']/props['volume']:.0f} кг/м³\n")
                f.write(f"  Центр масс: {props['center_of_mass']:.3f} м\n")
                f.write(f"  Доля массы: {props.get('mass_fraction', 0)*100:.1f}%\n")
            
            f.write("\nГРАНИЦЫ ОТСЕКОВ:\n")
            for comp, bounds in results['boundaries'].items():
                if comp in ['СКТ', 'БЧ']:
                    f.write(f"{comp}: {bounds['start']:.3f} - {bounds['end']:.3f} м "
                           f"(L={bounds['calculated_length']:.3f} м, V={bounds['actual_volume']:.6f} м³)\n")
                else:  # ДУ
                    f.write(f"{comp}: {bounds['start']:.3f} - {bounds['end']:.3f} м "
                           f"(L={bounds['target_length']:.3f} м, V={bounds['actual_volume']:.6f} м³)\n")
            
            f.write("\nКОНСОЛИ:\n")
            for position in ['front', 'rear']:
                props = results['console_properties'][position]
                if props['count'] > 0:
                    f.write(f"\n{position.upper()} ПОЯС ({props['type']}):\n")
                    f.write(f"  Количество: {props['count']} шт\n")
                    f.write(f"  Масса одной: {props['single_mass']:.3f} кг\n")
                    f.write(f"  Общая масса: {props['total_mass']:.2f} кг\n")
                    f.write(f"  Позиция установки: {props['installation_position']:.3f} м\n")
                    f.write(f"  Центр масс: {props['center_of_mass_global']:.3f} м\n")
        
        print(f"Результаты сохранены в файл: {filename}")




# ============================================================================
# V7 — ИСПРАВЛЕННАЯ ВЕРСИЯ
# ============================================================================
# Основная идея V7:
# 1) дискретизация не выходит за пределы заданного интервала;
# 2) обратные границы запрещены, abs(x_end-x_start) не используется;
# 3) границы СКТ/БЧ определяются по тому же дискретному интегратору, который
#    затем используется для построения массы;
# 4) проверяется, что все отсеки помещаются в корпус;
# 5) эффективная плотность ДУ определяется по фактическому геометрическому
#    объёму ДУ, а не по цилиндрическому приближению;
# 6) консоли добавляются как настоящие точечные массы — без искусственного
#    радиуса и без ложного вклада в объём корпуса;
# 7) относительный ЦМ интегратора нормируется на ту же длину L_total, которая
#    используется внутри калькулятора.
#
# Важно: модель корпуса из исходного V6 сохранена по смыслу: СКТ, БЧ и ДУ
# занимают весь внутренний объём соответствующего продольного участка, а
# заданные объёмы используются для определения его границы. Это допущение
# модели не изменялось без исходных данных о реальной внутренней компоновке.
# ============================================================================


class DiscreteMassCalculatorV7:
    """Дискретный калькулятор массовых характеристик без искусственных элементов."""

    class ElementProperties:
        def __init__(self, x, r_inner, r_outer, density, dx,
                     element_type="disk", component=None):
            if dx <= 0:
                raise ValueError("dx должен быть > 0")
            if r_inner < 0 or r_outer < 0:
                raise ValueError("Радиусы не могут быть отрицательными")
            if r_outer < r_inner:
                raise ValueError("r_outer должен быть >= r_inner")
            if density < 0:
                raise ValueError("Плотность не может быть отрицательной")

            self.x = float(x)
            self.r_inner = float(r_inner)
            self.r_outer = float(r_outer)
            self.density = float(density)
            self.dx = float(dx)
            self.type = element_type
            self.component = component

            area = math.pi * (self.r_outer**2 - self.r_inner**2)
            self.volume = area * self.dx
            self.mass = self.volume * self.density
            self.static_moment = self.mass * self.x

            if self.mass > 0:
                self.Ixx_local = 0.5 * self.mass * (
                    self.r_outer**2 + self.r_inner**2
                )
                self.Iyy_local = (self.mass / 12.0) * (
                    3.0 * (self.r_outer**2 + self.r_inner**2) + self.dx**2
                )
                self.Izz_local = self.Iyy_local
                self.Ixx = self.Ixx_local
                self.Iyy = self.Iyy_local + self.mass * self.x**2
                self.Izz = self.Izz_local + self.mass * self.x**2
            else:
                self.Ixx_local = self.Iyy_local = self.Izz_local = 0.0
                self.Ixx = self.Iyy = self.Izz = 0.0


    def __init__(self, dx=0.001):
        if dx <= 0:
            raise ValueError("Шаг дискретизации dx должен быть > 0")
        self.dx = float(dx)
        self.elements = []
        self.point_masses = []

    def clear(self):
        self.elements.clear()
        self.point_masses.clear()

    def add_element(self, x, r_inner, r_outer, density,
                    element_type="disk", component=None, dx=None):
        """Добавить физический дисковый/кольцевой элемент."""
        actual_dx = self.dx if dx is None else float(dx)
        element = self.ElementProperties(
            x, r_inner, r_outer, density, actual_dx, element_type, component
        )
        self.elements.append(element)

    def add_point_mass(self, x, mass, component="точечная масса"):
        """Добавить точечную массу без искусственного объёма."""
        if mass < 0:
            raise ValueError("Точечная масса не может быть отрицательной")
        self.point_masses.append({
            "x": float(x),
            "mass": float(mass),
            "component": component,
        })

    def _validate_interval(self, x_start, x_end):
        if x_end < x_start:
            raise ValueError(
                f"Некорректный интервал: x_end={x_end} < x_start={x_start}"
            )
        if x_end == x_start:
            return 0.0
        return float(x_end - x_start)

    def _segment_centers(self, x_start, x_end):
        """Генератор центров; последний шаг всегда точно попадает в интервал."""
        length = self._validate_interval(x_start, x_end)
        if length == 0:
            return
        n = max(1, int(math.ceil(length / self.dx)))
        step = length / n
        for i in range(n):
            yield x_start + (i + 0.5) * step, step

    def add_cylinder_segment(self, x_start, x_end, r_inner, r_outer,
                             density, component=None):
        for x_center, step in self._segment_centers(x_start, x_end):
            self.add_element(
                x_center, r_inner, r_outer, density,
                "ring" if r_inner > 0 else "disk", component, dx=step
            )

    def add_cylinder_segment_variable(self, x_start, x_end,
                                      r_inner_func, r_outer_func,
                                      density, component=None):
        for x_center, step in self._segment_centers(x_start, x_end):
            r_inner = r_inner_func(x_center) if callable(r_inner_func) else r_inner_func
            r_outer = r_outer_func(x_center) if callable(r_outer_func) else r_outer_func
            self.add_element(
                x_center, r_inner, r_outer, density,
                "ring" if r_inner > 0 else "disk", component, dx=step
            )

    def calculate_total_properties(self):
        total_mass = sum(e.mass for e in self.elements)
        total_volume = sum(e.volume for e in self.elements)
        total_static = sum(e.static_moment for e in self.elements)
        total_Ixx = sum(e.Ixx for e in self.elements)
        total_Iyy = sum(e.Iyy for e in self.elements)
        total_Izz = sum(e.Izz for e in self.elements)

        components = {}

        def ensure_component(name):
            if name not in components:
                components[name] = {
                    "mass": 0.0,
                    "volume": 0.0,
                    "static_moment": 0.0,
                    "elements": [],
                }
            return components[name]

        for element in self.elements:
            if element.component:
                data = ensure_component(element.component)
                data["mass"] += element.mass
                data["volume"] += element.volume
                data["static_moment"] += element.static_moment
                data["elements"].append(element)

        # Точечные массы не имеют объёма и не получают искусственный Ixx.
        for pm in self.point_masses:
            x = pm["x"]
            m = pm["mass"]
            total_mass += m
            total_static += m * x
            total_Iyy += m * x**2
            total_Izz += m * x**2

            data = ensure_component(pm["component"])
            data["mass"] += m
            data["static_moment"] += m * x

        center_of_mass = total_static / total_mass if total_mass > 0 else 0.0
        Ixx_cm = total_Ixx
        Iyy_cm = total_Iyy - total_mass * center_of_mass**2
        Izz_cm = total_Izz - total_mass * center_of_mass**2

        for data in components.values():
            if data["mass"] > 0:
                data["center_of_mass"] = data["static_moment"] / data["mass"]
                data["mass_fraction"] = data["mass"] / total_mass
            else:
                data["center_of_mass"] = 0.0
                data["mass_fraction"] = 0.0

        return {
            "total_mass": total_mass,
            "total_volume": total_volume,
            "center_of_mass": center_of_mass,
            "Ixx": Ixx_cm,
            "Iyy": Iyy_cm,
            "Izz": Izz_cm,
            "Ixx_origin": total_Ixx,
            "Iyy_origin": total_Iyy,
            "Izz_origin": total_Izz,
            "elements_count": len(self.elements),
            "point_masses_count": len(self.point_masses),
            "mass_density": total_mass / total_volume if total_volume > 0 else 0.0,
            "components": components,
        }


class MassPropertiesCalculatorV7(MassPropertiesCalculatorV6):
    """Исправленная версия MassPropertiesCalculatorV6."""

    def __init__(self, fuselage_geometry, config_scheme="default",
                 num_front_consoles=0, num_rear_consoles=0,
                 skt_volume=None, bc_volume=None, du_length=None,
                 fuel_mass=18.52, fuel_density=1650, dx=0.001):
        # Вызов V6 нужен для сохранения совместимости с существующими
        # WingGeometry/RudderGeometry и параметрами проекта.
        super().__init__(
            fuselage_geometry, config_scheme,
            num_front_consoles, num_rear_consoles,
            skt_volume=skt_volume,
            bc_volume=bc_volume,
            du_length=du_length,
            fuel_mass=fuel_mass,
            fuel_density=fuel_density,
        )

        self.disc_calc = DiscreteMassCalculatorV7(dx=dx)
        self._validate_inputs()
        # Плотность ДУ определяется после расчёта её фактической границы.

    def _validate_inputs(self):
        if self.skt_volume < 0:
            raise ValueError("skt_volume не может быть отрицательным")
        if self.bc_volume < 0:
            raise ValueError("bc_volume не может быть отрицательным")
        if self.du_length < 0:
            raise ValueError("du_length не может быть отрицательной")
        if self.fuel_mass < 0:
            raise ValueError("fuel_mass не может быть отрицательной")
        if self.fuel_density < 0:
            raise ValueError("fuel_density не может быть отрицательной")
        if self.L_total <= 0:
            raise ValueError("Полная длина корпуса должна быть > 0")
        if self.d_m <= 0:
            raise ValueError("Диаметр миделя должен быть > 0")

    def _update_du_effective_density(self, du_volume):
        """Плотность ДУ по фактическому объёму рассчитанного участка."""

        if du_volume <= 0:
            self.densities["ДУ"] = 1000.0
            return

        structure_density = 1000.0
        structure_mass = structure_density * du_volume
        total_du_mass = structure_mass + self.fuel_mass
        self.densities["ДУ"] = total_du_mass / du_volume
        self.du_geometric_volume_for_density = du_volume

    def _calculate_volume_between(self, x_start, x_end):
        """Дискретный объём без выхода за пределы [x_start, x_end]."""
        if x_end < x_start:
            raise ValueError(
                f"x_end ({x_end}) меньше x_start ({x_start})"
            )
        if x_end == x_start:
            return 0.0

        volume = 0.0
        for x_center, step in self.disc_calc._segment_centers(x_start, x_end):
            r = self._calculate_radius_at_x(x_center)
            if r < 0:
                raise ValueError(f"Получен отрицательный радиус r={r} в x={x_center}")
            volume += math.pi * r**2 * step
        return volume

    def calculate_total_volume_disk_method(self):
        return self._calculate_volume_between(0.0, self.L_total)

    def _find_end_by_volume(self, x_start, target_volume, x_max):
        """Находит x_end, для которого дискретный объём равен target_volume."""
        if target_volume < 0:
            raise ValueError("Целевой объём не может быть отрицательным")
        if x_start < 0 or x_max < x_start:
            raise ValueError("Некорректные границы поиска")
        if target_volume == 0:
            return x_start

        available = self._calculate_volume_between(x_start, x_max)
        tolerance = max(1e-10, target_volume * 1e-8)
        if target_volume > available + tolerance:
            raise ValueError(
                f"Недостаточно объёма корпуса: требуется {target_volume:.9g} м³, "
                f"доступно {available:.9g} м³ на участке "
                f"[{x_start:.6f}, {x_max:.6f}] м."
            )

        lo, hi = x_start, x_max
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            v = self._calculate_volume_between(x_start, mid)
            if v < target_volume:
                lo = mid
            else:
                hi = mid
            if hi - lo < max(1e-10, self.disc_calc.dx * 1e-4):
                break

        return 0.5 * (lo + hi)

    def _section_name_at(self, x):
        if x <= self.geo.l_nos:
            return "оживальная часть"
        if x <= self.geo.l_nos + self.geo.l_cil:
            return "цилиндрическая часть"
        if x <= self.L_total:
            return "коническая кормовая часть"
        return "за пределами корпуса"

    def calculate_component_boundaries_by_volume_and_length(self):
        """Последовательно размещает СКТ, БЧ и ДУ с проверкой границ."""
        boundaries = {}
        current_x = 0.0

        # СКТ
        skt_end = self._find_end_by_volume(
            current_x, self.skt_volume, self.L_total
        )
        skt_actual = self._calculate_volume_between(current_x, skt_end)
        boundaries["СКТ"] = {
            "start": current_x,
            "end": skt_end,
            "target_volume": self.skt_volume,
            "calculated_length": skt_end - current_x,
            "actual_volume": skt_actual,
            "volume_error": skt_actual - self.skt_volume,
            "location_status": (
                f"{self._section_name_at((current_x + skt_end) / 2)}"
            ),
            "fits_completely": True,
        }
        current_x = skt_end

        # БЧ — больше не предполагаем автоматически, что она только в цилиндре.
        bc_end = self._find_end_by_volume(
            current_x, self.bc_volume, self.L_total
        )
        bc_actual = self._calculate_volume_between(current_x, bc_end)
        boundaries["БЧ"] = {
            "start": current_x,
            "end": bc_end,
            "target_volume": self.bc_volume,
            "calculated_length": bc_end - current_x,
            "actual_volume": bc_actual,
            "volume_error": bc_actual - self.bc_volume,
            "location_status": (
                f"{self._section_name_at((current_x + bc_end) / 2)}"
            ),
            "fits_completely": True,
        }
        current_x = bc_end

        # ДУ задаётся длиной.
        du_end = current_x + self.du_length
        if du_end > self.L_total + 1e-12:
            raise ValueError(
                f"ДУ не помещается в корпус: конец ДУ {du_end:.6f} м, "
                f"длина корпуса {self.L_total:.6f} м."
            )
        du_actual = self._calculate_volume_between(current_x, du_end)
        boundaries["ДУ"] = {
            "start": current_x,
            "end": du_end,
            "target_length": self.du_length,
            "calculated_volume": du_actual,
            "actual_volume": du_actual,
            "location_status": (
                f"{self._section_name_at((current_x + du_end) / 2)}"
            ),
            "fits_completely": True,
        }

        return boundaries

    def build_fuselage_disk_model(self, boundaries):
        self.disc_calc.clear()

        for component in ("СКТ", "БЧ", "ДУ"):
            bounds = boundaries[component]
            self.disc_calc.add_cylinder_segment_variable(
                bounds["start"], bounds["end"],
                r_inner_func=0.0,
                r_outer_func=self._calculate_radius_at_x,
                density=self.densities[component],
                component=component,
            )

        self._add_fuselage_shell()
        return True

    def _add_fuselage_shell(self):
        thickness = self.thickness["обшивка"]
        for x_center, step in self.disc_calc._segment_centers(0.0, self.L_total):
            r_outer = self._calculate_radius_at_x(x_center)
            r_inner = max(0.0, r_outer - thickness)
            if r_outer > 0 and r_outer > r_inner:
                self.disc_calc.add_element(
                    x_center, r_inner, r_outer,
                    self.densities["обшивка"],
                    "ring", "обшивка", dx=step
                )

    def _console_mass_and_cm(self, geometry, console_type="крыло"):
        """Сохраняет исходную эмпирическую формулу массы консоли V6."""
        if geometry is None or console_type == "none":
            return 0.0, 0.0

        if console_type == "крыло":
            if hasattr(geometry, "type") and geometry.type == "стреловидное":
                specific_mass = 2.5
            else:
                specific_mass = self.specific_mass["крыло"]
        else:
            specific_mass = self.specific_mass["руль"]

        S = geometry.S
        complexity_factor = 1.0

        if hasattr(geometry, "b"):
            complexity_factor *= (1.0 - 0.1 * (geometry.b - 1.0))
        if hasattr(geometry, "ny") and geometry.ny < 0.5:
            complexity_factor *= 1.2
        if hasattr(geometry, "chi") and geometry.chi > 0:
            complexity_factor *= (1.0 + geometry.chi / 90.0)

        mass = S * specific_mass * complexity_factor

        if hasattr(geometry, "has_flaps") and geometry.has_flaps:
            mass *= 1.15

        if hasattr(geometry, "xA") and hasattr(geometry, "bA"):
            cm_local = geometry.xA + 0.4 * geometry.bA
            if hasattr(geometry, "chi") and geometry.chi > 0:
                cm_local += 0.05 * geometry.bA
        else:
            cm_local = 0.25 * geometry.bA if hasattr(geometry, "bA") else 0.3

        mass *= 1.10  # пилоны/узлы крепления — исходное допущение V6
        return mass, cm_local

    def _add_console(self, position_name, installation_x):
        params = self.console_params[position_name]
        count = params["count"]
        console_type = params["type"]
        geometry = params["geometry"]

        result = {
            "type": console_type,
            "count": count,
            "single_mass": 0.0,
            "total_mass": 0.0,
            "center_of_mass_local": 0.0,
            "installation_position": installation_x,
            "center_of_mass_global": 0.0,
            "static_moment": 0.0,
        }

        if count <= 0 or geometry is None or console_type == "none":
            return result

        single_mass, local_cm = self._console_mass_and_cm(
            geometry, console_type
        )
        global_cm = installation_x + local_cm

        if global_cm < 0 or global_cm > self.L_total:
            raise ValueError(
                f"ЦМ {position_name}-й консоли ({global_cm:.6f} м) "
                f"выходит за пределы корпуса [0, {self.L_total:.6f}] м."
            )

        total_mass = single_mass * count
        result.update({
            "single_mass": single_mass,
            "total_mass": total_mass,
            "center_of_mass_local": local_cm,
            "center_of_mass_global": global_cm,
            "static_moment": total_mass * global_cm,
        })

        for _ in range(count):
            self.disc_calc.add_point_mass(
                global_cm, single_mass, component="консоли"
            )

        return result

    def _add_consoles_to_model(self):
        positions = {p["position"]: p["x_installation"] for p in self.console_positions}
        return {
            "front": self._add_console("front", positions.get("front", 0.0)),
            "rear": self._add_console("rear", positions.get("rear", 0.0)),
        }

    def calculate_mass_properties_disk_method(self):
        print("\n" + "=" * 80)
        print("РАСЧЕТ МАССОВЫХ ХАРАКТЕРИСТИК — V7")
        print(f"Шаг дискретизации: {self.disc_calc.dx:.6f} м")
        print("=" * 80)

        boundaries = self.calculate_component_boundaries_by_volume_and_length()

        # Только теперь известен фактический участок ДУ.
        # Топливо распределяется равномерно по этому объёму — то же
        # упрощающее допущение, что использовалось в V6.
        self._update_du_effective_density(boundaries["ДУ"]["calculated_volume"])
        self.build_fuselage_disk_model(boundaries)
        console_properties = self._add_consoles_to_model()
        total_props = self.disc_calc.calculate_total_properties()

        results = {
            "component_properties": {},
            "total_mass": total_props["total_mass"],
            "overall_center_of_mass": total_props["center_of_mass"],
            "mass_fractions": {},
            "boundaries": boundaries,
            "console_properties": console_properties,
            "total_volume": total_props["total_volume"],
            "total_length": self.L_total,
            "moments_of_inertia": {
                "Ixx": total_props["Ixx"],
                "Iyy": total_props["Iyy"],
                "Izz": total_props["Izz"],
            },
            "disk_method_details": {
                "elements_count": total_props["elements_count"],
                "point_masses_count": total_props["point_masses_count"],
                "dx": self.disc_calc.dx,
            },
            "input_parameters": {
                "skt_volume": self.skt_volume,
                "bc_volume": self.bc_volume,
                "du_length": self.du_length,
                "fuel_mass": self.fuel_mass,
                "fuel_density": self.fuel_density,
            },
            "checks": {},
        }

        for comp_name, comp_data in total_props["components"].items():
            results["component_properties"][comp_name] = {
                "mass": comp_data["mass"],
                "volume": comp_data["volume"],
                "center_of_mass": comp_data.get("center_of_mass", 0.0),
                "mass_fraction": comp_data.get("mass_fraction", 0.0),
            }
            results["mass_fractions"][comp_name] = comp_data.get("mass_fraction", 0.0)

        cm = results["overall_center_of_mass"]
        results["checks"] = {
            "center_of_mass_inside_body": 0.0 <= cm <= self.L_total,
            "boundaries_inside_body": all(
                0.0 <= b["start"] <= b["end"] <= self.L_total + 1e-12
                for b in boundaries.values()
            ),
            "skt_volume_error": boundaries["СКТ"]["volume_error"],
            "bc_volume_error": boundaries["БЧ"]["volume_error"],
            "du_length_error": (
                boundaries["ДУ"]["end"] - boundaries["ДУ"]["start"] - self.du_length
            ),
        }

        print(f"Общая масса: {results['total_mass']:.6f} кг")
        print(f"Центр масс: {cm:.6f} м от носа")
        print(f"Центр масс: {cm / self.L_total:.6f} L")
        print(f"Ixx = {results['moments_of_inertia']['Ixx']:.6f} кг·м²")
        print(f"Iyy = {results['moments_of_inertia']['Iyy']:.6f} кг·м²")
        print(f"Izz = {results['moments_of_inertia']['Izz']:.6f} кг·м²")

        return results

    def plot_disk_model(self, results=None):
        """Упрощённая визуализация V7 без фальшивого объёма консолей."""
        if results is None:
            results = self.calculate_mass_properties_disk_method()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.x_points, self.radii, label="Верхний контур")
        ax.plot(self.x_points, -self.radii, label="Нижний контур")

        for name in ("СКТ", "БЧ", "ДУ"):
            if name in results["boundaries"]:
                b = results["boundaries"][name]
                ax.axvline(b["start"], linestyle="--")
                ax.axvline(b["end"], linestyle="--")
                ax.text(
                    0.5 * (b["start"] + b["end"]),
                    self.d_m / 2,
                    name,
                    ha="center",
                    va="bottom",
                )

        cm = results["overall_center_of_mass"]
        ax.axvline(cm, linestyle="-", label=f"ЦМ = {cm:.4f} м")

        for side in ("front", "rear"):
            c = results["console_properties"][side]
            if c["count"] > 0:
                ax.plot(c["center_of_mass_global"], 0, "o")

        ax.set_xlabel("x, м")
        ax.set_ylabel("r, м")
        ax.set_title(f"Массовая модель V7 — схема {self.config_scheme}")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()
        return fig

    def save_disk_method_results(self, results, filename="rocket_mass_v7.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("МАССОВЫЕ ХАРАКТЕРИСТИКИ РАКЕТЫ — V7\n")
            f.write("=" * 70 + "\n")
            f.write(f"Схема: {self.config_scheme}\n")
            f.write(f"Длина: {self.L_total:.9f} м\n")
            f.write(f"Диаметр миделя: {self.d_m:.9f} м\n")
            f.write(f"Масса: {results['total_mass']:.9f} кг\n")
            f.write(f"ЦМ: {results['overall_center_of_mass']:.9f} м\n")
            f.write(f"ЦМ/L: {results['overall_center_of_mass']/self.L_total:.9f}\n")
            f.write(f"Ixx: {results['moments_of_inertia']['Ixx']:.9f}\n")
            f.write(f"Iyy: {results['moments_of_inertia']['Iyy']:.9f}\n")
            f.write(f"Izz: {results['moments_of_inertia']['Izz']:.9f}\n\n")

            f.write("ОТСЕКИ\n")
            for name, b in results["boundaries"].items():
                f.write(f"{name}: {b}\n")

            f.write("\nКОМПОНЕНТЫ\n")
            for name, c in results["component_properties"].items():
                f.write(f"{name}: {c}\n")

            f.write("\nПРОВЕРКИ\n")
            for name, value in results["checks"].items():
                f.write(f"{name}: {value}\n")

        print(f"Результаты сохранены в файл: {filename}")


class IntegratedRocketMassCalculatorV7:
    """Интеграция V7 с существующим inter.py."""

    def __init__(self, aircraft_aerodynamics_object,
                 skt_volume=None, bc_volume=None, du_length=None,
                 fuel_mass=18.52, fuel_density=1650, dx=0.001):
        self.aircraft = aircraft_aerodynamics_object
        self.fuselage_geo = aircraft_aerodynamics_object.fuselage_geo
        self.config_scheme = aircraft_aerodynamics_object.config_scheme
        self.num_front_consoles = aircraft_aerodynamics_object.num_front_consoles
        self.num_rear_consoles = aircraft_aerodynamics_object.num_rear_consoles

        self.mass_calculator = MassPropertiesCalculatorV7(
            self.fuselage_geo,
            self.config_scheme,
            self.num_front_consoles,
            self.num_rear_consoles,
            skt_volume=skt_volume,
            bc_volume=bc_volume,
            du_length=du_length,
            fuel_mass=fuel_mass,
            fuel_density=fuel_density,
            dx=dx,
        )
        self.original_x0_rel = aircraft_aerodynamics_object.x0_rel

    def calculate_and_set_center_of_mass(self, plot_results=True,
                                         save_results=True):
        results = self.mass_calculator.calculate_mass_properties_disk_method()

        x_cm = results["overall_center_of_mass"]
        L_total = self.mass_calculator.L_total
        self.aircraft.x0_rel = x_cm / L_total

        if plot_results:
            self.mass_calculator.plot_disk_model(results)

        if save_results:
            filename = (
                f"rocket_mass_v7_{self.config_scheme}_"
                f"{self.num_front_consoles}_{self.num_rear_consoles}.txt"
            )
            self.mass_calculator.save_disk_method_results(results, filename)

        return results


# Для минимальных изменений в существующем проекте можно использовать V7,
# явно импортируя эти классы:
# from mass_properties_calculator_v7 import MassPropertiesCalculatorV7
# from mass_properties_calculator_v7 import IntegratedRocketMassCalculatorV7


def example_usage_v7(plot_results=False, save_results=True):
    """
    Пример запуска V7 по той же схеме, которая использовалась в исходном V6.

    Реальные FuselageGeometry и AircraftAerodynamics берутся из проекта.
    Параметры СКТ/БЧ/ДУ ниже оставлены такими же, как в исходном примере V6,
    чтобы можно было сразу проверить запуск.
    """
    from inter import AircraftAerodynamics
    from Krilo_class import WingGeometry

    print("\n" + "=" * 80)
    print("ЗАПУСК ТЕСТА MASS_PROPERTIES_CALCULATOR_V7")
    print("=" * 80)

    wing_geo = WingGeometry()

    aircraft = AircraftAerodynamics(
        H=wing_geo.H,
        x0_rel=0.5,
        config_scheme="no_rudders",
        num_front_consoles=0,
        num_rear_consoles=6,
    )

    skt_volume = 0.00345
    bc_volume = 0.00345
    du_length = 2.0

    print("Входные параметры теста:")
    print(f"  СКТ: {skt_volume:.6f} м³")
    print(f"  БЧ : {bc_volume:.6f} м³")
    print(f"  ДУ : {du_length:.6f} м")
    print(f"  Схема: {aircraft.config_scheme}")
    print(
        f"  Консоли: передние={aircraft.num_front_consoles}, "
        f"задние={aircraft.num_rear_consoles}"
    )

    integrator = IntegratedRocketMassCalculatorV7(
        aircraft,
        skt_volume=skt_volume,
        bc_volume=bc_volume,
        du_length=du_length,
        fuel_mass=18.52,
        fuel_density=1650,
        dx=0.001,
    )

    results = integrator.calculate_and_set_center_of_mass(
        plot_results=plot_results,
        save_results=save_results,
    )

    print("\n" + "-" * 80)
    print("ИТОГ V7")
    print("-" * 80)
    print(f"Общая масса:          {results['total_mass']:.6f} кг")
    print(f"Абсолютный ЦМ:        {results['overall_center_of_mass']:.6f} м")
    print(f"Относительный ЦМ:     {aircraft.x0_rel:.6f}")
    print(
        f"Ixx относительно ЦМ:  "
        f"{results['moments_of_inertia']['Ixx']:.6f} кг·м²"
    )
    print(
        f"Iyy относительно ЦМ:  "
        f"{results['moments_of_inertia']['Iyy']:.6f} кг·м²"
    )
    print(
        f"Izz относительно ЦМ:  "
        f"{results['moments_of_inertia']['Izz']:.6f} кг·м²"
    )

    print("\nГраницы отсеков:")
    for name, bounds in results["boundaries"].items():
        print(
            f"  {name}: "
            f"{bounds['start']:.6f} ... {bounds['end']:.6f} м"
        )

    print("\nМассы компонентов:")
    for name, data in results["component_properties"].items():
        print(f"  {name}: {data.get('mass', 0.0):.6f} кг")

    print("\nПроверки:")
    for name, value in results["checks"].items():
        print(f"  {name}: {value}")

    print("\n" + "=" * 80)
    print("ТЕСТ V7 ЗАВЕРШЕН")
    print("=" * 80)

    return aircraft, results


if __name__ == "__main__":
    try:
        example_usage_v7(plot_results=False, save_results=True)
    except Exception as exc:
        print("\n" + "!" * 80)
        print("ОШИБКА ПРИ ЗАПУСКЕ V7")
        print("!" * 80)
        print(f"{type(exc).__name__}: {exc}")
        print(
            "\nЕсли это ошибка исходной геометрии/параметров проекта, "
            "пришли текст ошибки — разберем ее отдельно."
        )
        raise
