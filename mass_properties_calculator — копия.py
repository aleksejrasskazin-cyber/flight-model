# [file name]: mass_properties_calculator_v6_disk_method.py
"""
Модуль для расчета массовых характеристик методом дискретизации (тонких дисков)
Интеграция с inter.py для расчета центра масс ракеты
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
                 num_front_consoles=0, num_rear_consoles=0):
        """
        Инициализация с геометрией и схемой расположения консолей
        """
        self.geo = fuselage_geometry
        self.config_scheme = config_scheme
        self.num_front_consoles = num_front_consoles
        self.num_rear_consoles = num_rear_consoles
        
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
        
        # Стандартные плотности компонентов (кг/м³) - УПРОЩЕННЫЕ
        self.densities = {
            'СКТ': 1050.0,     # Система коррекции траектории
            'БЧ': 4400.0,      # Боевая часть
            'ДУ': 1000.0,      # Двигательная установка
            'обшивка': 2800.0, # Материал обшивки
            'консоли': 2500.0, # Средняя плотность консолей
        }
        
        # Толщины конструкций (м)
        self.thickness = {
            'обшивка': 0.002,  # Толщина обшивки фюзеляжа (2 мм)
        }
        
        # Доли объема от общего объема фюзеляжа (СКТ/БЧ/ДУ)
        self.volume_fractions = {
            'СКТ': 0.10,  # 10% (по вашему описанию)
            'БЧ': 0.40,   # 30% (по вашему описанию)
            'ДУ': 0.50    # 60% (по вашему описанию)
        }
        
        # Удельные массы консолей (кг/м²)
        self.specific_mass = {
            'крыло': 2.0,      # Легкие композитные крылья
            'руль': 8.0,       # Рули с приводами немного тяжелее
        }
        
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
    
    def calculate_component_boundaries_disk_method(self):
        """Определение границ отсеков методом дискретизации"""
        V_total = self.calculate_total_volume_disk_method()
        target_volumes = {
            'СКТ': V_total * self.volume_fractions['СКТ'],
            'БЧ': V_total * self.volume_fractions['БЧ'],
            'ДУ': V_total * self.volume_fractions['ДУ']
        }
        
        boundaries = {}
        current_x = 0
        
        # Для каждого компонента находим границу методом итерации
        for component in ['СКТ', 'БЧ', 'ДУ']:
            target_vol = target_volumes[component]
            
            # Бинарный поиск границы
            x_low = current_x
            x_high = self.L_total
            tolerance = 0.00001 * V_total  # Допуск 0.001% от объема
            
            for _ in range(100):  # Максимум 100 итераций
                x_mid = (x_low + x_high) / 2
                actual_vol = self._calculate_volume_between(current_x, x_mid)
                
                if abs(actual_vol - target_vol) < tolerance:
                    break
                
                if actual_vol < target_vol:
                    x_low = x_mid
                else:
                    x_high = x_mid
            
            x_boundary = (x_low + x_high) / 2
            
            boundaries[component] = {
                'start': current_x,
                'end': x_boundary,
                'target_volume': target_vol,
                'actual_volume': self._calculate_volume_between(current_x, x_boundary)
            }
            
            current_x = x_boundary
        
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
        Расчет массы консоли методом дискретизации
        
        Параметры:
        geometry - объект геометрии крыла/руля
        console_type - тип консоли ('крыло' или 'руль')
        """
        # Упрощенная модель консоли как трапециевидной пластины
        if console_type == 'крыло':
            specific_mass = self.specific_mass['крыло']
        else:
            specific_mass = self.specific_mass['руль']
        
        # Площадь консоли
        S = geometry.S  # м²
        
        # Масса
        mass = S * specific_mass
        
        # Для рулей добавляем массу привода
        if console_type == 'руль':
            mass *= 1.2
        
        # Центр масс консоли (примерно на 40% САХ от корневой хорды)
        if hasattr(geometry, 'xA') and hasattr(geometry, 'bA'):
            cm_local = geometry.xA + 0.4 * geometry.bA
        else:
            cm_local = 0.3  # Примерное значение
        
        return mass, cm_local
    
    def calculate_mass_properties_disk_method(self):
        """Расчет полных массовых характеристик методом дискретизации"""
        print("\n" + "="*80)
        print("РАСЧЕТ МАССОВЫХ ХАРАКТЕРИСТИК МЕТОДОМ ДИСКРЕТИЗАЦИИ")
        print(f"Шаг дискретизации: {self.disc_calc.dx:.4f} м")
        print("="*80)
        
        # 1. Определение границ отсеков
        print("\n1. Определение границ отсеков...")
        boundaries = self.calculate_component_boundaries_disk_method()
        
        # Вывод границ
        print("   Границы отсеков:")
        for component, bounds in boundaries.items():
            print(f"   {component}: от {bounds['start']:.4f} м до {bounds['end']:.4f} м")
            print(f"     Объем: {bounds['actual_volume']:.6f} м³ (цель: {bounds['target_volume']:.6f} м³)")
        
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
    #Iyy
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
                    r_outer=0.01,  # Маленький радиус для точечной массы
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
        
        # 1. Геометрия с дискретизацией
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
                
                mask = (self.x_points >= x_start) & (self.x_points <= x_end)
                if np.any(mask):
                    ax1.fill_between(self.x_points[mask], self.radii[mask], 
                                    -self.radii[mask], alpha=0.3, color=color, 
                                    label=f'{component} ({bounds["actual_volume"]:.4f} м³)')
        
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
        ax1.set_aspect('auto')
        
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
                f.write(f"{comp}: {bounds['start']:.3f} - {bounds['end']:.3f} м "
                       f"(V={bounds['actual_volume']:.6f} м³)\n")
            
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


# Класс для интеграции с inter.py
class IntegratedRocketMassCalculatorV6:
    """Класс для полной интеграции с основным кодом inter.py"""
    
    def __init__(self, aircraft_aerodynamics_object):
        self.aircraft = aircraft_aerodynamics_object
        self.fuselage_geo = aircraft_aerodynamics_object.fuselage_geo
        
        self.config_scheme = aircraft_aerodynamics_object.config_scheme
        self.num_front_consoles = aircraft_aerodynamics_object.num_front_consoles
        self.num_rear_consoles = aircraft_aerodynamics_object.num_rear_consoles
        
        self.mass_calculator = MassPropertiesCalculatorV6(
            self.fuselage_geo,
            self.config_scheme,
            self.num_front_consoles,
            self.num_rear_consoles
        )
        
        self.original_x0_rel = aircraft_aerodynamics_object.x0_rel
    
    def calculate_and_set_center_of_mass(self, plot_results=True, save_results=True):
        """Полный расчет с обновлением x0_rel"""
        print("\n" + "="*80)
        print("РАСЧЕТ ЦЕНТРА МАСС РАКЕТЫ МЕТОДОМ ДИСКРЕТИЗАЦИИ")
        print(f"Схема: {self.config_scheme}")
        print(f"Консоли: передние={self.num_front_consoles}, задние={self.num_rear_consoles}")
        print("="*80)
        
        # Расчет масс методом дискретизации
        mass_props = self.mass_calculator.calculate_mass_properties_disk_method()
        
        # Обновление x0_rel
        x_cm_absolute = mass_props['overall_center_of_mass']
        x_cm_relative = x_cm_absolute / self.fuselage_geo.l_f
        self.aircraft.x0_rel = x_cm_relative
        
        print(f"\nРЕЗУЛЬТАТЫ РАСЧЕТА:")
        print(f"Абсолютный ЦМ: {x_cm_absolute:.3f} м")
        print(f"Относительный ЦМ (x0_rel): {x_cm_relative:.3f}")
        print(f"Общая масса: {mass_props['total_mass']:.2f} кг")
        print(f"Общий объем: {mass_props['total_volume']:.6f} м³")
        print(f"Моменты инерции относительно ЦМ:")
        print(f"  Ixx = {mass_props['moments_of_inertia']['Ixx']:.4f} кг·м²")
        print(f"  Iyy = {mass_props['moments_of_inertia']['Iyy']:.4f} кг·м²")
        print(f"  Izz = {mass_props['moments_of_inertia']['Izz']:.4f} кг·м²")
        
        # Детали по консолям
        console_total = 0
        for pos in ['front', 'rear']:
            if mass_props['console_properties'][pos]['count'] > 0:
                console_total += mass_props['console_properties'][pos]['total_mass']
        
        if console_total > 0:
            print(f"Масса консолей: {console_total:.2f} кг ({console_total/mass_props['total_mass']*100:.1f}%)")
        
        # Графики
        if plot_results:
            self.mass_calculator.plot_disk_model(mass_props)
        
        # Сохранение
        if save_results:
            filename = f"rocket_mass_{self.config_scheme}_{self.num_front_consoles}_{self.num_rear_consoles}.txt"
            self.mass_calculator.save_disk_method_results(mass_props, filename)
        
        return mass_props


# Тестирование
def test_disk_method_calculation():
    """Тестирование метода дискретизации"""
    from Fesulyadge_class import FuselageGeometry
    
    print("=== ТЕСТИРОВАНИЕ МЕТОДА ДИСКРЕТИЗАЦИИ ===")
    
    fuselage_geo = FuselageGeometry()
    
    # Выводим параметры фюзеляжа для проверки
    print(f"Параметры фюзеляжа:")
    print(f"  Длина носовой части: {fuselage_geo.l_nos:.3f} м")
    print(f"  Длина цилиндрической части: {fuselage_geo.l_cil:.3f} м")
    print(f"  Длина кормовой части: {fuselage_geo.l_don:.3f} м")
    print(f"  Диаметр миделя: {fuselage_geo.d_m:.3f} м")
    print(f"  Диаметр кормы: {fuselage_geo.d_k:.3f} м")
    print(f"  Общая длина: {fuselage_geo.l_f:.3f} м")
    
    # Тестируем разные схемы
    test_cases = [
        ("no_rudders", 0, 6, "Только 6 крыльев"),
        ("no_rudders", 4, 4, "4 крыла спереди и 4 сзади"),
        ("default", 4, 4, "4 руля спереди, 4 крыла сзади"),
        ("no_rudders_wings", 0, 0, "Без консолей"),
    ]
    
    for config, front, rear, desc in test_cases:
        print(f"\n{'='*70}")
        print(f"Схема: {config} - {desc}")
        print(f"Консоли: {front} передних, {rear} задних")
        print(f"{'='*70}")
        
        calculator = MassPropertiesCalculatorV6(fuselage_geo, config, front, rear)
        mass_props = calculator.calculate_mass_properties_disk_method()
        
        print(f"Общая масса: {mass_props['total_mass']:.2f} кг")
        print(f"Центр масс: {mass_props['overall_center_of_mass']:.3f} м")
        print(f"Относительно: {mass_props['overall_center_of_mass']/calculator.L_total:.3f} L")
        print(f"Общий объем: {mass_props['total_volume']:.6f} м³")
        
        # Масса консолей
        console_mass = 0
        for pos in ['front', 'rear']:
            if mass_props['console_properties'][pos]['count'] > 0:
                console_mass += mass_props['console_properties'][pos]['total_mass']
        
        if console_mass > 0:
            print(f"Масса консолей: {console_mass:.2f} кг ({console_mass/mass_props['total_mass']*100:.1f}%)")
        
        # Объемные доли
        print(f"\nОбъемные доли:")
        for component in ['СКТ', 'БЧ', 'ДУ']:
            if component in mass_props['component_properties']:
                vol = mass_props['component_properties'][component]['volume']
                mass = mass_props['component_properties'][component]['mass']
                print(f"{component}: {vol:.6f} м³, масса: {mass:.2f} кг, плотность: {mass/vol:.0f} кг/м³")
        
        # Построение графиков
        if config == "no_rudders" and rear > 0:
            print("\nПостроение графиков дискретной модели...")
            calculator.plot_disk_model(mass_props)
            
            # Сохранение
            calculator.save_disk_method_results(mass_props, f"mass_disk_{config}_{front}_{rear}.txt")
    
    print("\n=== ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===")


# Пример использования с inter.py
def integrate_with_inter_example():
    """Пример интеграции с основным кодом inter.py"""
    from inter import AircraftAerodynamics
    from Krilo_class import WingGeometry
    
    print("=== ИНТЕГРАЦИЯ С INTER.PY ===")
    
    # Создаем объект аэродинамики (как в inter.py)
    wing_geo = WingGeometry()
    aircraft = AircraftAerodynamics(
        H=wing_geo.H, 
        x0_rel=0.5, 
        config_scheme="no_rudders_wings",
        num_front_consoles=0, 
        num_rear_consoles=0
    )
    
    # Создаем интеграционный калькулятор
    integrator = IntegratedRocketMassCalculatorV6(aircraft)
    
    # Расчет центра масс
    mass_props = integrator.calculate_and_set_center_of_mass(
        plot_results=True,
        save_results=True
    )
    
    # Проверяем обновленный x0_rel
    print(f"\nОбновленный x0_rel: {aircraft.x0_rel:.3f}")
    print(f"Старый x0_rel: {integrator.original_x0_rel:.3f}")
    
    return aircraft, mass_props


if __name__ == "__main__":
    # Запуск тестирования
    #test_disk_method_calculation()
    
    # Пример интеграции (раскомментировать при необходимости)
    integrate_with_inter_example()