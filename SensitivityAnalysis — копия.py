# SensitivityAnalysis.py
import numpy as np
import math
from CalcTrajectory import CalcTrajectory

class SensitivityAnalysis:
    """
    Анализ чувствительности траектории к возмущающим факторам
    Метод линеаризации (малые возмущения ±1 единица)
    """
    
    def __init__(self, calcParams, inits, limits, CData, NData, RData, MethodData):
        # Сохраняем ссылки на все данные
        self.calcParams = calcParams
        self.inits = inits
        self.limits = limits
        self.CData = CData
        self.NData = NData
        self.RData = RData
        self.MethodData = MethodData
        
        # Сохраняем копии для восстановления (будет заполнено позже)
        self._store_nominal_parameters()
        self.hit_points = []  # Список для хранения (x, z) всех прогонов
        # Результаты
        self.sensitivity = {}
        self.deviations = {}
    
    def _store_nominal_parameters(self):
        """Сохраняем копии всех параметров, которые можем менять"""
        self.nominal = {
            'm0': self.RData.m_r0,
            'x_cm': self.RData.x_O,
            'Theta_r0': self.inits.Theta_r0,
            'Psi_r0': self.inits.Psi_r0,
            'v_r0': self.inits.v_r0,
            'omega_x0': self.inits.omega_x0,
            'gamma_r0': self.inits.gamma_r0,
            'k_cx': 1.0,
            'k_cy': 1.0,
        }
        
        # Сохраняем таблицу МЦХ
        self.nominal['MC_tabl'] = [row[:] for row in self.RData.MC_tabl]
        
        # Сохраняем таблицу тяги-расхода
        self.nominal['t_P_G_tabl'] = [row[:] for row in self.RData.t_P_G_tabl]
        
        # Вычисляем массу топлива и удельный импульс
        self.nominal['fuel_mass'] = self._get_fuel_mass()
        self.nominal['Isp'] = self._get_Isp()
        self.nominal['total_impulse'] = self._get_total_impulse()
    
    def _restore_nominal(self):
        """Восстанавливаем все номинальные параметры"""
        self.RData.m_r0 = self.nominal['m0']
        self.RData.x_O = self.nominal['x_cm']
        self.RData.MC_tabl = [row[:] for row in self.nominal['MC_tabl']]
        self.RData.t_P_G_tabl = [row[:] for row in self.nominal['t_P_G_tabl']]
        
        self.inits.Theta_r0 = self.nominal['Theta_r0']
        self.inits.Psi_r0 = self.nominal['Psi_r0']
        self.inits.v_r0 = self.nominal['v_r0']
        self.inits.omega_x0 = self.nominal['omega_x0']
        self.inits.gamma_r0 = self.nominal['gamma_r0']
        
        # Убираем множители АДХ
        if hasattr(self.RData, 'k_cx'):
            self.RData.k_cx = 1.0
        if hasattr(self.RData, 'k_cy'):
            self.RData.k_cy = 1.0
    
    def _get_fuel_mass(self):
        """Вычисление массы топлива из таблицы тяги-расхода"""
        fuel_mass = 0.0
        t_P_G = self.RData.t_P_G_tabl
        
        for i in range(len(t_P_G)):
            t_start, P, G_t = t_P_G[i]
            if P > 0 and G_t > 0:
                # Ищем конец работы (следующая запись с нулевой тягой)
                for j in range(i + 1, len(t_P_G)):
                    if t_P_G[j][1] == 0:
                        t_end = t_P_G[j][0]
                        fuel_mass += G_t * (t_end - t_start)
                        break
        print('fuel_mass', fuel_mass)
        return fuel_mass
    
    def _get_total_impulse(self):
        """Полный импульс тяги"""
        total = 0.0
        t_P_G = self.RData.t_P_G_tabl
        
        for i in range(len(t_P_G)):
            t_start, P, G_t = t_P_G[i]
            if P > 0:
                for j in range(i + 1, len(t_P_G)):
                    if t_P_G[j][1] == 0:
                        t_end = t_P_G[j][0]
                        total += P * (t_end - t_start)
                        break
        return total
    
    def _get_Isp(self):
        """Расчёт удельного импульса (в секундах)"""
        fuel_mass = self._get_fuel_mass()
        total_impulse = self._get_total_impulse()
        if fuel_mass > 0:
            print('ИМПУЛЬС', total_impulse / fuel_mass)
            return total_impulse / fuel_mass
        return 0.0
    
    def _run_trajectory(self):
        """Запуск расчёта траектории с текущими параметрами"""
        result = []
        try:
            CalcTrajectory(self.calcParams, self.inits, self.limits,
                          self.CData, self.NData, self.RData,
                          self.MethodData, result)
        except Exception as e:
            print(f"Ошибка при расчёте траектории: {e}")
            return {'x_r': [0], 'z_r': [0], 't': [0]}
        
        if result:
            x_r = np.array([r.x_r for r in result])
            z_r = np.array([r.z_r for r in result])
            t = np.array([r.t for r in result])
            
            # Сохраняем конечную точку (последние координаты)
            self.hit_points.append((x_r[-1], z_r[-1]))
            
            return {'x_r': x_r, 'z_r': z_r, 't': t}
        return {'x_r': [0], 'z_r': [0], 't': [0]}
    
    # ========== ФУНКЦИИ ПРИМЕНЕНИЯ ВОЗМУЩЕНИЙ ==========
    
    def _apply_mass_perturbation(self, delta):
        """Изменение стартовой массы (delta в кг)"""
        self.RData.m_r0 = self.nominal['m0'] + delta
        # Обновляем таблицу МЦХ
        old_row = self.nominal['MC_tabl'][0]
        self.RData.MC_tabl = [[self.RData.m_r0, old_row[1], old_row[2], old_row[3], old_row[4]]]
    
    def _apply_fuel_mass_perturbation(self, delta):
        """Изменение массы топлива (delta в кг)"""
        nominal_fuel = self.nominal['fuel_mass']
        
        # ===== ПРОВЕРКА НА НУЛЕВОЙ ПАРАМЕТР =====
        if nominal_fuel == 0:
            print("  ВНИМАНИЕ: масса топлива равна 0, возмущение не применяется")
            return
        
        new_fuel = nominal_fuel + delta
        # Пересчитываем расход топлива
        t_work = 2.68  # время работы из вашей таблицы #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        new_G_t = new_fuel / t_work
        
        # Обновляем таблицу тяги-расхода
        for i, row in enumerate(self.RData.t_P_G_tabl):
            if row[0] <= t_work and row[2] != 0:
                self.RData.t_P_G_tabl[i][2] = new_G_t
        
        # Обновляем стартовую массу
        delta_mass = new_fuel - nominal_fuel
        self.RData.m_r0 = self.nominal['m0'] + delta_mass
        # Обновляем МЦХ
        old_row = self.nominal['MC_tabl'][0]
        self.RData.MC_tabl = [[self.RData.m_r0, old_row[1], old_row[2], old_row[3], old_row[4]]]
    
    def _apply_Isp_perturbation(self, delta):
        """Изменение удельного импульса (delta в м/с)"""
        nominal_Isp = self.nominal['Isp']
        
        # ===== ПРОВЕРКА НА НУЛЕВОЙ ПАРАМЕТР =====
        if nominal_Isp == 0:
            print("  ВНИМАНИЕ: удельный импульс равен 0, возмущение не применяется")
            return
        
        new_Isp = nominal_Isp + delta
        fuel_mass = self.nominal['fuel_mass']
        
        # Обновляем таблицу тяги-расхода
        for i, row in enumerate(self.RData.t_P_G_tabl):
            if row[1] > 0 and row[2] > 0:  # ненулевая тяга и расход
                new_P = new_Isp * row[2]  # * g
                self.RData.t_P_G_tabl[i][1] = new_P
    
    def _apply_cm_perturbation(self, delta):
        """Изменение координаты центра масс (delta в миллиметрах)"""
        # ЦМ может быть нулевым, но это допустимо
        delta_m = delta / 1000.0  # переводим мм в метры
        self.RData.x_O = self.nominal['x_cm'] + delta_m
        # Обновляем МЦХ
        old_row = self.nominal['MC_tabl'][0]
        self.RData.MC_tabl = [[old_row[0], self.RData.x_O, old_row[2], old_row[3], old_row[4]]]
    
    def _apply_angle_perturbation(self, delta, param_name):
        """Изменение начального угла (delta в градусах)"""
        # Углы могут быть нулевыми, но это допустимо
        delta_rad = math.radians(delta)
        if param_name == 'Theta_r0':
            self.inits.Theta_r0 = self.nominal['Theta_r0'] + delta_rad
        elif param_name == 'Psi_r0':
            self.inits.Psi_r0 = self.nominal['Psi_r0'] + delta_rad
    
    def _apply_velocity_perturbation(self, delta):
        """Изменение начальной скорости (delta в м/с)"""
        # Скорость может быть нулевой, но это допустимо
        self.inits.v_r0 = self.nominal['v_r0'] + delta
    
    def _apply_omega_perturbation(self, delta):
        """Изменение начальной угловой скорости крена (delta в рад/с)"""
        # Угловая скорость может быть нулевой, но это допустимо
        self.inits.omega_x0 = self.nominal['omega_x0'] + delta
    
    def _apply_Cx_perturbation(self, delta):
        """Изменение Cx на delta процентов"""
        factor = 1.0 + delta / 100.0
        if not hasattr(self.RData, 'k_cx'):
            self.RData.k_cx = 1.0
        self.RData.k_cx = factor
    
    def _apply_Cy_perturbation(self, delta):
        """Изменение Cy на delta процентов"""
        factor = 1.0 + delta / 100.0
        if not hasattr(self.RData, 'k_cy'):
            self.RData.k_cy = 1.0
        self.RData.k_cy = factor
    
    # ========== ОСНОВНОЙ МЕТОД АНАЛИЗА ==========
    
    def define_perturbations(self):
        """Определяем список возмущающих факторов"""
        self.perturbations = [
            {
                'name': 'm0',
                'apply_func': self._apply_mass_perturbation,
                'unit': 'кг',
                'unit_value': 0.5,
                'nominal': self.nominal['m0']
            },
            {
                'name': 'fuel_mass',
                'apply_func': self._apply_fuel_mass_perturbation,
                'unit': 'кг',
                'unit_value': 0.1,
                'nominal': self.nominal['fuel_mass']
            },
            {
                'name': 'Isp',
                'apply_func': self._apply_Isp_perturbation,
                'unit': 'м/с',
                'unit_value': 10.0,
                'nominal': self.nominal['Isp']
            },
            {
                'name': 'x_cm',
                'apply_func': self._apply_cm_perturbation,
                'unit': 'мм',
                'unit_value': 5.0,
                'nominal': self.nominal['x_cm']
            },
            {
                'name': 'Theta_r0',
                'apply_func': lambda d: self._apply_angle_perturbation(d, 'Theta_r0'),
                'unit': 'градус',
                'unit_value': 0.1,
                'nominal': math.degrees(self.nominal['Theta_r0'])
            },
            {
                'name': 'Psi_r0',
                'apply_func': lambda d: self._apply_angle_perturbation(d, 'Psi_r0'),
                'unit': 'градус',
                'unit_value': 0.1,
                'nominal': math.degrees(self.nominal['Psi_r0'])
            },
            {
                'name': 'v_r0',
                'apply_func': self._apply_velocity_perturbation,
                'unit': 'м/с',
                'unit_value': 1.0,
                'nominal': self.nominal['v_r0']
            },
            {
                'name': 'omega_x0',
                'apply_func': self._apply_omega_perturbation,
                'unit': 'рад/с',
                'unit_value': 0.1,
                'nominal': self.nominal['omega_x0']
            },
            {
                'name': 'k_cx',
                'apply_func': self._apply_Cx_perturbation,
                'unit': '%',
                'unit_value': 1.0,
                'nominal': 1.0
            },
            {
                'name': 'k_cy',
                'apply_func': self._apply_Cy_perturbation,
                'unit': '%',
                'unit_value': 1.0,
                'nominal': 1.0
            },
        ]
    
    def run_sensitivity_analysis(self):
        """
        Выполняет анализ чувствительности для всех возмущающих факторов
        """
        print("\n" + "="*80)
        print("НОМИНАЛЬНЫЙ РАСЧЁТ (без возмущений)")
        print("="*80)
        
        self._restore_nominal()
        result_nominal = self._run_trajectory()
        
        if len(result_nominal['x_r']) == 0:
            print("ОШИБКА: номинальный расчёт не выполнен")
            return {}
        
        x_hit_nominal = result_nominal['x_r'][-1]
        z_hit_nominal = result_nominal['z_r'][-1]
        
        print(f"Номинальная дальность: {x_hit_nominal:.1f} м")
        print(f"Номинальное боковое отклонение: {z_hit_nominal:.1f} м")
        
        print("\n" + "="*80)
        print("РАСЧЁТ ЧУВСТВИТЕЛЬНОСТИ К ВОЗМУЩАЮЩИМ ФАКТОРАМ")
        print("="*80)
        
        # ===== ПРОВЕРКА НА НУЛЕВЫЕ ПАРАМЕТРЫ ПЕРЕД РАСЧЁТОМ =====
        zero_params = []
        for pert in self.perturbations:
            name = pert['name']
            nominal_value = pert['nominal']
            
            # Проверяем, является ли параметр нулевым или очень маленьким
            if abs(nominal_value) < 1e-10:
                zero_params.append(name)
                print(f"\n⚠ ВНИМАНИЕ: параметр '{name}' равен нулю")
                print(f"  -> Этот параметр будет исключён из анализа чувствительности")
        
        # Фильтруем возмущения, исключая нулевые параметры
        active_perturbations = []
        for pert in self.perturbations:
            if pert['name'] not in zero_params:
                active_perturbations.append(pert)
            else:
                # Для нулевых параметров устанавливаем нулевую чувствительность
                self.sensitivity[pert['name']] = {
                    'dx_per_unit': 0.0,
                    'dz_per_unit': 0.0,
                    'unit': pert['unit'],
                    'unit_value': pert['unit_value'],
                    'skipped': True,
                    'reason': 'Нулевой номинальный параметр'
                }
        
        if len(active_perturbations) == 0:
            print("\n⚠ ВСЕ ПАРАМЕТРЫ НУЛЕВЫЕ! Анализ чувствительности невозможен.")
            print("Проверьте исходные данные.")
            return self.sensitivity
        
        print(f"\nАктивных параметров для анализа: {len(active_perturbations)} из {len(self.perturbations)}")
        
        for pert in active_perturbations:
            name = pert['name']
            unit = pert['unit']
            unit_value = pert['unit_value']
            nominal_value = pert['nominal']
            
            print(f"\nФактор: {name} (номинал = {nominal_value:.3f} {unit}, единица = {unit_value} {unit})")
            
            # Прогон с +единица
            self._restore_nominal()
            pert['apply_func'](+unit_value)
            result_plus = self._run_trajectory()
            dx_plus = abs(result_plus['x_r'][-1] - x_hit_nominal) if len(result_plus['x_r']) > 0 else 0
            dz_plus = abs(result_plus['z_r'][-1] - z_hit_nominal) if len(result_plus['z_r']) > 0 else 0
            
            # Прогон с -единица
            self._restore_nominal()
            pert['apply_func'](-unit_value)
            result_minus = self._run_trajectory()
            dx_minus = abs(result_minus['x_r'][-1] - x_hit_nominal) if len(result_minus['x_r']) > 0 else 0
            dz_minus = abs(result_minus['z_r'][-1] - z_hit_nominal) if len(result_minus['z_r']) > 0 else 0
            
            # Усреднённая чувствительность
            sens_dx = ((dx_plus + dx_minus) / 2.0) / unit_value
            sens_dz = ((dz_plus + dz_minus) / 2.0) / unit_value
            
            self.sensitivity[name] = {
                'dx_per_unit': sens_dx,
                'dz_per_unit': sens_dz,
                'unit': unit,
                'unit_value': unit_value,
                'nominal': nominal_value,
                'skipped': False
            }
            
            print(f"  Чувствительность по дальности: {sens_dx:.3f} м/{unit}")
            print(f"  Чувствительность по боку: {sens_dz:.3f} м/{unit}")
        
        print("\n" + "="*80)
        print("СТАТИСТИКА ПО ТОЧКАМ ПОПАДАНИЙ (всех прогонов)")
        print("="*80)
        
        if self.hit_points:
            points = np.array(self.hit_points)
            x_points = points[:, 0]
            z_points = points[:, 1]
            
            x_center = np.mean(x_points)
            z_center = np.mean(z_points)
            sigma_x = np.std(x_points)
            sigma_z = np.std(z_points)
            cep = 0.5887 * np.sqrt(sigma_x**2 + sigma_z**2)
            
            print(f"Количество точек: {len(self.hit_points)}")
            print(f"Центр рассеивания (X, Z): ({x_center:.1f}, {z_center:.1f}) м")
            print(f"СКО по дальности (σ_x): {sigma_x:.1f} м")
            print(f"СКО по боку (σ_z): {sigma_z:.1f} м")
            print(f"КВО (CEP): {cep:.1f} м")
            
            # Сохраняем точки в файл для последующего построения графиков
            np.savetxt("hit_points.txt", np.column_stack((x_points, z_points)))
            print("\nТочки попаданий сохранены в 'hit_points.txt'")
        
        return self.sensitivity

    
    def calculate_dispersion(self, tolerances):
        # Начинаем с номинальной точки
        x_nominal = self.x_hit_nominal  # сохраните номинальную дальность
        z_nominal = self.z_hit_nominal  # сохраните номинальное боковое
        
        # Список всех точек с учетом допусков
        x_points = [x_nominal]
        z_points = [z_nominal]
        
        for pert in self.perturbations:
            name = pert['name']
            if name in tolerances and name in self.sensitivity:
                sens = self.sensitivity[name]
                tol = tolerances[name]
                
                # Вклад в дальность и бок
                delta_x = sens['dx_per_unit'] * tol
                delta_z = sens['dz_per_unit'] * tol
                
                # Точка с положительным допуском
                x_points.append(x_nominal + delta_x)
                z_points.append(z_nominal + delta_z)
                
                # Точка с отрицательным допуском
                x_points.append(x_nominal - delta_x)
                z_points.append(z_nominal - delta_z)
        
        # Теперь у вас есть точки с учетом ВСЕХ допусков
        # По ним измеряем эллипс
        axis_x = max(x_points) - min(x_points)
        axis_z = max(z_points) - min(z_points)
        
        Bd = axis_x / 8.0
        Bb = axis_z / 8.0
        
        # Сохраняем
        self.Bd = Bd
        self.Bb = Bb
        self.axis_x = axis_x
        self.axis_z = axis_z
        self.x_points_with_tolerances = x_points
        self.z_points_with_tolerances = z_points
        
        return {'Bd': Bd, 'Bb': Bb, 'axis_x': axis_x, 'axis_z': axis_z}