# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 22:42:36 2025

@author: Пользователь
"""

import numpy as np
import math
from typing import List, Tuple
import csv
from dataclasses import dataclass

 
from Headlines import *
from CalcTrajectory import *
from KinematicFunc2D import *
import numpy as np
from typing import List
import aircraft_data
from Fesulyadge_class import FuselageGeometry, AerodynamicCalculator as FuselageCalculator

     
def Linspace(a: float, b: float, num: int) -> List[float]:
    """
    Создает массив равномерно распределенных значений от a до b.
    
    Args:
        a: начальное значение
        b: конечное значение
        num: количество точек
        
    Returns:
        Список равномерно распределенных значений
    """
    if num <= 0:
        return []
    if num == 1:
        return [a]
    
    # Используем numpy для создания равномерного распределения
    return np.linspace(a, b, num).tolist()
# Константы
deg = math.pi / 180.0  
 
def main():
    
    import locale
    locale.setlocale(locale.LC_CTYPE, 'Russian')
    
    # Формируем структуры исходных данных
    calcParams = CalcParameters()  # Параметры расчёта
    calcParams.dt = 0.1        # шаг по времени, с
    calcParams.r_por = 10.0       # расстояние поражения, м
     
    inits = InitConditions()      # Начальные условия
    inits.Theta_c0 = 0.0 / deg    # угол направления вектора скорости цели, рад
    inits.v_c0 = 0.0           # скорость цели, м/с
    inits.x_c0 = 8000          # координата цели, м
    inits.y_c0 = 0           # координата цели, м
    inits.z_c0 = 0
    inits.Theta_n0 = 0.52
    # угол направления вектора скорости носителя, рад {если NaN, то равен Theta_r}  
    inits.v_n0 = 0            # скорость носителя, м/с  
    inits.x_pusk = 0.0            # координата пуска (начальная координата носителя и ракеты), м
    inits.y_pusk = 100.0         # координата пуска (начальная координата носителя и ракеты), м
    inits.z_pusk = 0.0 
    inits.Theta_r0 = np.deg2rad(10)  # (45 град = 0.785398 рад) угол направления вектора скорости ракеты, рад {если NaN, то определяется методом наведения}  
    inits.v_r0 = 0         # скорость ракеты, м/с  
    # ДОБАВЛЕНО: Начальные условия для вращения вокруг оси x
    inits.rpm = 0  # оборотов в минуту
    inits.omega_x0 = inits.rpm* 2 * np.pi / 60  # преобразование в рад/с
    inits.gamma_r0 =  np.deg2rad(0)  # начальный угол крена
    inits.Psi_r0 = np.deg2rad(0.1)
     
    limits = LimitConditions()    # Ограничения
    limits.n_ya_r_max = 100.0      # максимальная перегрузка ракеты
    limits.t_min = 1.0            # минимальное время полёта ракеты, с
    limits.t_max = 200.0          # максимальное время полёта ракеты, с
    limits.R_min = float('nan')   # минимальная дальность полёта ракеты, м {NaN: не проверяется}
    limits.R_max = float('nan')   # максимальная дальность полёта ракеты, м {NaN: не проверяется}  
    limits.y_r_min = 0 # минимальная высота полёта ракеты, м {NaN: не проверяется}
    limits.y_r_max = float('nan') # максимальная высота полёта ракеты, м {NaN: не проверяется}
    limits.epsilon_nc_min = float('nan') / deg  # минимальный угол наклона линии визирования носитель-цель, рад {NaN: не проверяется}
    limits.epsilon_nc_max = float('nan') / deg  # максимальный угол наклона линии визирования носитель-цель, рад {NaN: не проверяется}
    limits.v_r_min = 0.0        # минимальная скорость полёта ракеты, м/с {NaN: не проверяется}
    limits.v_r_max = 1000.0       # максимальная скорость полёта ракеты, м/с {NaN: не проверяется}
    
    CData = C_Data()              # Данные по цели
    CData.n_xa_c_pot = [          # тангенциальная перегрузка цели (таблица время-перегрузка)
        [0.0, 0.0],
        [30.0, 0.0]
    ]
    CData.n_ya_c_pot = [          # нормальная скоростная перегрузка цели (таблица время-перегрузка)  
        [0.0, 1.0],
        [10.0, 0.0],
        [10.01, 1.0]
    ]
    
    NData = N_Data()              # Данные по носителю
    NData.n_xa_n_pot = [          # тангенциальная перегрузка носителя (таблица время-перегрузка)
        [0.0, 0.0],
        [5.0, 0.0]
    ]
    NData.n_ya_n_pot = [          # нормальная скоростная перегрузка носителя (таблица время-перегрузка)  
        [0.0, 2.0],
        [10.0, 2.0]
    ]
    
    RData = R_Data()              # Данные по ракете
    RData.m_r0 = 61.7393196489447 
    eta_0 = 25
    P = eta_0 * RData.m_r0 * 9.8
    RData.t_P_G_tabl = [          # таблица время-тяга-расход
        [0.0, P, 6.889],
        [2.68, P, 6.889],      
#        [0.51, 6000, 2.5],       # баллиститное топливо с I_уд = 2400 
#        [1.5, 6000, 2.5],
#        [1.51, 0.0, 0.0],
#        [3.5, 0.0, 0.0],
#        [3.51, 10000, 3.0],     # смесевое топливо с I_уд = 3000
#        [5.5, 10000, 3.0],
        [2.68, 0.0, 0.0],
        [5.51, 0.0, 0.0]
    ]
    
    fuselage_geo = FuselageGeometry()
    fuselage_calc = FuselageCalculator(fuselage_geo)
    
     
    RData.L_tpk = 1.558 # Длина ТПК для С13 (Б13Л)
    RData.S_a = (np.pi*fuselage_geo.d_m**2/4 )*0.8#0.9     # площадь выходного сечения сопла, м^2
    RData.S_m = (np.pi*fuselage_geo.d_m**2/4 )        # площадь миделя, м^2
    RData.x_O = 1.208     # координата центра масс, к которой приведены моменты АДХ, м
    RData.L = fuselage_geo.l_f              # характерный линейный размер, м
    Iz=39.6344
    Ix=0.1123
    Iy=39.6344
    RData.AD_tabl = aircraft_data.aerodynamics_data
    # H(высота) - M(Мах) - delta_100a_0h(высотная поправка) - 100a_0_дн (коэффициент донного сопротивления)
    RData.AD_tabl2 = aircraft_data.AD_tabl2
    
    RData.MC_tabl = [             # таблица МЦХ(m_r - x_cm - Iz - Ix - Iy)
        #[28.8, 0.103, 10.13001, 95.5, 10.13001],    Для ракеты без топлива
        [RData.m_r0, RData.x_O, Iz, Ix, Iy]
    ]
    # Коэффициент демпфирования ТОЛЬКО для фезюляжа
    m_z_omega_z_val = fuselage_calc.m_z_omega_z(RData.x_O)


    RData.AD_tabl3 = [            # таблица АДХ(x_0 - M - m_z_omega_z - m_z_dot_alpha - m_x_omega_x)
         
        [RData.x_O, 0.5, m_z_omega_z_val, 0, 0],
        [RData.x_O, 0.8, m_z_omega_z_val, 0, 0],
        [RData.x_O, 0.9, m_z_omega_z_val, 0, 0],
        [RData.x_O, 1.3, m_z_omega_z_val, 0, 0],
        [RData.x_O, 1.5, m_z_omega_z_val, 0, 0],
        
        
    ]
    # RData.AD_tabl3 = [            # таблица АДХ(x_0 - M - m_z_omega_z - m_z_dot_alpha - m_x_omega_x)
         
    #     [RData.x_O, 0.5, 0, 0, 0],
    #     [RData.x_O, 0.8, 0, 0, 0],
    #     [RData.x_O, 0.9, 0, 0, 0],
    #     [RData.x_O, 1.3, 0, 0, 0],
    #     [RData.x_O, 1.5, 0, 0, 0],
        
        
    # ]
    
    MethodData = GuidanceMethod_Data()  # Данные о методе наведения
    MethodData.Method = GuidanceMethod.Null  # метод наведения
    MethodData.gamma_r_upr = float('nan') / deg  # угол упреждения, рад
    MethodData.k = float('nan')              # коэффициент пропорциональности
    MethodData.lambda_val = float('nan')     # коэффициент пропорциональности
    MethodData.C = float('nan')              # коэффициент в методе "С"
    MethodData.m = float('nan')              # коэффициент в методе спрямления
    
    # Объявляем структуру для результата
    result = []
    
    Mode = 0
    num = 300
    resultCode = "t_max"
    
    x = Linspace(50, 25, num)
    j = 0
    maxx = 0.0
    maxy = 0.0
    maxrc = 0.0
    
    if Mode == 1:
        while j < num:
            inits.x_c0 = x[j] * 1000
            resultCode = CalcTrajectory(calcParams, inits, limits, CData, NData, RData, MethodData, result)
            print(f"Результат: {resultCode}; {x[j] * 1000}")
            
            if resultCode == "0":
                print("Done! ")
                Mode = 0
                maxx = x[j] * 1000
                maxy = 10000
                maxrc = r(inits.x_pusk, inits.y_pusk, maxx, maxy)
                break
            j += 1
    
    
    if Mode == 0:  # Единичный расчёт
        resultCode = ""
        
        # Открываем файлы для записи результатов
        try:
            C_File = open("Result_C.gra", "w", encoding='utf-8')
            N_File = open("Result_N.gra", "w", encoding='utf-8')
            R_File = open("Result_R.gra", "w", encoding='utf-8')
            Kinematic_File = open("Result_Kinematic.gra", "w", encoding='utf-8')
            
            resultCode = CalcTrajectory(calcParams, inits, limits, CData, NData, RData, MethodData, result)
            
            # Записываем заголовки
            C_File.write("t\tTheta_c\tv_c\tx_c\ty_c\tz_c\tn_xa_c\tn_ya_c")
            N_File.write("t\tTheta_n\tv_n\tx_n\ty_n\tz_n\tn_xa_n\tn_ya_n")
            R_File.write("t\tTheta_r\tv_r\tx_r\ty_r\tn_xa_r\tn_ya_r\tm_r\tP\tG_t\tM\talpha\tdelta\tx_cm\tn_ya_rasp_plus\tn_ya_rasp_minus\ttheta_r\tomega_z\tdot_alpha\tI_z\tM_z\tn_ya_r_pot\tomega_x\tgamma_r\tI_x\tM_x\tY_a\tX_a\tZ_a\tM_y\tbeta\tPsi_r\tomega_y\tz_r\tgamma_a\tpsi_r\tX\tY\tZ\tM_z_st\tM_z_omega\tc_xa")

            
            Kinematic_File.write("t\tr_rc\tdot_r_rc\tddot_r_rc\tepsilon_rc\tdot_epsilon_rc\tddot_epsilon_rc" +
                                "\tr_nr\tdot_r_nr\tddot_r_nr\tepsilon_nr\tdot_epsilon_nr\tddot_epsilon_nr" +
                                "\tr_nc\tdot_r_nc\tddot_r_nc\tepsilon_nc\tdot_epsilon_nc\tddot_epsilon_nc" +
                                "\tDelta_r\tdot_Delta_r\tddot_Delta_r\tepsilon_l\tdot_epsilon_l\tddot_epsilon_l")
            
            print(f"Результат: {resultCode}")
            print(f"Начальное вращение: {inits.omega_x0:.2f} рад/с ({inits.omega_x0 * 60 / (2 * np.pi):.1f} об/мин)")
            print(f"Количество точек траектории: {len(result)}")
            
            if len(result) == 0:
                print("ОШИБКА: Результат расчета пуст!")
                # Закрываем файлы и выходим
                C_File.close()
                N_File.close()
                R_File.close()
                Kinematic_File.close()
                exit()
            
            v_max = 0.0
            ind_v_max = 0
            m_min = 1e9
            ind_m_min = 0
            
            # Находим максимальную скорость и минимальную массу
            for i, res in enumerate(result):
                if res.v_r > v_max:
                    v_max = res.v_r
                    ind_v_max = i
                
                if res.m_r < m_min:
                    m_min = res.m_r
                    ind_m_min = i
            
            print(f"Запись {len(result)} точек в файлы...")
            
            # Записываем данные
            for i, res in enumerate(result):
                # Запись данных носителя
                N_File.write(f"\n{res.t}\t{res.Theta_n}\t{res.v_n}\t{res.x_n}\t{res.y_n}\t{res.z_n}\t{res.n_xa_n}\t{res.n_ya_n}")
                
                # Запись данных ракеты

                
                R_File.write(f"\n{res.t}\t{res.Theta_r}\t{res.v_r}\t{res.x_r}\t{res.y_r}\t{res.n_xa_r}\t{res.n_ya_r}\t" +
                            f"{res.m_r}\t{res.P}\t{res.G_t}\t{res.M}\t{res.alpha}\t{res.delta}\t{res.x_cm}\t" +
                            f"{res.n_ya_rasp_plus}\t{res.n_ya_rasp_minus}\t{res.theta_r}\t{res.omega_z}\t" +
                            f"{res.dot_alpha}\t{res.I_z}\t{res.M_z}\t{res.n_ya_r_pot}\t" +
                            f"{getattr(res, 'omega_x', 0.0)}\t{getattr(res, 'gamma_r', 0.0)}\t" +
                            f"{getattr(res, 'I_x', 0.0)}\t{getattr(res, 'M_x', 0.0)}\t{res.Y_a}\t{res.X_a}\t{res.Z_a}\t{res.M_y}\t{res.beta}\t{res.Psi_r}\t" + 
                            f"{getattr(res, 'omega_y', 0.0)}\t{res.z_r}\t{res.gamma_a}\t{res.psi_r}\t{res.X}\t{res.Y}\t{res.Z}\t{res.M_z_st}\t{res.M_z_omega}\t{res.c_xa}")

                
                # Запись данных цели
                C_File.write(f"\n{res.t}\t{res.Theta_c}\t{res.v_c}\t{res.x_c}\t{res.y_c}\t{res.z_c}\t{res.n_xa_c}\t{res.n_ya_c}")
                
                # Запись кинематических данных
                Kinematic_File.write(f"\n{res.t}\t" +
                                    f"{res.r_rc}\t{res.dot_r_rc}\t{res.ddot_r_rc}\t{res.epsilon_rc}\t{res.dot_epsilon_rc}\t{res.ddot_epsilon_rc}\t" +
                                    f"{res.r_nr}\t{res.dot_r_nr}\t{res.ddot_r_nr}\t{res.epsilon_nr}\t{res.dot_epsilon_nr}\t{res.ddot_epsilon_nr}\t" +
                                    f"{res.r_nc}\t{res.dot_r_nc}\t{res.ddot_r_nc}\t{res.epsilon_nc}\t{res.dot_epsilon_nc}\t{res.ddot_epsilon_nc}\t" +
                                    f"{res.Delta_r}\t{res.dot_Delta_r}\t{res.ddot_Delta_r}\t{res.epsilon_l}\t{res.dot_epsilon_l}\t{res.ddot_epsilon_l}")
            
            print("Запись файлов завершена.")
            # ========== АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ И КУЧНОСТИ ==========
            # Этот блок можно включить/отключить флагом
            DO_SENSITIVITY_ANALYSIS = False  # поставьте False, если не нужно

            if DO_SENSITIVITY_ANALYSIS:
                from SensitivityAnalysis import SensitivityAnalysis
                
                print("\n" + "="*80)
                print("ЗАПУСК АНАЛИЗА ЧУВСТВИТЕЛЬНОСТИ")
                print("="*80)
                
                # Создаём объект анализа
                sensitivity = SensitivityAnalysis(calcParams, inits, limits, CData, NData, RData, MethodData)
                
                # Определяем список возмущающих факторов
                sensitivity.define_perturbations()
                
                # Запускаем анализ чувствительности
                sens_results = sensitivity.run_sensitivity_analysis()
                 
                if sens_results:
                    # Задаём допуски (это пример, подберите реальные значения)
                    tolerances = {
                        'm0': 1.0,        # кг
                        'fuel_mass': 0.5, # кг
                        'Isp': 25.0,       # м/с
                        'x_cm': 10.0,       # мм
                        'Theta_r0': 0.5,   # градусы
                        'Psi_r0': 0.5,     # градусы
                        'v_r0': 10.0,       # м/с
                        'omega_x0': 2.0,   # рад/с
                        'k_cx': 5.0,       # %
                        'k_cy': 5.0,       # %
                    }
                    
                    # Рассчитываем кучность
                    dispersion = sensitivity.calculate_dispersion(tolerances)
                    
                    # Можно сохранить результаты в файл
                    with open("sensitivity_results.txt", "w", encoding='utf-8') as f:
                        f.write("РЕЗУЛЬТАТЫ АНАЛИЗА ЧУВСТВИТЕЛЬНОСТИ\n")
                        f.write("="*60 + "\n\n")
                        
                        f.write("Чувствительность к возмущающим факторам:\n")
                        for name, sens in sens_results.items():
                            f.write(f"  {name}: {sens['dx_per_unit']:.2f} м/{sens['unit_value']:.3f}{sens['unit']} (дальность), {sens['dz_per_unit']:.8f} м/{sens['unit']} (бок)\n")
                        
                        f.write("\nДопуски и их вклад:\n")
                        for name, tol in tolerances.items():
                            if name in sens_results:
                                unit_value = sens_results[name]['unit_value']
                                factor = tol / unit_value
                                delta_x = sens_results[name]['dx_per_unit'] * factor
                                delta_z = sens_results[name]['dz_per_unit'] * factor
                                f.write(f"  {name}: допуск {tol} -> вклад {delta_x:.3f} м (дальность), {delta_z:.8f} м (бок)\n")
                        f.write(f"\n=== РЕАЛЬНЫЙ ЭЛЛИПС С УЧЕТОМ ДОПУСКОВ ===\n")
                        f.write(f"  Длина оси по дальности: {dispersion['axis_x']:.1f} м\n")
                        f.write(f"  Длина оси по боку: {dispersion['axis_z']:.1f} м\n")
                        f.write(f"  Срединное отклонение по дальности (Вд = ось/8): {dispersion['Bd']:.1f} м\n")
                        f.write(f"  Срединное отклонение по боку (Вб = ось/8): {dispersion['Bb']:.1f} м\n")
                        f.write(f"\nИТОГОВАЯ КУЧНОСТЬ:\n")
                        f.write(f"  СКО по дальности (σ_x): {dispersion['sigma_x']:.8f} м\n")
                        f.write(f"  СКО по боку (σ_z): {dispersion['sigma_z']:.8f} м\n")
                        f.write(f"  КВО (CEP): {dispersion['CEP']:.8f} м\n")
                    
                    print("\nРезультаты сохранены в 'sensitivity_results.txt'") 
                    
                    
        except Exception as e:
            print(f"Ошибка при записи файлов: {e}")
        
        finally:
            # Закрываем файлы
            if 'C_File' in locals():
                C_File.close()
            if 'N_File' in locals():
                N_File.close()
            if 'R_File' in locals():
                R_File.close()
            if 'Kinematic_File' in locals():
                Kinematic_File.close()
    

    
if __name__ == "__main__":
    main()
    
# Графики
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Чтение данных
Result_C = pd.read_csv('Result_C.gra', delimiter="\t")
Result_N = pd.read_csv('Result_N.gra', delimiter="\t")
Result_R = pd.read_csv('Result_R.gra', delimiter="\t")
Result_Kinematic = pd.read_csv('Result_Kinematic.gra', delimiter="\t")


print("\n=== ДИАГНОСТИКА ===")
print(f"Максимальный угол атаки: {np.rad2deg(max(abs(Result_R['alpha']))):.1f}°")
print(f"Максимальный угол тангажа: {np.rad2deg(max(abs(Result_R['theta_r']))):.1f}°")
print(f"Максимальная угловая скорость: {max(abs(Result_R['omega_z'])):.2f} рад/с")



# Создание фигур
plt.style.use('default')

# 1. Траектории с линиями визирования
fig1, ax1 = plt.subplots(figsize=(10, 8))
N_lv = 5  # количество линий визирования для построения

# Построение траекторий
ax1.plot(Result_R['x_r'], Result_R['y_r'], color='#9B1215', label='Ракета', linewidth=2)
ax1.plot(Result_C['x_c'], Result_C['y_c'], color='#004E9F', label='Цель', linewidth=2)
ax1.plot(Result_N['x_n'], Result_N['y_n'], color='black', label='Носитель', linewidth=2)

# Построение линий визирования
for i in range(N_lv):
    t_index = math.floor(len(Result_R['t']) / (N_lv) * (i))
    x_points = [Result_N['x_n'][t_index], Result_R['x_r'][t_index], Result_C['x_c'][t_index]]
    y_points = [Result_N['y_n'][t_index], Result_R['y_r'][t_index], Result_C['y_c'][t_index]]
    ax1.plot(x_points, y_points, color='black', linestyle='dashdot', linewidth=1, alpha=0.7)
    
 
ax1.set_xlabel('x, м')
ax1.set_ylabel('y, м')
ax1.set_title('Траектории')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

# 2. Координаты
fig2, ((ax2_1, ax2_2, ax2_3), (ax2_4, ax2_5, ax2_6))  = plt.subplots(2, 3, figsize=(16, 10))

# Координата x
ax2_1.plot(Result_R['t'], Result_R['x_r'], color='#9B1215', label='$x_р$')
ax2_1.plot(Result_C['t'], Result_C['x_c'], color='#004E9F', label='$x_ц$')
ax2_1.plot(Result_N['t'], Result_N['x_n'], color='black', label='$x_н$')
ax2_1.set_xlabel('t, с')
ax2_1.set_ylabel('x, м')
ax2_1.set_title('Координаты X')
ax2_1.grid(True, alpha=0.3)
ax2_1.legend()

# Координата y
ax2_2.plot(Result_R['t'], Result_R['y_r'], color='#9B1215', label='$y_р$')
ax2_2.plot(Result_C['t'], Result_C['y_c'], color='#004E9F', label='$y_ц$')
ax2_2.plot(Result_N['t'], Result_N['y_n'], color='black', label='$y_н$')
ax2_2.set_xlabel('t, с')
ax2_2.set_ylabel('y, м')
ax2_2.set_title('Координаты Y')
ax2_2.grid(True, alpha=0.3)
ax2_2.legend()

# Координата z
ax2_3.plot(Result_R['t'], Result_R['z_r'], color='#9B1215', label='$z_р$')
ax2_3.plot(Result_C['t'], Result_C['z_c'], color='#004E9F', label='$z_ц$')
ax2_3.plot(Result_N['t'], Result_N['z_n'], color='black', label='$z_н$')
ax2_3.set_xlabel('t, с')
ax2_3.set_ylabel('z, м')
ax2_3.set_title('Координаты Z')
ax2_3.grid(True, alpha=0.3)
ax2_3.legend()


# Координата y(x)
ax2_4.plot(Result_R['x_r'], Result_R['y_r'], color='#9B1215', label='$y(x)_р$')
ax2_4.plot(Result_C['x_c'], Result_C['y_c'], color='#004E9F', label='$y(x)_ц$')
ax2_4.set_xlabel('x, м')
ax2_4.set_ylabel('y, м')
ax2_4.set_title('Зависимость Y(x) - (Вид сбоку →)')
ax2_4.grid(True, alpha=0.3)
ax2_4.legend()

# Координата x(z)
ax2_5.plot(Result_R['z_r'], Result_R['x_r'], color='#9B1215', label='$x(z)_р$')
ax2_5.plot(Result_C['z_c'], Result_C['x_c'], color='#004E9F', label='$x(z)_ц$')
ax2_5.set_xlabel('z, м')
ax2_5.set_ylabel('x, м')
ax2_5.set_title('Зависимость X(z) - (Вид сверху ↑)')
ax2_5.grid(True, alpha=0.3)
ax2_5.legend()


# Координата y(z)
ax2_6.plot(Result_R['z_r'], Result_R['y_r'], color='#9B1215', label='р')
ax2_6.plot(Result_C['z_c'], Result_C['y_c'], color='#004E9F', label='ц')
ax2_6.plot(Result_N['z_n'], Result_N['y_n'], color='black', label='н')
ax2_6.set_xlabel('z, м')
ax2_6.set_ylabel('y, м')
ax2_6.set_title('Зависимость Y(z) - (Вид сзади (X))')
ax2_6.grid(True, alpha=0.3)
ax2_6.legend()


plt.tight_layout()
plt.show()
plt.close()

# Координаты в Трёхмерной картинке
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
N_lv = 5  # количество линий визирования для построения
# Построение траекторий в 3D
ax_3d.plot(Result_R['z_r'], Result_R['x_r'], Result_R['y_r'],
            color='#9B1215', label='Ракета', linewidth=2)
ax_3d.plot(Result_C['z_c'], Result_C['x_c'], Result_C['y_c'],
            color='#004E9F', label='Цель', linewidth=2)

for i in range(N_lv):
    t_index = math.floor(len(Result_R['t']) / N_lv * i)
    
    # Координаты точек для линии визирования (носитель → ракета → цель)
    x_points = [Result_R['x_r'][t_index],
                Result_C['x_c'][t_index]]
    y_points = [Result_R['y_r'][t_index],
                Result_C['y_c'][t_index]]
    z_points = [Result_R['z_r'][t_index],
                Result_C['z_c'][t_index]]
    
    ax_3d.plot(z_points, x_points, y_points,
                color='gray', linestyle='dashdot', linewidth=1.5, alpha=0.8)
ax_3d.set_xlabel('Z, м')
ax_3d.set_ylabel('X, м')
ax_3d.set_zlabel('Y, м')
ax_3d.set_title('Траектории и линии визирования в 3D')
ax_3d.legend(loc='upper left')
ax_3d.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
    
# 3. Скорость и углы
fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(12, 5))

# Скорость
ax3_1.plot(Result_R['t'], Result_R['v_r'], color='#9B1215', label='$v_р$')
ax3_1.plot(Result_C['t'], Result_C['v_c'], color='#004E9F', label='$v_ц$')
ax3_1.plot(Result_N['t'], Result_N['v_n'], color='black', label='$v_н$')
ax3_1.set_xlabel('t, с')
ax3_1.set_ylabel('v, м/с')
ax3_1.set_title('Скорость')
ax3_1.grid(True, alpha=0.3)
ax3_1.legend()

# Углы
ax3_2.plot(Result_R['t'], np.rad2deg(Result_R['Theta_r']), color='#9B1215', label='$\Theta_р$')
ax3_2.plot(Result_C['t'], np.rad2deg(Result_C['Theta_c']), color='#004E9F', label='$\Theta_ц$')
ax3_2.plot(Result_N['t'], np.rad2deg(Result_N['Theta_n']), color='black', label='$\Theta_н$')
ax3_2.plot(Result_R['t'], np.rad2deg(Result_R['theta_r']), color='#9B1215', linestyle='--', label='$\\theta_р$')
ax3_2.set_xlabel('t, с')
ax3_2.set_ylabel('$\Theta$, °')
ax3_2.set_title('Углы наклона траектории и тангажа')
ax3_2.grid(True, alpha=0.3)
ax3_2.legend()

plt.tight_layout()
plt.show()
plt.close()

# 4. Перегрузки
fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(12, 5))

# Тангенциальная перегрузка
ax4_1.plot(Result_R['t'], Result_R['n_xa_r'], color='#9B1215', label='$n_{x_a р}$')
ax4_1.plot(Result_C['t'], Result_C['n_xa_c'], color='#004E9F', label='$n_{x_a ц}$')
ax4_1.plot(Result_N['t'], Result_N['n_xa_n'], color='black', label='$n_{x_a н}$')
ax4_1.set_xlabel('t, с')
ax4_1.set_ylabel('$n_{x_a}$')
ax4_1.set_title('Тангенциальная перегрузка')
ax4_1.grid(True, alpha=0.3)
ax4_1.legend()

# Нормальная скоростная перегрузка
ax4_2.plot(Result_R['t'], Result_R['n_ya_r'], color='#9B1215', label='$n_{y_a р}$')
ax4_2.plot(Result_C['t'], Result_C['n_ya_c'], color='#004E9F', label='$n_{y_a ц}$')
ax4_2.plot(Result_N['t'], Result_N['n_ya_n'], color='black', label='$n_{y_a н}$')
if 'n_ya_rasp_plus' in Result_R.columns:
    ax4_2.plot(Result_R['t'], Result_R['n_ya_rasp_plus'], color='gray', linestyle='--', label='$n_{y_a}^{расп}$')
if 'n_ya_rasp_minus' in Result_R.columns:
    ax4_2.plot(Result_R['t'], Result_R['n_ya_rasp_minus'], color='gray', linestyle='--')
if 'n_ya_r_pot' in Result_R.columns:
    ax4_2.plot(Result_R['t'], Result_R['n_ya_r_pot'], color='#9B1215', linestyle='--', label='$n_{y_a}^{пот}$')
ax4_2.set_xlabel('t, с')
ax4_2.set_ylabel('$n_{y_a}$')
ax4_2.set_title('Нормальная скоростная перегрузка')
ax4_2.grid(True, alpha=0.3)
ax4_2.legend()

plt.tight_layout()
plt.show()
plt.close()

# 5. Отдельный график для нормальной перегрузки
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(Result_C['t'], Result_C['n_ya_c'], color='#004E9F', label='$n_{y_a ц}$')
ax5.plot(Result_R['t'], Result_R['n_ya_r'], color='#9B1215', label='$n_{y_a р}$')
ax5.set_xlabel('t, с')
ax5.set_ylabel('$n_{y_a}$')
ax5.set_title('Нормальная скоростная перегрузка')
ax5.grid(True, alpha=0.3)
ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.show()
plt.close()

# 6. Тяга и углы атаки
fig6, (ax6_1, ax6_2) = plt.subplots(1, 2, figsize=(12, 5))

# Тяга
ax6_1.plot(Result_R['t'], Result_R['P'], color='#9B1215', label='P')
ax6_1.set_xlabel('t, с')
ax6_1.set_ylabel('P, Н')
ax6_1.set_title('Тяга')
ax6_1.grid(True, alpha=0.3)
ax6_1.legend()

# Углы атаки
ax6_2.plot(Result_R['t'], np.rad2deg(Result_R['alpha']), color='green', label='$\\alpha$')
ax6_2.plot(Result_R['t'], np.rad2deg(Result_R['delta']), color='#9B1215', linestyle='--', label='$\\delta$')
ax6_2.plot(Result_R['t'], np.rad2deg(Result_R['beta']), color='blue', linestyle='--', label='$\\beta$')

ax6_2.set_xlabel('t, с')
ax6_2.set_ylabel('$\\alpha, \delta$, °')
ax6_2.set_title('Угол атаки и отклонения')
ax6_2.grid(True, alpha=0.3)
ax6_2.legend()

plt.tight_layout()
plt.show()
plt.close()

# 7. Массовый расход и число Маха
fig7, (ax7_1, ax7_2, ax7_3) = plt.subplots(1, 3, figsize=(14, 5))

# Массовый расход
ax7_1.plot(Result_R['t'], Result_R['G_t'], color='#9B1215', label='$G_т$')
ax7_1.set_xlabel('t, с')
ax7_1.set_ylabel('$G_т$, кг/с')
ax7_1.set_title('Массовый расход')
ax7_1.grid(True, alpha=0.3)
ax7_1.legend()

# Число Маха (от координаты X)
ax7_2.plot(Result_R['x_r'], Result_R['M'], color='#9B1215', label='M')
ax7_2.set_xlabel('x, м')
ax7_2.set_ylabel('M')
ax7_2.set_title('Число Маха (от координаты X)')
ax7_2.grid(True, alpha=0.3)
ax7_2.legend()

# Число Маха (от времени)
ax7_3.plot(Result_R['t'], Result_R['M'], color='#9B1215', label='M')
ax7_3.set_xlabel('t, с')
ax7_3.set_ylabel('M')
ax7_3.set_title('Число Маха (от времени)')
ax7_3.grid(True, alpha=0.3)
ax7_3.legend()

plt.tight_layout()
plt.show()
plt.close()

# 8. Масса и момент тангажа
fig8, (ax8_1, ax8_2) = plt.subplots(1, 2, figsize=(12, 5))

# Масса ракеты
ax8_1.plot(Result_R['t'], Result_R['m_r'], color='#9B1215', label='$m_р$')
ax8_1.set_xlabel('t, с')
ax8_1.set_ylabel('$m_р$, кг')
ax8_1.set_title('Масса ракеты')
ax8_1.grid(True, alpha=0.3)
ax8_1.legend()

# Момент тангажа
ax8_2.plot(Result_R['t'], Result_R['M_z'], color='#9B1215', label='$M_z$')
ax8_2.set_xlabel('t, с')
ax8_2.set_ylabel('$M_z$, Н·м')
ax8_2.set_title('Момент тангажа ракеты')
ax8_2.grid(True, alpha=0.3)
ax8_2.legend()

plt.tight_layout()
plt.show()
plt.close()

# 9. Центр масс и угловые скорости
fig9, (ax9_1, ax9_2) = plt.subplots(1, 2, figsize=(12, 5))

# Координата центра масс
ax9_1.plot(Result_R['t'], Result_R['x_cm'], color='#9B1215', label='$x_{цм}$')
ax9_1.set_xlabel('t, с')
ax9_1.set_ylabel('$x_{цм}$, м')
ax9_1.set_title('Координата центра масс ракеты')
ax9_1.grid(True, alpha=0.3)
ax9_1.legend()

# Угловые скорости
ax9_2.plot(Result_R['t'], Result_R['omega_z'] * 60 / (2 * np.pi), color='green', label='$\omega_z$')
ax9_2.plot(Result_R['t'], Result_R['dot_alpha'], color='#9B1215', linestyle='--', label='$\dot \\alpha$')
ax9_2.set_xlabel('t, с')
ax9_2.set_ylabel('$\omega_z, \dot \\alpha$, рад/с')
ax9_2.set_title('Угловые скорости тангажа и угла атаки')
ax9_2.grid(True, alpha=0.3)
ax9_2.legend()

plt.tight_layout()
plt.show()
plt.close()

# 10. Момент инерции и центр масс
fig10, (ax10_1, ax10_2) = plt.subplots(1, 2, figsize=(12, 5))

# Момент инерции
ax10_1.plot(Result_R['t'], Result_R['I_z'], color='#9B1215', label='$I_z$')
ax10_1.set_xlabel('t, с')
ax10_1.set_ylabel('$I_z$, м²·кг')
ax10_1.set_title('Момент инерции ракеты')
ax10_1.grid(True, alpha=0.3)
ax10_1.legend()

# Координата центра масс (дублирование для удобства)
ax10_2.plot(Result_R['t'], Result_R['x_cm'], color='#9B1215', label='$x_{цм}$')
ax10_2.set_xlabel('t, с')
ax10_2.set_ylabel('$x_{цм}$, м')
ax10_2.set_title('Координата центра масс ракеты')
ax10_2.grid(True, alpha=0.3)
ax10_2.legend()

plt.tight_layout()
plt.show()
plt.close()
# 11. ДОБАВЛЕНО: Графики для вращения вокруг оси x
fig11, ((ax11_1, ax11_2), (ax11_3, ax11_4)) = plt.subplots(2, 2, figsize=(12, 10))

# Угловая скорость крена (об/мин)
if 'omega_x' in Result_R.columns:
    ax11_1.plot(Result_R['t'], Result_R['omega_x'] * 60 / (2 * np.pi), color='purple', label='$\omega_x$')
    ax11_1.set_xlabel('t, с')
    ax11_1.set_ylabel('$\omega_x$, об/мин')
    ax11_1.set_title('Угловая скорость крена')
    ax11_1.grid(True, alpha=0.3)
    ax11_1.legend()

# Угол крена
if 'gamma_r' in Result_R.columns:
    ax11_2.plot(Result_R['t'], np.rad2deg(Result_R['gamma_r']), color='orange', label='$\gamma_r$')
    ax11_2.set_xlabel('t, с')
    ax11_2.set_ylabel('$\gamma_r$, °')
    ax11_2.set_title('Угол крена')
    ax11_2.grid(True, alpha=0.3)
    ax11_2.legend()

# Момент инерции вокруг оси x
if 'I_x' in Result_R.columns:
    ax11_3.plot(Result_R['t'], Result_R['I_x'], color='brown', label='$I_x$')
    ax11_3.set_xlabel('t, с')
    ax11_3.set_ylabel('$I_x$, м²·кг')
    ax11_3.set_title('Момент инерции вокруг оси x')
    ax11_3.grid(True, alpha=0.3)
    ax11_3.legend()


# Момент вокруг оси x
if 'M_x' in Result_R.columns:
    ax11_4.plot(Result_R['t'], Result_R['M_x'], color='red', label='$M_x$')
    ax11_4.set_xlabel('t, с')
    ax11_4.set_ylabel('$M_x$, Н·м')
    ax11_4.set_title('Момент вокруг оси x')
    ax11_4.grid(True, alpha=0.3)
    ax11_4.legend()
    
plt.tight_layout()
plt.show()
plt.close()

# 12. График угла тангажа от времени
fig12, (ax12_1, ax12_2) = plt.subplots(1, 2, figsize=(12, 5))
ax12_1.plot(Result_R['t'], np.rad2deg(Result_R['theta_r']), color='darkgreen', linewidth=2, label='$\\vartheta_r$ (угол тангажа)')
ax12_1.plot(Result_R['t'], np.rad2deg(Result_R['Theta_r']), color='#9B1215', linestyle='--', linewidth=1.5, label='$\\Theta_р$ (угол наклона траектории)')
ax12_1.plot(Result_R['t'], np.rad2deg(Result_R['Psi_r']), color='blue', linestyle='--', linewidth=1.5, label='$\\Psi_р$ (угол поворота траектории)')
ax12_1.set_xlabel('t, с')
ax12_1.set_ylabel('Угол, °')
ax12_1.set_title('Угол тангажа, наклона траектории, курса от времени')
ax12_1.grid(True, alpha=0.3)
ax12_1.legend(loc='best')

ax12_2.plot(Result_R['x_r'], np.rad2deg(Result_R['theta_r']), color='darkgreen', linewidth=2, label='$\\vartheta_r$ (угол тангажа)')
ax12_2.plot(Result_R['x_r'], np.rad2deg(Result_R['Theta_r']), color='#9B1215', linestyle='--', linewidth=1.5, label='$\\Theta_р$ (угол наклона траектории)')
ax12_2.plot(Result_R['x_r'], np.rad2deg(Result_R['Psi_r']), color='blue', linestyle='--', linewidth=1.5, label='$\\Psi_р$ (угол поворота траектории)')
ax12_2.set_xlabel('x, м')
ax12_2.set_ylabel('Угол, °')
ax12_2.set_title('Угол тангажа, наклона и поворота траектории от X')
ax12_2.grid(True, alpha=0.3)
ax12_2.legend(loc='best')

plt.tight_layout()
plt.show()
plt.close()

# Угол тангажа, курса, атаки, курса и крена(в связанной и скоростной СК) 
fig_debug, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 10))
ax1.plot(Result_R['t'], np.rad2deg(Result_R['theta_r']), '#9B1215', label='$\\vartheta_r$')
ax1.set_xlabel('t, с')
ax1.set_ylabel('Угол в, °')
ax1.set_title('Угол тангажа $\\vartheta_r$ (в связанной СК)')
ax1.grid(True)
ax1.legend()

ax2.plot(Result_R['t'], np.rad2deg(Result_R['psi_r']), 'b', label='$\psi_r$')
ax2.set_xlabel('t, с')
ax2.set_ylabel('Угол в, °')
ax2.set_title('Угол рысканья ракеты $\psi_r$ (в связанной СК)')
ax2.grid(True)
ax2.legend()

ax3.plot(Result_R['t'], np.rad2deg(Result_R['gamma_r']), 'g', label='$\gamma_r$')
ax3.set_xlabel('t, с')
ax3.set_ylabel('Угол в, °')
ax3.set_title('Угол крена ракеты $\gamma_r$ (в связанной СК)')
ax3.grid(True)
ax3.legend()

ax4.plot(Result_R['t'], np.rad2deg(Result_R['alpha']), '#9B1215', label='α')
ax4.set_xlabel('t, с')
ax4.set_ylabel('Угол в, °')
ax4.set_title('Угол атаки α (в скоростной СК)')
ax4.grid(True)
ax4.legend()

ax5.plot(Result_R['t'], np.rad2deg(Result_R['beta']), 'b', label='$\\beta$')
ax5.set_xlabel('t, с')
ax5.set_ylabel('Угол в, °')
ax5.set_title('Угол скольжения $\\beta$ (в скоростной СК)')
ax5.grid(True)
ax5.legend()

ax6.plot(Result_R['t'], np.rad2deg(Result_R['gamma_a']), 'g', label='$\gamma_a$')
ax6.set_xlabel('t, с')
ax6.set_ylabel('Угол в, °')
ax6.set_title('Угол крена $\gamma_a$ (в скоростной СК)')
ax6.grid(True)
ax6.legend()

plt.tight_layout()
plt.show()
plt.close()


# Угловые скорости от времени
fig_omegi, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(Result_R['t'],Result_R['omega_x'] * 60 / (2 * np.pi), color='#9B1215', label='$\omega_x$')
ax1.set_xlabel('t, с')
ax1.set_ylabel('$\omega_x$, об/мин')
ax1.set_title('Угловая скорость крена')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(Result_R['t'],Result_R['omega_y'] * 60 / (2 * np.pi), color='b', label='$\omega_y$')
ax2.set_xlabel('t, с')
ax2.set_ylabel('$\omega_y$, об/мин')
ax2.set_title('Угловая скорость курса')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3.plot(Result_R['t'],Result_R['omega_z'] * 60 / (2 * np.pi), color='g', label='$\omega_z$')
ax3.set_xlabel('t, с')
ax3.set_ylabel('$\omega_z$, об/мин')
ax3.set_title('Угловая скорость тангажа')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()
plt.close()


# Аэродинамические СИЛЫ
fig13, ((ax13_1, ax13_2, ax13_3), (ax13_4, ax13_5, ax13_6)) = plt.subplots(2, 3, figsize=(16, 10))
# Сила сопротивления
ax13_1.plot(Result_R['t'], Result_R['X_a'], color='#9B1315', label='$X_{a}$')
ax13_1.set_xlabel('t, с')
ax13_1.set_ylabel('$X_{a}$')
ax13_1.set_title('Сила сопротивления')
ax13_1.grid(True, alpha=0.3)
ax13_1.legend()




# Подъёмная сила
ax13_2.plot(Result_R['t'], Result_R['Y_a'], color='#9B1315', label='$Y_{a}$')
ax13_2.set_xlabel('t, с')
ax13_2.set_ylabel('$Y_{a}$')
ax13_2.set_title('Подъёмная сила')
ax13_2.grid(True, alpha=0.3)
ax13_2.legend()

ax13_3.plot(Result_R['t'], Result_R['Z_a'], color='#9B1315', label='$Z_{a}$')
ax13_3.set_xlabel('t, с')
ax13_3.set_ylabel('$Z_{a}$')
ax13_3.set_title('Боковая сила')
ax13_3.grid(True, alpha=0.3)
ax13_3.legend()

ax13_4.plot(Result_R['t'], Result_R['X'], color='#9B1315', label='$X$')
ax13_4.set_xlabel('t, с')
ax13_4.set_ylabel('$X$')
ax13_4.set_title('Продольная сила')
ax13_4.grid(True, alpha=0.3)
ax13_4.legend()

# Подъёмная сила
ax13_5.plot(Result_R['t'], Result_R['Y'], color='#9B1315', label='$Y$')
ax13_5.set_xlabel('t, с')
ax13_5.set_ylabel('$Y$')
ax13_5.set_title('Нормальная сила')
ax13_5.grid(True, alpha=0.3)
ax13_5.legend()

ax13_6.plot(Result_R['t'], Result_R['Z'], color='#9B1315', label='$Z$')
ax13_6.set_xlabel('t, с')
ax13_6.set_ylabel('$Z$')
ax13_6.set_title('Поперечная сила')
ax13_6.grid(True, alpha=0.3)
ax13_6.legend()

plt.tight_layout()
plt.show()
plt.close()


fig100, ax100_1 = plt.subplots( figsize=(16, 10))
# кожэффициент силы лобового сопротивления
ax100_1.plot(Result_R['t'], Result_R['c_xa'], color='red', label='$c_{xa}$')
ax100_1.set_xlabel('t, с')
ax100_1.set_ylabel('$c_{xa}$')
ax100_1.set_title('коэффициент силы лобового сопротивления')
ax100_1.grid(True)
ax100_1.legend()
plt.tight_layout()
plt.show()
plt.close()


# # 14. Моменты крена, курса, тангажа
# fig14, (ax14_1, ax14_2, ax14_3) = plt.subplots(1, 3, figsize=(15, 5))

# ax14_1.plot(Result_R['t'], Result_R['M_x'], color='#9B1314', label='$M_{x}$')
# ax14_1.set_xlabel('t, с')
# ax14_1.set_ylabel('$M_{x}$')
# ax14_1.set_title('Момент крена')
# ax14_1.grid(True, alpha=0.3)
# ax14_1.legend()

# ax14_2.plot(Result_R['t'], Result_R['M_y'], color='#9B1314', label='$M_{y}$')
# ax14_2.set_xlabel('t, с')
# ax14_2.set_ylabel('$M_{y}$')
# ax14_2.set_title('Момент рысканья')
# ax14_2.grid(True, alpha=0.3)
# ax14_2.legend()

# ax14_3.plot(Result_R['t'], Result_R['M_z'], color='#9B1314', label='$M_{z}$')
# ax14_3.set_xlabel('t, с')
# ax14_3.set_ylabel('$M_{y}$')
# ax14_3.set_title('Момент тангажа')
# ax14_3.grid(True, alpha=0.3)
# ax14_3.legend()

# plt.tight_layout()
# plt.show()
# plt.close()

# figus, (ax1) = plt.subplots(figsize=(12, 5))
# ax1.plot(Result_R['t'], np.rad2deg(Result_R['Theta_r']), color='#9B1215', label='$\Theta_р$')
# ax1.plot(Result_R['t'], np.rad2deg(Result_R['theta_r']), color='#9B1215', linestyle='--', label='$\\theta_р$')
# ax1.set_xlabel('t, с')
# ax1.set_ylabel('$\Theta$, °')
# ax1.set_title('Углы наклона траектории и тангажа')
# ax1.grid(True, alpha=0.3)
# ax1.legend()

# 15. Моменты тангажа
fig15, (ax15_1, ax15_2, ax15_3) = plt.subplots(1, 3, figsize=(15, 5))

ax15_1.plot(Result_R['t'], Result_R['M_z_st'], color='#9B1314', label='$M_{z_st}$')
ax15_1.set_xlabel('t, с')
ax15_1.set_ylabel('$M_{z_st}$')
ax15_1.set_title('Момент тангажа (статический)')
ax15_1.grid(True, alpha=0.3)
ax15_1.legend()

ax15_2.plot(Result_R['t'], Result_R['M_z_omega'], color='#9B1314', label='$M_{z_omega}$')
ax15_2.set_xlabel('t, с')
ax15_2.set_ylabel('$M_{z_omega}$')
ax15_2.set_title('Момент тангажа (демпфирования)')
ax15_2.grid(True, alpha=0.3)
ax15_2.legend()

# ax15_3.plot(Result_R['t'], Result_R['M_z'], color='#9B1314', label='$M_{z}$')
ax15_3.plot(Result_R['t'], Result_R['M_z_omega'] + Result_R['M_z_st'], color='#9B1314', label='$M_{z}$')
ax15_3.set_xlabel('t, с')
ax15_3.set_ylabel('$M_{z}$')
ax15_3.set_title('Момент тангажа (суммарный)')
ax15_3.grid(True, alpha=0.3)
ax15_3.legend()

plt.tight_layout()
plt.show()
plt.close()

# ============================================================
# 16. ЭЛЛИПС РАССЕИВАНИЯ ПО РЕАЛЬНЫМ ПРОГОНАМ
# ============================================================
def read_dispersion_from_file(filename="sensitivity_results.txt"):
    """Читает sigma_x, sigma_z, CEP из файла результатов"""
    sigma_x = None
    sigma_z = None
    cep = None
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Ищем строки с СКО
            import re
            match_x = re.search(r'СКО по дальности \(σ_x\):\s*([\d.]+)', content)
            match_z = re.search(r'СКО по боку \(σ_z\):\s*([\d.]+)', content)
            match_cep = re.search(r'КВО \(CEP\):\s*([\d.]+)', content)
            
            if match_x:
                sigma_x = float(match_x.group(1))
            if match_z:
                sigma_z = float(match_z.group(1))
            if match_cep:
                cep = float(match_cep.group(1))
                
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
    
    return sigma_x, sigma_z, cep


# ============================================================
# ПОСТРОЕНИЕ ЭЛЛИПСА ПО ДАННЫМ ИЗ ФАЙЛА
# ============================================================

# Читаем данные из файла
sigma_x, sigma_z, cep = read_dispersion_from_file("sensitivity_results.txt")

if sigma_x is not None and sigma_z is not None and cep is not None:
    try:
        # Загружаем точки из файла
        hit_points = np.loadtxt("hit_points.txt")
        
        if len(hit_points) == 0:
            print("Файл hit_points.txt пуст")
        else:
            if len(hit_points.shape) == 1:
                hit_points = hit_points.reshape(1, -1)
            
            x_points = hit_points[:, 0]
            z_points = hit_points[:, 1]
            
            x_center = np.mean(x_points)
            z_center = np.mean(z_points)
            
            print("\n" + "="*80)
            print("ПОСТРОЕНИЕ ЭЛЛИПСА ПО ДАННЫМ ИЗ ФАЙЛА")
            print("="*80)
            print(f"Центр (X, Z): ({x_center:.1f}, {z_center:.1f}) м")
            print(f"σ_x = {sigma_x:.1f} м, σ_z = {sigma_z:.1f} м")
            print(f"КВО (CEP) = {cep:.1f} м")
            print(f"Количество точек: {len(x_points)}")
            
            # Рисуем эллипс
            fig_ellipse, ax_ellipse = plt.subplots(figsize=(12, 10))
            
            # Точки попаданий
            ax_ellipse.scatter(x_points, z_points, c='lightgray', s=40, alpha=0.7,
                              edgecolors='gray', linewidth=0.5,
                              label=f'Попадания (n={len(x_points)})')
            
            # Эллипсы
            theta = np.linspace(0, 2*np.pi, 100)
            
            # 1σ
            x_ellipse = sigma_x * np.cos(theta) + x_center
            z_ellipse = sigma_z * np.sin(theta) + z_center
            ax_ellipse.plot(x_ellipse, z_ellipse, 'r-', linewidth=2.5,
                           label=f'1σ эллипс (σ_x={sigma_x:.1f}, σ_z={sigma_z:.1f})')
            
            # 2σ
            x_ellipse_2 = 2 * sigma_x * np.cos(theta) + x_center
            z_ellipse_2 = 2 * sigma_z * np.sin(theta) + z_center
            ax_ellipse.plot(x_ellipse_2, z_ellipse_2, 'r--', linewidth=1.5, alpha=0.7,
                           label='2σ (95.4%)')
            
            # 3σ
            x_ellipse_3 = 3 * sigma_x * np.cos(theta) + x_center
            z_ellipse_3 = 3 * sigma_z * np.sin(theta) + z_center
            ax_ellipse.plot(x_ellipse_3, z_ellipse_3, 'r:', linewidth=1, alpha=0.7,
                           label='3σ (99.7%)')
            
            # КВО
            cep_circle = plt.Circle((x_center, z_center), cep,
                                    fill=False, color='blue', linestyle='--',
                                    linewidth=2, label=f'КВО = {cep:.1f} м (50%)')
            ax_ellipse.add_patch(cep_circle)
            
            # Центр
            ax_ellipse.plot(x_center, z_center, 'go', markersize=12, label='Центр рассеивания')
            
            # Оси
            ax_ellipse.axhline(y=z_center, color='gray', linestyle=':', alpha=0.5)
            ax_ellipse.axvline(x=x_center, color='gray', linestyle=':', alpha=0.5)
            
            # Подписи
            ax_ellipse.annotate(f'σ_x = {sigma_x:.1f} м',
                               xy=(x_center + sigma_x, z_center),
                               xytext=(x_center + sigma_x + 50, z_center + 20),
                               fontsize=10, ha='center', color='red',
                               arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
            
            ax_ellipse.annotate(f'σ_z = {sigma_z:.1f} м',
                               xy=(x_center, z_center + sigma_z),
                               xytext=(x_center + 50, z_center + sigma_z + 20),
                               fontsize=10, va='center', color='red',
                               arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
            
            # Настройки
            ax_ellipse.set_xlabel('Дальность X, м')
            ax_ellipse.set_ylabel('Боковое отклонение Z, м')
            ax_ellipse.set_title('Эллипс рассеивания ракеты')
            ax_ellipse.grid(True, alpha=0.3)
            ax_ellipse.axis('equal')
            ax_ellipse.legend(loc='best')
            plt.tight_layout()
            plt.show()
             
    except FileNotFoundError:
        print("Файл hit_points.txt не найден")
    except Exception as e:
        print(f"Ошибка при построении эллипса: {e}")
else:
    print("\n⚠ ВНИМАНИЕ: Не удалось прочитать данные из файла sensitivity_results.txt")
    print("   Убедитесь, что файл существует и содержит строки:")
    print("   'СКО по дальности (σ_x): X.X м'")
    print("   'СКО по боку (σ_z): X.X м'")
    print("   'КВО (CEP): X.X м'")