# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 22:54:12 2026

@author: Рассказин Алексей
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================================
# ШАГ 1: Исходные данные
# ============================================================================

# Геометрия
d = 0.24                      # диаметр, м
L = 1.4                       # характерная длина, м (стр. 4)
V = 400.0                     # скорость, м/с
rho = 1.225                   # плотность воздуха, кг/м³

# Массовые характеристики
m = 50.0                      # масса, кг (оценка, т.к. не задана явно)
Jz = 120.0                    # момент инерции, кг·м² (стр. 5)
g = 9.807                     # ускорение свободного падения, м/с²

# Аэродинамические коэффициенты
CY = 0.072                    # коэффициент боковой силы (стр. 4)
m_omega = -0.37               # производная момента по угловой скорости (стр. 1)

# Нагрузки (для определения плеча)
Y = 320.0                     # боковая сила, Н (стр. 1)
Mz = 210.0                    # момент, Н·м (стр. 1)

# Геометрия для аэродинамических производных
ln = 0.4                      # длина носка, м (стр. 1)
lc = 0.6                      # длина хвоста, м (стр. 4)
xt = 0.539                    # координата центра масс, м (стр. 4)

# Углы (для упрощения принимаем нулевыми)
gamma = 0.0                   # угол крена, рад
beta = 0.0                    # угол скольжения, рад
Psi = 0.0                     # угол курса, рад
psi = 0.0                     # угол рыскания, рад

# Параметры интегрирования
tk = 170.0                      # время моделирования, с
dt = 0.001                    # шаг интегрирования, с
n_points = int(tk / dt) + 1   # количество точек

# ============================================================================
# ШАГ 2: Производные параметры
# ============================================================================

# Площадь миделя
S = np.pi * d**2 / 4.0        # м²

# Скоростной напор
q = rho * V**2 / 2.0          # Па

# Плечо момента (центр давления)
x_shoulder = Mz / Y           # м

# ============================================================================
# ШАГ 3: Геометрические параметры для аэродинамики
# ============================================================================

lambda_n = ln / d             # удлинение носка
lambda_f = (ln + lc) / d      # удлинение "нос+хвост"

print(f"λn = {lambda_n:.3f}")
print(f"λf = {lambda_f:.3f}")

# ============================================================================
# ШАГ 4: Аэродинамические производные
# ============================================================================

# Параметр положения фокуса (xL)
lambda_f2 = lambda_f**(-2)
lambda_n2 = lambda_n**(-2)

xL_num = 2 * lambda_f2 - lambda_n2
xL_den = 4 * lambda_f2 * (lambda_f2 - (2/3) * lambda_n)
xL = xL_num / xL_den

print(f"xL = {xL:.3f}")

# Производная момента по углу атаки (m_ZWZ)
xf_L = 0.9                    # относительное положение фокуса (из стр. 1)
m_ZWZ = -2 * (1 - xf_L + xf_L**2 - xL)

print(f"m_ZWZ = {m_ZWZ:.3f}")

# Демпфирующий момент (m_zωz)
xt_L = xt / L
m_zωz = -2 * (1 - xt_L + xt_L**2 - xL)

print(f"m_zωz = {m_zωz:.3f}")

# ============================================================================
# ШАГ 5: Функции для расчёта сил и моментов
# ============================================================================

def angle_of_attack(theta, Theta):
    """
    Расчёт угла атаки.
    theta - угол тангажа, рад
    Theta - угол наклона траектории, рад
    При γ=β=Ψ=ψ=0: α = θ - Θ
    """
    # Упрощённая формула при нулевых углах
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_Theta = np.sin(Theta)
    cos_Theta = np.cos(Theta)
    
    numerator = (sin_theta * np.cos(gamma) * np.cos(Psi - psi) + 
                 np.sin(gamma) * np.sin(Psi - psi)) * cos_Theta - \
                 sin_Theta * cos_theta * np.cos(gamma)
    denominator = np.cos(beta)
    
    alpha = np.arcsin(numerator / denominator)
    return alpha

def lateral_force(alpha):
    """
    Боковая сила Y1(α) и её проекция Ya(α).
    alpha - угол атаки, рад
    """
    # Y1(α) = q * S * CY * α (в радианах)
    Y1 = q * S * CY * alpha
    
    # Проекция на вертикаль
    Ya = Y1 * np.cos(alpha)
    
    return Y1, Ya

def aerodynamic_moment(theta, Theta):
    """
    Аэродинамический момент Mad(θ, Θ).
    """
    alpha = angle_of_attack(theta, Theta)
    Y1, _ = lateral_force(alpha)
    Mad = -Y1 * x_shoulder
    return Mad

def damping_moment(omega):
    """
    Демпфирующий момент Mω(ω) от вращения.
    omega - угловая скорость тангажа, рад/с
    """
    # Mω = m_omega * ω * (L²/V) * q * S
    M_omega = m_omega * omega * (L**2 / V) * q * S
    return M_omega

# ============================================================================
# ШАГ 6: Система дифференциальных уравнений
# ============================================================================

def equations(t, X):
    """
    Система ОДУ для продольного движения.
    X = [Theta, theta, omega]
    Theta - угол наклона траектории, рад
    theta - угол тангажа, рад
    omega - угловая скорость тангажа, рад/с
    """
    Theta, theta, omega = X
    
    # Угол атаки
    alpha = angle_of_attack(theta, Theta)
    
    # Силы и моменты
    Y1, Ya = lateral_force(alpha)
    Mad = aerodynamic_moment(theta, Theta)
    M_omega = damping_moment(omega)
    
    # Уравнения движения
    dTheta_dt = (Ya - m * g * np.cos(Theta)) / (m * V)
    dtheta_dt = omega
    domega_dt = (Mad + M_omega) / Jz
    
    return [dTheta_dt, dtheta_dt, domega_dt]

# ============================================================================
# ШАГ 7: Начальные условия
# ============================================================================

# Начальные условия: [Theta0, theta0, omega0]
X0 = [0.0, 0.0, 0.0]         # все углы и скорость нулевые

# ============================================================================
# ШАГ 8: Численное интегрирование
# ============================================================================

print("\nНачало интегрирования...")

# Создание временной сетки
t_span = (0, tk)
t_eval = np.linspace(0, tk, n_points)

# Решение системы ОДУ
solution = solve_ivp(
    equations,
    t_span,
    X0,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9
)

print(f"Интегрирование завершено. Успешно: {solution.success}")
print(f"Количество шагов: {len(solution.t)}")

# ============================================================================
# ШАГ 9: Извлечение результатов
# ============================================================================

t = solution.t
Theta = solution.y[0]         # угол наклона траектории, рад
theta = solution.y[1]         # угол тангажа, рад
omega = solution.y[2]         # угловая скорость, рад/с

# Расчёт угла атаки для каждого момента времени
alpha = np.zeros_like(t)
Y1_arr = np.zeros_like(t)
Ya_arr = np.zeros_like(t)
Mad_arr = np.zeros_like(t)
M_omega_arr = np.zeros_like(t)

for i in range(len(t)):
    alpha[i] = angle_of_attack(theta[i], Theta[i])
    Y1_arr[i], Ya_arr[i] = lateral_force(alpha[i])
    Mad_arr[i] = aerodynamic_moment(theta[i], Theta[i])
    M_omega_arr[i] = damping_moment(omega[i])

# Перевод в градусы для удобства визуализации
Theta_deg = np.rad2deg(Theta)
theta_deg = np.rad2deg(theta)
alpha_deg = np.rad2deg(alpha)

# ============================================================================
# ШАГ 10: Вывод численных значений
# ============================================================================

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ РАСЧЁТА")
print("="*60)

print(f"\nПараметры в конце моделирования (t = {tk:.1f} с):")
print(f"  Угол наклона траектории Θ: {Theta_deg[-1]:.4f}°")
print(f"  Угол тангажа θ: {theta_deg[-1]:.4f}°")
print(f"  Угол атаки α: {alpha_deg[-1]:.4f}°")
print(f"  Угловая скорость ω: {omega[-1]:.6f} рад/с")

print(f"\nСтатистика по углу атаки:")
print(f"  Максимальный: {np.max(alpha_deg):.4f}°")
print(f"  Минимальный: {np.min(alpha_deg):.4f}°")
print(f"  Среднеквадратичное: {np.std(alpha_deg):.4f}°")

# ============================================================================
# ШАГ 11: Построение графиков
# ============================================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# График 1: Углы во времени
ax = axes[0, 0]
ax.plot(t, theta_deg, 'b-', label='θ (тангаж)')
ax.plot(t, Theta_deg, 'r--', label='Θ (траектория)')
ax.plot(t, alpha_deg, 'g-.', label='α (атака)')
ax.set_xlabel('Время, с')
ax.set_ylabel('Угол, °')
ax.set_title('Углы во времени')
ax.grid(True, alpha=0.3)
ax.legend()

# График 2: Угловая скорость
ax = axes[0, 1]
ax.plot(t, omega, 'b-')
ax.set_xlabel('Время, с')
ax.set_ylabel('Угловая скорость, рад/с')
ax.set_title('Угловая скорость тангажа')
ax.grid(True, alpha=0.3)

# График 3: Аэродинамический момент
ax = axes[1, 0]
ax.plot(t, Mad_arr, 'r-')
ax.set_xlabel('Время, с')
ax.set_ylabel('Момент, Н·м')
ax.set_title('Аэродинамический момент M_ad')
ax.grid(True, alpha=0.3)

# График 4: Демпфирующий момент
ax = axes[1, 1]
ax.plot(t, M_omega_arr, 'g-')
ax.set_xlabel('Время, с')
ax.set_ylabel('Момент, Н·м')
ax.set_title('Демпфирующий момент M_ω')
ax.grid(True, alpha=0.3)

# График 5: Боковая сила
ax = axes[2, 0]
ax.plot(t, Y1_arr, 'b-', label='Y₁')
ax.plot(t, Ya_arr, 'r--', label='Y_a')
ax.set_xlabel('Время, с')
ax.set_ylabel('Сила, Н')
ax.set_title('Боковая сила')
ax.grid(True, alpha=0.3)
ax.legend()

# График 6: Фазовый портрет (θ - ω)
ax = axes[2, 1]
ax.plot(theta, omega, 'b-', linewidth=1)
ax.set_xlabel('Угол тангажа θ, рад')
ax.set_ylabel('Угловая скорость ω, рад/с')
ax.set_title('Фазовый портрет')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# ДОПОЛНИТЕЛЬНО: Проверка устойчивости (линеаризованная модель)
# ============================================================================

print("\n" + "="*60)
print("АНАЛИЗ УСТОЙЧИВОСТИ (линеаризованная модель)")
print("="*60)

# Линеаризованные коэффициенты
# m_zα = -CY * x_shoulder / L (производная момента по α)
m_zα = -CY * x_shoulder / L

# Характеристическое уравнение для короткопериодического движения
# s² + ( -M_ω/ω - Ya_α/(m V) ) s + ( -M_α/Jz - Ya_α * M_ω/(Jz m V) ) = 0

# Производная силы по α
Ya_α = q * S * CY  # ∂Ya/∂α

# Коэффициент демпфирования
n_omega = -m_omega * (L**2 / V) * q * S / Jz

# Частота
k_alpha = -m_zα * q * S * L / Jz

# Упрощённая линеаризованная система:
# dα/dt = ω - (Ya_α/(m V)) α
# dω/dt = (k_alpha) α + n_omega ω

A = np.array([
    [-Ya_α/(m*V), 1.0],
    [k_alpha, n_omega]
])

eigenvalues = np.linalg.eigvals(A)

print(f"\nМатрица линеаризованной системы:")
print(f"A = [[{A[0,0]:.4f}, {A[0,1]:.1f}],")
print(f"     [{A[1,0]:.4f}, {A[1,1]:.4f}]]")

print(f"\nСобственные значения:")
for i, ev in enumerate(eigenvalues):
    print(f"  λ{i+1} = {ev.real:.4f} + {ev.imag:.4f}i")

if np.all(np.real(eigenvalues) < 0):
    print("\nВывод: Система УСТОЙЧИВА (все Re(λ) < 0)")
else:
    print("\nВывод: Система НЕУСТОЙЧИВА (есть Re(λ) ≥ 0)")

# Период колебаний, если есть комплексные корни
if np.any(np.imag(eigenvalues) != 0):
    for ev in eigenvalues:
        if np.abs(ev.imag) > 1e-6:
            T = 2 * np.pi / np.abs(ev.imag)
            print(f"  Период колебаний: {T:.4f} с")
            break

print("\n" + "="*60)
print("РАСЧЁТ ЗАВЕРШЁН")
print("="*60)

figur, ax = plt.subplots(figsize=(14, 10))

# График 1: Углы во времени
# ax.plot(t, theta_deg, 'b-', label='θ (тангаж)')
# ax.plot(t, Theta_deg, 'r--', label='Θ (траектория)')
ax.plot(t[169900:], alpha_deg[169900:], 'g-.', label='α (атака)')
ax.set_xlabel('Время, с')
ax.set_ylabel('Угол, °')
ax.set_title('Углы во времени')
ax.grid(True, alpha=0.3)
ax.legend()