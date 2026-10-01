# diagnostics.py
# -*- coding: utf-8 -*-
"""
Скрипт для диагностики различий между МТ и ТТ моделями
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_data(filename):
    """Загрузка данных из .gra файла"""
    try:
        return pd.read_csv(filename, delimiter="\t")
    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
        return None

def compare_trajectories():
    """Сравнение траекторий МТ и ТТ"""
    
    # Проверяем оба файла
    mt_file = 'Result_R_MT.gra'
    tt_file = 'Result_R_TT.gra'
    
    # Если нет файлов, пытаемся найти альтернативные имена
    if not os.path.exists(mt_file):
        # Ищем любой файл, который может быть МТ
        possible_mt = ['Result_R_MT.gra', 'Result_R_bal.gra', 'Result_R_ballistic.gra']
        for f in possible_mt:
            if os.path.exists(f):
                mt_file = f
                break
    
    if not os.path.exists(tt_file):
        possible_tt = ['Result_R_TT.gra', 'Result_R_full.gra', 'Result_R.gra']
        for f in possible_tt:
            if os.path.exists(f):
                tt_file = f
                break
    
    # Загружаем данные
    try:
        mt_data = pd.read_csv(mt_file, delimiter="\t")
        print(f"Загружен МТ файл: {mt_file}")
    except:
        mt_data = None
        print(f"Файл {mt_file} не найден!")
    
    try:
        tt_data = pd.read_csv(tt_file, delimiter="\t")
        print(f"Загружен ТТ файл: {tt_file}")
    except:
        tt_data = None
        print(f"Файл {tt_file} не найден!")
    
    if mt_data is None or tt_data is None:
        print("\nОШИБКА: Не удалось загрузить данные!")
        print("Убедитесь, что файлы существуют.")
        print("Сначала запустите run_both_modes.py")
        return None, None
    
    print("="*80)
    print("ДИАГНОСТИКА РАЗЛИЧИЙ МЕЖДУ МТ И ТТ МОДЕЛЯМИ")
    print("="*80)
    
    # 1. Сравнение дальности и высоты
    print("\n1. СРАВНЕНИЕ ДАЛЬНОСТИ И ВЫСОТЫ")
    print("-"*60)
    
    x_mt_final = mt_data['x_r'].iloc[-1]
    y_mt_final = mt_data['y_r'].iloc[-1]
    x_tt_final = tt_data['x_r'].iloc[-1]
    y_tt_final = tt_data['y_r'].iloc[-1]
    
    print(f"МТ модель:   X = {x_mt_final:.1f} м,  Y = {y_mt_final:.1f} м")
    print(f"ТТ модель:   X = {x_tt_final:.1f} м,  Y = {y_tt_final:.1f} м")
    print(f"Разница:     ΔX = {x_tt_final - x_mt_final:.1f} м,  ΔY = {y_tt_final - y_mt_final:.1f} м")
    
    # Создаём фигуру с траекториями
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Траектории OXY
    ax1 = axes[0, 0]
    ax1.plot(mt_data['x_r'], mt_data['y_r'], 'b-', label='МТ', linewidth=2)
    ax1.plot(tt_data['x_r'], tt_data['y_r'], 'r-', label='ТТ', linewidth=2)
    ax1.scatter([mt_data['x_r'].iloc[-1]], [mt_data['y_r'].iloc[-1]], color='b', s=100, zorder=5)
    ax1.scatter([tt_data['x_r'].iloc[-1]], [tt_data['y_r'].iloc[-1]], color='r', s=100, zorder=5)
    ax1.set_xlabel('X, м')
    ax1.set_ylabel('Y, м')
    ax1.set_title('Траектории OXY')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # 2. Высота от времени
    ax2 = axes[0, 1]
    ax2.plot(mt_data['t'], mt_data['y_r'], 'b-', label='МТ')
    ax2.plot(tt_data['t'], tt_data['y_r'], 'r-', label='ТТ')
    ax2.set_xlabel('t, с')
    ax2.set_ylabel('Y, м')
    ax2.set_title('Высота от времени')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Дальность от времени
    ax3 = axes[0, 2]
    ax3.plot(mt_data['t'], mt_data['x_r'], 'b-', label='МТ')
    ax3.plot(tt_data['t'], tt_data['x_r'], 'r-', label='ТТ')
    ax3.set_xlabel('t, с')
    ax3.set_ylabel('X, м')
    ax3.set_title('Дальность от времени')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Скорость
    ax4 = axes[1, 0]
    ax4.plot(mt_data['t'], mt_data['v_r'], 'b-', label='МТ')
    ax4.plot(tt_data['t'], tt_data['v_r'], 'r-', label='ТТ')
    ax4.set_xlabel('t, с')
    ax4.set_ylabel('v, м/с')
    ax4.set_title('Скорость от времени')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Число Маха
    ax5 = axes[1, 1]
    ax5.plot(mt_data['t'], mt_data['M'], 'b-', label='МТ')
    ax5.plot(tt_data['t'], tt_data['M'], 'r-', label='ТТ')
    ax5.set_xlabel('t, с')
    ax5.set_ylabel('M')
    ax5.set_title('Число Маха')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Угол наклона траектории
    ax6 = axes[1, 2]
    ax6.plot(mt_data['t'], np.rad2deg(mt_data['Theta_r']), 'b-', label='МТ')
    ax6.plot(tt_data['t'], np.rad2deg(tt_data['Theta_r']), 'r-', label='ТТ')
    if 'theta_r' in tt_data.columns:
        ax6.plot(tt_data['t'], np.rad2deg(tt_data['theta_r']), 'r--', label='ТТ (тангаж)', alpha=0.7)
    ax6.set_xlabel('t, с')
    ax6.set_ylabel('Угол, °')
    ax6.set_title('Угол наклона траектории')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('diagnostics_trajectories.png', dpi=150)
    plt.show()
    
    return mt_data, tt_data

def analyze_alpha_and_force(tt_data):
    """Анализ угла атаки и подъёмной силы в модели ТТ"""
    
    print("\n2. АНАЛИЗ УГЛА АТАКИ И ПОДЪЁМНОЙ СИЛЫ")
    print("-"*60)
    
    # Статистика угла атаки
    alpha_deg = np.rad2deg(tt_data['alpha'])
    print(f"Угол атаки (α):")
    print(f"  Среднее:     {np.mean(alpha_deg):.4f}°")
    print(f"  Макс |α|:    {np.max(np.abs(alpha_deg)):.4f}°")
    print(f"  Стандарт:    {np.std(alpha_deg):.4f}°")
    print(f"  На конце:    {alpha_deg.iloc[-1]:.4f}°")
    print(f"  Медиана:     {np.median(alpha_deg):.4f}°")
    
    # Статистика подъёмной силы
    if 'Y_a' in tt_data.columns:
        print(f"\nПодъёмная сила (Y_a):")
        print(f"  Среднее:     {np.mean(tt_data['Y_a']):.1f} Н")
        print(f"  Макс |Y_a|:  {np.max(np.abs(tt_data['Y_a'])):.1f} Н")
        print(f"  Стандарт:    {np.std(tt_data['Y_a']):.1f} Н")
        print(f"  На конце:    {tt_data['Y_a'].iloc[-1]:.1f} Н")
    
    # Создаём фигуру
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Угол атаки
    ax1 = axes[0, 0]
    ax1.plot(tt_data['t'], alpha_deg, 'b-', linewidth=1.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('t, с')
    ax1.set_ylabel('α, °')
    ax1.set_title('Угол атаки')
    ax1.grid(True, alpha=0.3)
    
    # 2. Подъёмная сила
    ax2 = axes[0, 1]
    if 'Y_a' in tt_data.columns:
        ax2.plot(tt_data['t'], tt_data['Y_a'], 'r-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('t, с')
        ax2.set_ylabel('Y_a, Н')
        ax2.set_title('Подъёмная сила')
    else:
        ax2.text(0.5, 0.5, 'Нет данных Y_a', ha='center', va='center')
    ax2.grid(True, alpha=0.3)
    
    # 3. Нормальная сила (связанная СК)
    ax3 = axes[0, 2]
    if 'Y' in tt_data.columns:
        ax3.plot(tt_data['t'], tt_data['Y'], 'g-', linewidth=1.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('t, с')
        ax3.set_ylabel('Y, Н')
        ax3.set_title('Нормальная сила (связанная СК)')
    else:
        ax3.text(0.5, 0.5, 'Нет данных Y', ha='center', va='center')
    ax3.grid(True, alpha=0.3)
    
    # 4. Момент тангажа
    ax4 = axes[1, 0]
    if 'M_z' in tt_data.columns:
        ax4.plot(tt_data['t'], tt_data['M_z'], 'm-', linewidth=1.5)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('t, с')
        ax4.set_ylabel('M_z, Н·м')
        ax4.set_title('Момент тангажа')
    else:
        ax4.text(0.5, 0.5, 'Нет данных M_z', ha='center', va='center')
    ax4.grid(True, alpha=0.3)
    
    # 5. Угол атаки (увеличенный масштаб для колебаний)
    ax5 = axes[1, 1]
    # Возьмём первую секунду для детального просмотра
    t_limit = min(1.0, tt_data['t'].iloc[-1])
    mask = tt_data['t'] <= t_limit
    ax5.plot(tt_data['t'][mask], alpha_deg[mask], 'b-', linewidth=1.5)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('t, с')
    ax5.set_ylabel('α, °')
    ax5.set_title(f'Угол атаки (первые {t_limit:.1f} с)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Зависимость Y_a от α
    ax6 = axes[1, 2]
    if 'Y_a' in tt_data.columns:
        # Строим только для малых углов (чтобы увидеть линейность)
        mask_small = np.abs(alpha_deg) < 2.0
        ax6.scatter(alpha_deg[mask_small], tt_data['Y_a'][mask_small], 
                   c=tt_data['t'][mask_small], cmap='viridis', s=10, alpha=0.7)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax6.set_xlabel('α, °')
        ax6.set_ylabel('Y_a, Н')
        ax6.set_title('Зависимость Y_a от α')
        cbar = plt.colorbar(ax6.collections[0], ax=ax6)
        cbar.set_label('t, с')
    else:
        ax6.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagnostics_alpha_force.png', dpi=150)
    plt.show()
    
    return alpha_deg

def main():
    """Главная функция диагностики"""
    print("\n" + "="*80)
    print("ЗАПУСК ДИАГНОСТИКИ")
    print("="*80)
    
    # 1. Сравнение траекторий
    mt_data, tt_data = compare_trajectories()
    
    if tt_data is not None:
        # 2. Анализ угла атаки и подъёмной силы
        alpha_deg = analyze_alpha_and_force(tt_data)
    
    print("\n" + "="*80)
    print("ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("Сохранено:")
    print("  - diagnostics_trajectories.png")
    print("  - diagnostics_alpha_force.png")
    print("="*80)

if __name__ == "__main__":
    main()