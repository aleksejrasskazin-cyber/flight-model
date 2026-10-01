# run_both_modes.py
# -*- coding: utf-8 -*-
"""
Скрипт для автоматического запуска обоих режимов (МТ и ТТ)
"""

import os
import shutil
import sys

def run_with_mode(mode_name, ballistic_mode):
    """
    Запускает main.py с заданным режимом
    """
    print(f"\n{'='*80}")
    print(f"ЗАПУСК В РЕЖИМЕ: {mode_name}")
    print(f"BALLISTIC_MODE = {ballistic_mode}")
    print(f"{'='*80}")
    
    # 1. Меняем режим в Model.py
    with open('Model.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ищем строку с BALLISTIC_MODE
    import re
    # Заменяем значение
    if ballistic_mode:
        new_content = re.sub(
            r'self\.BALLISTIC_MODE = (True|False)',
            'self.BALLISTIC_MODE = True',
            content
        )
    else:
        new_content = re.sub(
            r'self\.BALLISTIC_MODE = (True|False)',
            'self.BALLISTIC_MODE = False',
            content
        )
    
    with open('Model.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # 2. Запускаем main.py
    print("Запуск main.py...")
    result = os.system('python main.py')
    
    if result != 0:
        print(f"ОШИБКА: main.py завершился с кодом {result}")
        return False
    
    # 3. Переименовываем результат
    if os.path.exists('Result_R.gra'):
        new_name = f'Result_R_{mode_name}.gra'
        shutil.move('Result_R.gra', new_name)
        print(f"Результат сохранён как: {new_name}")
        return True
    else:
        print("ОШИБКА: Result_R.gra не создан!")
        return False

def main():
    print("="*80)
    print("АВТОМАТИЧЕСКИЙ ЗАПУСК МТ И ТТ МОДЕЛЕЙ")
    print("="*80)
    
    # 1. Запускаем МТ модель (баллистическую)
    success_mt = run_with_mode("MT", True)
    
    if not success_mt:
        print("Ошибка при запуске МТ модели!")
        return
    
    # 2. Запускаем ТТ модель (полную)
    success_tt = run_with_mode("TT", False)
    
    if not success_tt:
        print("Ошибка при запуске ТТ модели!")
        return
    
    print("\n" + "="*80)
    print("ОБА РЕЖИМА ЗАПУЩЕНЫ УСПЕШНО!")
    print("Файлы созданы:")
    print("  - Result_R_MT.gra")
    print("  - Result_R_TT.gra")
    print("="*80)
    
    # 3. Запускаем диагностику
    print("\nЗапуск диагностики...")
    os.system('python diagnostics.py')

if __name__ == "__main__":
    main()
