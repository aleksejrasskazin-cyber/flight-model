import math
from LIBCONSTFUNC import g, sqr

def n_ya_pot_PurePursuit(dot_epsilon_rc, v_r, Theta_r):
    """
    Функция для вычисления потребной перегрузки по методу чистой погони (самонаведение)
    
    Args:
        dot_epsilon_rc: производная угла визирования
        v_r: скорость ракеты
        Theta_r: угол наклона траектории ракеты
    
    Returns:
        Потребная перегрузка
    """
    return v_r / g * dot_epsilon_rc + math.cos(Theta_r)

def n_ya_pot_ParallelNav(epsilon_rc, v_r, v_c, Theta_r, Theta_c, a_xa_r, a_xa_c, a_ya_c):
    """
    Функция для вычисления потребной перегрузки по методу параллельного сближения (самонаведение)
    
    Args:
        epsilon_rc: угол визирования
        v_r: скорость ракеты
        v_c: скорость цели
        Theta_r: угол наклона траектории ракеты
        Theta_c: угол наклона траектории цели
        a_xa_r: продольное ускорение ракеты
        a_xa_c: продольное ускорение цели
        a_ya_c: нормальное ускорение цели
    
    Returns:
        Потребная перегрузка
    """
    numerator = (a_ya_c * v_r * math.cos(epsilon_rc - Theta_c) - 
                (a_xa_c * v_r - a_xa_r * v_c) * math.sin(epsilon_rc - Theta_c))
    denominator = (g * v_r * math.sqrt(1.0 - sqr(v_c / v_r * math.sin(epsilon_rc - Theta_c))))
    return numerator / denominator + math.cos(Theta_r)

def n_ya_pot_Null(dot_epsilon_rc, v_r, k, Theta_r):
    return 0

def n_ya_pot_Proportional_const(dot_epsilon_rc, v_r, k, Theta_r):
    """
    Функция для вычисления потребной перегрузки по методу пропорционального сближения с const коэфф. (самонаведение)
    
    Args:
        dot_epsilon_rc: производная угла визирования
        v_r: скорость ракеты
        k: коэффициент пропорциональности
        Theta_r: угол наклона траектории ракеты
    
    Returns:
        Потребная перегрузка
    """
    return v_r / g * k * dot_epsilon_rc + math.cos(Theta_r)

def n_ya_pot_Proportional_var(dot_epsilon_rc, dot_r_rc, lambda_, Theta_r):
    """
    Функция для вычисления потребной перегрузки по методу пропорционального сближения с var коэфф. (самонаведение)
    
    Args:
        dot_epsilon_rc: производная угла визирования
        dot_r_rc: производная дальности
        lambda_: переменный коэффициент
        Theta_r: угол наклона траектории ракеты
    
    Returns:
        Потребная перегрузка
    """
    return lambda_ / g * abs(dot_r_rc) * dot_epsilon_rc + math.cos(Theta_r)

def n_ya_pot_BeamRiding(epsilon_l, dot_epsilon_l, ddot_epsilon_l, r_nr, dot_r_nr, 
                       Theta_r, Theta_n, a_xa_r, a_xa_n, a_ya_n):
    """
    Функция для вычисления потребной перегрузки по методу совмещения (теленаведение), для общего случая
    
    Args:
        epsilon_l: угол визирования
        dot_epsilon_l: первая производная угла визирования
        ddot_epsilon_l: вторая производная угла визирования
        r_nr: дальность
        dot_r_nr: производная дальности
        Theta_r: угол наклона траектории ракеты
        Theta_n: угол наклона траектории носителя
        a_xa_r: продольное ускорение ракеты
        a_xa_n: продольное ускорение носителя
        a_ya_n: нормальное ускорение носителя
    
    Returns:
        Потребная перегрузка
    """
    term1 = ddot_epsilon_l * r_nr + 2.0 * dot_epsilon_l * dot_r_nr
    term2 = a_xa_n * math.sin(Theta_n - epsilon_l) + a_ya_n * math.cos(epsilon_l - Theta_n)
    term3 = a_xa_r * math.sin(Theta_r - epsilon_l)
    
    numerator = term1 + term2 - term3
    denominator = g * math.cos(epsilon_l - Theta_r)
    
    return numerator / denominator + math.cos(Theta_r)