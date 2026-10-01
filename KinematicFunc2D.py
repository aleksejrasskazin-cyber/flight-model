import math
from typing import Optional

# Импортируем константы и функции из LibConstFunc
from LIBCONSTFUNC import pi, sqr

def r(x_r: float, y_r: float, x_c: float, y_c: float) -> float:
    """
    Функция для вычисления расстояния между двумя точками
    
    Args:
        x_r: координата x первой точки, м
        y_r: координата y первой точки, м
        x_c: координата x второй точки, м
        y_c: координата y второй точки, м
    
    Returns:
        Расстояние, м
    """
    return math.sqrt(sqr(y_c - y_r) + sqr(x_c - x_r))

def dot_r(x_r: float, y_r: float, x_c: float, y_c: float, 
          v_r: float, v_c: float, Theta_r: float, Theta_c: float) -> float:
    """
    Функция для вычисления скорости изменения расстояния между двумя точками
    
    Args:
        x_r: координата x первой точки, м
        y_r: координата y первой точки, м
        x_c: координата x второй точки, м
        y_c: координата y второй точки, м
        v_r: скорость первой точки, м/с
        v_c: скорость второй точки, м/с
        Theta_r: угол направления вектора скорости первой точки, рад
        Theta_c: угол направления вектора скорости второй точки, рад
    
    Returns:
        Скорость изменения расстояния, м/с
    """
    r_rc = r(x_r, y_r, x_c, y_c)
    
    Delta_x = x_c - x_r
    Delta_y = y_c - y_r
    Delta_dot_x = v_c * math.cos(Theta_c) - v_r * math.cos(Theta_r)
    Delta_dot_y = v_c * math.sin(Theta_c) - v_r * math.sin(Theta_r)
    
    if r_rc != 0.0:
        return (Delta_y * Delta_dot_y + Delta_x * Delta_dot_x) / r_rc
    else:
        return math.sqrt(sqr(Delta_dot_x) + sqr(Delta_dot_y))

def ddot_r(x_r: float, y_r: float, x_c: float, y_c: float, 
           v_r: float, v_c: float, Theta_r: float, Theta_c: float,
           a_xa_r: float, a_ya_r: float, a_xa_c: float, a_ya_c: float) -> float:
    """
    Функция для вычисления ускорения изменения расстояния между двумя точками
    
    Args:
        x_r: координата x первой точки, м
        y_r: координата y первой точки, м
        x_c: координата x второй точки, м
        y_c: координата y второй точки, м
        v_r: скорость первой точки, м/с
        v_c: скорость второй точки, м/с
        Theta_r: угол направления вектора скорости первой точки, рад
        Theta_c: угол направления вектора скорости второй точки, рад
        a_xa_r: тангенциальное ускорение первой точки, м/с²
        a_ya_r: нормальное скоростное ускорение первой точки, м/с²
        a_xa_c: тангенциальное ускорение второй точки, м/с²
        a_ya_c: нормальное скоростное ускорение второй точки, м/с²
    
    Returns:
        Ускорение изменения расстояния, м/с²
    """
    r_rc = r(x_r, y_r, x_c, y_c)
    dot_r_rc = dot_r(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c)
    
    Delta_x = x_c - x_r
    Delta_y = y_c - y_r
    
    Delta_dot_x = v_c * math.cos(Theta_c) - v_r * math.cos(Theta_r)
    Delta_dot_y = v_c * math.sin(Theta_c) - v_r * math.sin(Theta_r)
    
    Delta_ddot_x = (a_xa_c * math.cos(Theta_c) - a_xa_r * math.cos(Theta_r) - 
                   a_ya_c * math.sin(Theta_c) + a_ya_r * math.sin(Theta_r))
    Delta_ddot_y = (a_xa_c * math.sin(Theta_c) - a_xa_r * math.sin(Theta_r) + 
                   a_ya_c * math.cos(Theta_c) - a_ya_r * math.cos(Theta_r))
    
    if r_rc != 0.0:
        return (sqr(Delta_dot_x) + Delta_x * Delta_ddot_x + Delta_y * Delta_ddot_y + 
                sqr(Delta_dot_y) - sqr(dot_r_rc)) / r_rc
    else:
        return math.sqrt(sqr(Delta_ddot_x) + sqr(Delta_ddot_y))

def epsilon(x_r: float, y_r: float, x_c: float, y_c: float) -> float:
    """
    Функция для вычисления угла наклона линии визирования между двумя точками
    
    Args:
        x_r: координата x первой точки, м
        y_r: координата y первой точки, м
        x_c: координата x второй точки, м
        y_c: координата y второй точки, м
    
    Returns:
        Угол наклона линии визирования, рад
    """
    delta_x = x_c - x_r
    delta_y = y_c - y_r
    
    if delta_x > 0.0:
        return math.atan(delta_y / delta_x)
    elif delta_x < 0.0 and delta_y >= 0.0:
        return math.atan(delta_y / delta_x) + pi
    elif delta_x < 0.0 and delta_y < 0.0:
        return math.atan(delta_y / delta_x) - pi
    elif delta_x == 0.0 and delta_y >= 0.0:
        return pi / 2.0
    elif delta_x == 0.0 and delta_y < 0.0:
        return -pi / 2.0
    else:
        return 0.0  # Для случая, когда обе точки совпадают

def dot_epsilon(x_r: float, y_r: float, x_c: float, y_c: float, 
                v_r: float, v_c: float, Theta_r: float, Theta_c: float) -> float:
    """
    Функция для вычисления угловой скорости линии визирования между двумя точками
    
    Args:
        x_r: координата x первой точки, м
        y_r: координата y первой точки, м
        x_c: координата x второй точки, м
        y_c: координата y второй точки, м
        v_r: скорость первой точки, м/с
        v_c: скорость второй точки, м/с
        Theta_r: угол направления вектора скорости первой точки, рад
        Theta_c: угол направления вектора скорости второй точки, рад
    
    Returns:
        Угловая скорость линии визирования, рад/с
    """
    Delta_x = x_c - x_r
    Delta_y = y_c - y_r
    
    Delta_dot_x = v_c * math.cos(Theta_c) - v_r * math.cos(Theta_r)
    Delta_dot_y = v_c * math.sin(Theta_c) - v_r * math.sin(Theta_r)
    
    return (Delta_x * Delta_dot_y - Delta_y * Delta_dot_x) / sqr(r(x_r, y_r, x_c, y_c))

def ddot_epsilon(x_r: float, y_r: float, x_c: float, y_c: float, 
                 v_r: float, v_c: float, Theta_r: float, Theta_c: float,
                 a_xa_r: float, a_ya_r: float, a_xa_c: float, a_ya_c: float) -> float:
    """
    Функция для вычисления углового ускорения линии визирования между двумя точками
    
    Args:
        x_r: координата x первой точки, м
        y_r: координата y первой точки, м
        x_c: координата x второй точки, м
        y_c: координата y второй точки, м
        v_r: скорость первой точки, м/с
        v_c: скорость второй точки, м/с
        Theta_r: угол направления вектора скорости первой точки, рад
        Theta_c: угол направления вектора скорости второй точки, рад
        a_xa_r: тангенциальное ускорение первой точки, м/с²
        a_ya_r: нормальное скоростное ускорение первой точки, м/с²
        a_xa_c: тангенциальное ускорение второй точки, м/с²
        a_ya_c: нормальное скоростное ускорение второй точки, м/с²
    
    Returns:
        Угловое ускорение линии визирования, рад/с²
    """
    r_rc = r(x_r, y_r, x_c, y_c)
    
    Delta_x = x_c - x_r
    Delta_y = y_c - y_r
    
    Delta_dot_x = v_c * math.cos(Theta_c) - v_r * math.cos(Theta_r)
    Delta_dot_y = v_c * math.sin(Theta_c) - v_r * math.sin(Theta_r)
    
    Delta_ddot_x = (a_xa_c * math.cos(Theta_c) - a_xa_r * math.cos(Theta_r) - 
                   a_ya_c * math.sin(Theta_c) + a_ya_r * math.sin(Theta_r))
    Delta_ddot_y = (a_xa_c * math.sin(Theta_c) - a_xa_r * math.sin(Theta_r) + 
                   a_ya_c * math.cos(Theta_c) - a_ya_r * math.cos(Theta_r))
    
    return ((Delta_x * Delta_ddot_y - Delta_y * Delta_ddot_x) / sqr(r_rc) -
            2.0 * (Delta_y * Delta_dot_y + Delta_x * Delta_dot_x) * 
            (Delta_x * Delta_dot_y - Delta_y * Delta_dot_x) / math.pow(r_rc, 4.0))



def r_3d(x_r: float, y_r: float, z_r: float, x_c: float, y_c: float, z_c: float) -> float:
    """Расстояние между двумя точками в 3D"""
    return math.sqrt(sqr(x_c - x_r) + sqr(y_c - y_r) + sqr(z_c - z_r))

def beta_3d(x_r: float, y_r: float, z_r: float, x_c: float, y_c: float, z_c: float) -> float:
    """Угол линии визирования в горизонтальной плоскости (азимут)"""
    delta_x = x_c - x_r
    delta_z = z_c - z_r
    
    if abs(delta_x) > 1e-10:
        return math.atan2(delta_z, delta_x)
    else:
        return 0.0

def dot_beta_3d(x_r: float, y_r: float, z_r: float, 
                x_c: float, y_c: float, z_c: float,
                v_r: float, v_c: float, 
                Theta_r: float, Theta_c: float,
                Psi_r: float, Psi_c: float) -> float:
    """
    Угловая скорость линии визирования в горизонтальной плоскости
    """
    r_val = r_3d(x_r, y_r, z_r, x_c, y_c, z_c)
    
    if r_val < 1e-6:
        return 0.0
    
    # Компоненты скорости
    v_rx = v_r * math.cos(Theta_r) * math.cos(Psi_r)
    v_rz = -v_r * math.cos(Theta_r) * math.sin(Psi_r)
    
    v_cx = v_c * math.cos(Theta_c) * math.cos(Psi_c)
    v_cz = -v_c * math.cos(Theta_c) * math.sin(Psi_c)
    
    # Относительные координаты и скорости
    delta_x = x_c - x_r
    delta_z = z_c - z_r
    delta_vx = v_cx - v_rx
    delta_vz = v_cz - v_rz
    
    # Формула для угловой скорости в горизонтальной плоскости
    return (delta_x * delta_vz - delta_z * delta_vx) / (r_val * r_val)