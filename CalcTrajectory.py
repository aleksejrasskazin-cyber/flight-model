import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import math
from copy import deepcopy

 
from DATASTRUCTS import *
from Model import *

def CalcTrajectory(calcParams: CalcParameters, inits: InitConditions, 
                  limits: LimitConditions, CData: C_Data, NData: N_Data,
                  RData: R_Data, MethodData: GuidanceMethod_Data, 
                  s: List[TrajectoryParameters]) -> str:
    """
    Основная функция для расчета траектории
    """
    
    dt = calcParams.dt
    N_max = math.ceil(limits.t_max / dt)
    
     
    s.clear()
    s.extend([TrajectoryParameters() for _ in range(N_max)])
    
   
    model = Model(calcParams, inits, limits, CData, NData, RData, MethodData)
    model.Init()
    
     
    i = 0
    s[i] = deepcopy(model.s)   
    s[i].t = 0.0
    
    returnCode = "t_max"
    
    while s[i].t < limits.t_max and i < N_max - 1:
        model.i = i
        model.s_last = deepcopy(model.s)
        
        model.DoStep(s[i].t, dt)
        
         
        s[i + 1] = deepcopy(model.s)   
        s[i + 1].t = s[i].t + dt   
        
        # Проверка ошибок
        if model.error != "":
            returnCode = model.error
            break
        
        # Проверка ограничений
        if abs(s[i + 1].n_ya_r) > limits.n_ya_r_max:
            returnCode = "n_ya_r_max"
            break
        
        if s[i + 1].r_rc < calcParams.r_por:
            if s[i + 1].t > limits.t_min:
                returnCode = "0"  # успешное завершение
                break
            else:
                returnCode = "t_min"
                break
            
        if not math.isnan(limits.y_r_min) and s[i + 1].y_r <= limits.y_r_min: # ограничение на встречу с землёй 
                returnCode = "y_r_min"  # достигнута минимальная высота
                break
        # if ((s[i + 1].n_ya_r > s[i + 1].n_ya_rasp_plus and s[i + 1].n_ya_r > 0) or 
        #     (s[i + 1].n_ya_r < s[i + 1].n_ya_rasp_minus and s[i + 1].n_ya_r < 0)):
        #     returnCode = "n_ya_rasp"
        #     break
        
        i += 1
    
     
    del s[i+1:]
    
    return returnCode

 
def CalcTrajectoryWithReturn(calcParams: CalcParameters, inits: InitConditions, 
                           limits: LimitConditions, CData: C_Data, NData: N_Data,
                           RData: R_Data, MethodData: GuidanceMethod_Data) -> Tuple[str, List[TrajectoryParameters]]:
     
    s = []
    return_code = CalcTrajectory(calcParams, inits, limits, CData, NData, RData, MethodData, s)
    return return_code, s