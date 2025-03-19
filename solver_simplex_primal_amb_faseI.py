import numpy as np
from numpy.linalg import inv

def faseI(A, b):
    def aux_faseI(B, An, cb, cn, z, Xb, indices_basicas, indices_no_basicas):
        # Cálculo de los coeficientes reducidos
        cb_x_invB = cb @ B_inv  
        cb_x_invB_An = cb_x_invB @ An  
        r = cn - cb_x_invB_An 

        # Selección de la variable de entrada usando la regla de Bland
        negativos = np.where(r < 0)[0]  # Índices de valores negativos en r

        if negativos.size == 0:  # No hay valores negativos en r → solución óptima
            print("SBF óptima")
            # Verificación de la factibilidad en la fase I
            if z > 0:  # Si z > 0, la solución no es factible
                print("No factible")
            else:
                for indice in indices_basicas:  # Verificamos si alguna variable básica es artificial
                    if indice in indices_inventadas:
                        print("No se cumple la condición de factibilidad") 
            print(r, z, A, b, q, p)
            return
        else:
            q = negativos[0]  # Se elige la primera variable negativa siguiendo la regla de Bland

        q = np.array([q])  
        Aq = A_fase_1[:, q]  # Tomamos la columna q de A_fase_1

        db = -B_inv @ Aq  
        db_min = np.copy(db)  

        db_min[db_min > 0] = -0.0001  

        # Verificamos si el problema es no acotado
        if np.all(db >= 0):  
            print('(PL) no acotado')
            # stop

        # Calcular pretheta solo con valores donde db_min < 0
        valid_indices = db_min < 0  # Filtramos solo los valores negativos de db_min
        pretheta = -Xb[valid_indices] / db_min[valid_indices]  
        # Si no hay valores válidos, el problema es no acotado
        if pretheta.size == 0:
            print('(PL) no acotado')
        else:
            theta_num = float(np.min(pretheta))  # Mínimo de pretheta
            theta_index = np.argmin(pretheta)  # Índice del mínimo en pretheta

            p = int(theta_index)  
        
        # Actualización de Xb
        Xb_actual = Xb + theta_num * db  

        # Corrección de valores cercanos a 0
        Xb_actual[np.isclose(Xb_actual, 0)] = theta_num  

        # Actualización de Z
        z_actual_actualizacion = z + r[q] * theta_num  

        # Actualización de índices de variables básicas y no básicas
        indices_basicas[p], indices_no_basicas[q] = indices_no_basicas[q], indices_basicas[p]

        # Intercambio de columnas entre B y An
        An[:, q], B[:, p] = B[:, p].copy(), An[:, q].copy()

        # Ordenación de variables no básicas y actualización de An
        indices_no_basicas_sort = np.argsort(indices_no_basicas)
        An = An[:, indices_no_basicas_sort]

        # Actualización de costos
        cn = c_fase_1[indices_no_basicas[indices_no_basicas_sort]]
        cb = c_fase_1[indices_basicas]

        #Comprovaciones
        # Recalcular Xb usando la inversa de B
        B_inv_2 = np.linalg.inv(B)
        Xb_actual_2 = B_inv_2 @ b  

        # Recalcular Z
        z_actual = np.dot(cb, Xb_actual_2)

        if np.allclose(Xb_actual, Xb_actual_2) and np.isclose(z_actual, z_actual_actualizacion) and z_actual_actualizacion < z:
            print(".")
            aux_faseI(B, An, cb, cn, z_actual, Xb_actual, indices_basicas, indices_no_basicas_sort)


    m, n = len(A), len(A[0])  # Número de restricciones y variables

    # Matriz de restricciones con variables artificiales
    A_fase_1 = np.hstack((A, np.eye(m)))  

    # Función objetivo de la fase I
    c_fase_1 = np.concatenate((np.zeros(n), np.ones(m)))  

    # Índices de variables básicas, no básicas y inventadas (esta la usamos más adelante )
    indices_basicas = np.arange(n, n + m)
    indices_no_basicas = np.arange(n)
    indices_inventadas = indices_basicas

    # Obtener B y An
    B = A_fase_1[:, indices_basicas]
    An = A_fase_1[:, indices_no_basicas]

    # Costos de variables básicas y no básicas
    cn = c_fase_1[indices_no_basicas]
    cb = c_fase_1[indices_basicas]

    # Calcular B_inv y Xb
    B_inv = inv(B)
    Xb = B_inv @ b 

    # Calcular el valor de la función objetivo en la fase I
    z = cb @ Xb  

    # Verificar si es una SBF
    if np.all(Xb >= 0):
        aux_faseI(B, An, cb, cn, z, Xb, indices_basicas, indices_no_basicas)
    else:
        print("No es una SBF")
        return

    
A = np.array([[15, -36, 30, -54, 39, -92, 54, -76, 75, 30, 54, -36, -18, 79, 0, 0, 0, 0, 0, 0], 
     [-41, -38, 90, 74, 56, 37, 32, 92, 32, 35, 57, 64, 98, -51, 0, 0, 0, 0, 0, 0], 
     [-48, 38, -18, 20, 89, 15, 56, -64, -78, 35, 65, 55, -87, -23, 0, 0, 0, 0, 0, 0],
     [-65, 36, 77, 49, 74, 68, 56, -91, 37, 78, 9, 63, -36, -63, 0, 0, 0, 0, 0, 0],
     [100, 93, 50, 86, 50, 81, 67, 52, 52, 87, 61, 72, 68, 97, 1, 0, 0, 0, 0, 0],
     [-23, -27, -69, 40, -93, 48, -11, -51, 64, -82, 0, 47, 65, 97, 0, 1, 0, 0, 0, 0],
     [90, 18, 51, -78, 2, 98, -11, 74, -1, -7, 45, 70, 88, 1, 0, 0, 1, 0, 0, 0],
     [-3, 95, -57, -18, -33, 100, -18, 85, 84, -17, 42, 29, -52, 74, 0, 0, 0, 1, 0, 0],
     [47, 24, -19, 89, -74, 77, 13, 60, 83, 66, -17, -24, -3, 58, 0, 0, 0, 0, 1, 0],
     [-75, 43, 46, 72, 21, -99, 97, 92, 82, -25, 41, 50, -5, 57, 0, 0, 0, 0, 0, 1]])

#coeficientes de la función objetivo
c = np.array([44,18,60,27,-79,-51,92,-78,-69,-35,26,-56,-10,-55,0,0,0,0,0,0])

#términos independientes de las restricciones
b = np.array([64, 537, 55, 292, 1017, 6, 441, 312, 381, 398])

print(faseI(A,b))




    