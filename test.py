import numpy as np
from numpy.linalg import inv

def calcular_costes_reducidos(cb, B_inv, An, cn):
    """Calcular costes reducidos para las no básicas"""
    cb_x_invB = cb @ B_inv  
    cb_x_invB_An = cb_x_invB @ An  
    r = cn - cb_x_invB_An 
    return r

def seleccionar_var_entrada(r):
    """Seleccionar la variable de entrada siguiendo la regla de Bland"""
    negativos = np.where(r < 0)[0]  
    if negativos.size == 0:
        return None  
    return int(negativos[0])  

def calcular_DBF_descenso(q, B_inv, A):
    """Calcular dB y comprobar que DBF de descenso es acotada"""
    Aq = A[:, q] 
    db = -B_inv @ Aq  
    db_min = np.copy(db)
    db_min[db_min > 0] = -0.0001  
    if np.all(db >= 0):  
        print('(PL) no acotado') 
        return
    return db, db_min

def calcular_theta_y_var_salida(Xb,db_min):
    """Calcular theta y p para elegir la variable de salida"""
    pretheta = np.where(db_min < 0, -Xb / db_min, np.inf)
    theta = np.min(pretheta)
    p = np.argmin(pretheta)
    return theta, p

def actualizar(Xb,z,theta,m, db,r,q,p,indices_basicas, indices_no_basicas,An,B,B_inv):
    """
    Actualiza las variables básicas y la función objetivo tras determinar theta y la variable de salida.
    """
    pos_q = np.where(indices_no_basicas == q)[0][0]

    # Intercambiar los valores
    indices_basicas[p], indices_no_basicas[pos_q] = indices_no_basicas[pos_q], indices_basicas[p]
    indices_no_basicas = np.sort(indices_no_basicas)
    print("indices_basicas: ", indices_basicas)
    print("indices_no_basicas: ", indices_no_basicas)
    print("pos_q: ",pos_q)
    print("p: ",p)
    print("theta: ",theta)

    col_An = pos_q
    col_B = p

    temp_An = An[:, col_An].copy()
    temp_B = B[:, col_B].copy()
    # Realizamos el intercambio
    An[:, col_An] = temp_B
    B[:, col_B] = temp_An

    print("An: ", An)
    print("B: ", B)
    #Xb
    theta_x_db= theta*db
    Xb_actual=np.zeros(m)

    for i in range(m):
        if i==p:
            Xb_actual[i]=theta
        else:
            Xb_actual[i]=Xb[i]+theta_x_db[i]
    print("Xb: ", Xb_actual)

    negativos = np.where(Xb < 0)[0]  
    if negativos.size != 0:
        print("Solución no factible")
        return
    #z
    z_float=float(z)
    z=z_float+r[q]*theta

    #B_inv
    print("B_inv: ", B_inv)
    E = np.eye(m)
    db = -B_inv @ A[:, q]  
    E[:, p] = db / db[p]  
    B_inv = E @ B_inv  

    B_inv_2 = inv(B)
    #Comprobar si B_inv es igual a B_inv_2
    if np.allclose(B_inv, B_inv_2):
        print("B_inv es igual a B_inv_2")
        print("B_inv: ", B_inv)
        print("B_inv_2: ", B_inv_2)
    else:
        print("B_inv no es igual a B_inv_2")
        print("B_inv: ", B_inv)
        print("B_inv_2: ", B_inv_2)

    return Xb_actual, z, indices_basicas, indices_no_basicas, An, B, B_inv_2



def iteracion_simplex(A,B,B_inv ,An,c, cb, cn, z, Xb, indices_basicas, indices_no_basicas, indices_inventadas,m,n):
    r = calcular_costes_reducidos(cb, B_inv, An, cn)
    q = seleccionar_var_entrada(r)
    if q is None:
            print("Solución optima encontrada")
            #Chequear factibilidad de la solución
    db, db_min = calcular_DBF_descenso(q,B_inv, A)
    theta,p = calcular_theta_y_var_salida(Xb,db_min)

    Xb,z_nueva,indices_basicas,indices_no_basicas,An,B, B_inv = actualizar(Xb,z,theta,m, db,r,q,p,indices_basicas, indices_no_basicas,An,B,B_inv)

    return iteracion_simplex(A,B,B_inv ,An,c, cb, cn, z_nueva, Xb, indices_basicas, indices_no_basicas, indices_inventadas,m,n)
    

def faseI(A,b):
    m, n = len(A), len(A[0])  # Número de restricciones y variables
    A_fase_1 = np.hstack((A, np.eye(m)))  # Matriz de restricciones con variables artificiales
    c_fase_1 = np.concatenate((np.zeros(n), np.ones(m)))  # Función objetivo de la fase I
    indices_basicas = np.arange(n, n + m)
    indices_no_basicas = np.arange(n)
    indices_inventadas = indices_basicas # Índices inventadas (la usamos para comprobar que en el final de la faseI no quedan variables artificiales)
    B = A_fase_1[:, indices_basicas] # Obtener B
    An = A_fase_1[:, indices_no_basicas] # Obtener An
    cn = c_fase_1[indices_no_basicas] # coeficientes de variables básicas de la función obj de la faseI 
    cb = c_fase_1[indices_basicas] # coeficientes de variables básicas de la función obj de la faseI 
    B_inv = inv(B) # Calcular B_inv 
    Xb = B_inv @ b # Calcular Xb
    z = cb @ Xb  # Calcular el valor de la función objetivo en la fase I
    # Verificar si es una SBF
    if np.all(Xb >= 0):
        return iteracion_simplex(A_fase_1, B, B_inv, An,c_fase_1,cb, cn, z, Xb, indices_basicas, indices_no_basicas, indices_inventadas,m,n)
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

faseI(A,b)

