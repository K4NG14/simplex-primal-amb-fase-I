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
    db_min[db_min >= -1e-10] = 0  # Usar una tolerancia pequeña
    if np.all(db >= -1e-10):  
        
        return None, None
    return db, db_min

def calcular_theta_y_var_salida(Xb, db_min):
    """Calcular theta y p para elegir la variable de salida"""
    # Usar una tolerancia para valores cercanos a cero
    pretheta = np.where(db_min < -1e-10, -Xb / db_min, np.inf)
    if np.all(pretheta == np.inf):
        return 0, 0  # Caso especial: ninguna restricción limita
    theta = np.min(pretheta)
    p = np.argmin(pretheta)
    return theta, p

def actualizar(Xb, z, theta, m, db, r, q, p, indices_basicas, indices_no_basicas, An, B, B_inv, A):
    """
    Actualiza las variables básicas y la función objetivo tras determinar theta y la variable de salida.
    """
    # Intercambiar los índices
    temp = indices_basicas[p]
    indices_basicas[p] = q
    
    # Actualizar índices no básicos (quitar q y añadir el que salió)
    indices_no_basicas = np.array([i for i in range(A.shape[1]) if i not in indices_basicas])
    
    # Actualizar las matrices B y An
    B = A[:, indices_basicas]
    An = A[:, indices_no_basicas]
    
    # Actualizar Xb
    Xb_actual = np.zeros(m)
    for i in range(m):
        if i == p:
            Xb_actual[i] = theta
        else:
            Xb_actual[i] = Xb[i] + theta * db[i]
    
    # Verificar factibilidad
    if np.any(Xb_actual < -1e-10):  # Tolerancia numérica
        print("Solución no factible")
        return None, None, None, None, None, None, None
    
    # Actualizar z
    z_nuevo = float(z) + r * theta
    
    # Calcular y_k = B_inv * a_k (columna que entra)
    a_k = A[:, q]
    y_k = B_inv @ a_k
    
    # Construir la matriz de transformación E
    E = np.eye(m)
    E[:, p] = -y_k / y_k[p]
    E[p, p] = 1 / y_k[p]
    
    # Actualizar B_inv usando E
    B_inv_actualizada = E @ B_inv
    
    # Verificar la precisión comparando con inv(B)
    """
    try:
        B_inv_directa = inv(B)
        print(f"B_inv correctamente actualizada: {np.allclose(B_inv_actualizada, B_inv_directa)}")
    except np.linalg.LinAlgError:
        print("Error al calcular la inversa de B directamente")
        return None, None, None, None, None, None, None
    """
    return Xb_actual, z_nuevo, indices_basicas, indices_no_basicas, An, B, B_inv_actualizada

def iteracion_simplex(A, B, B_inv, An, c, cb, cn, z, Xb, indices_basicas, indices_no_basicas, indices_inventadas, m, n, fase, max_iter=100):
    """Iteración del método simplex con límite de iteraciones para evitar bucles infinitos"""
    iter_count = 0
    total_iter = 0  # Para contar las iteraciones totales
    
    if 'total_iteraciones' in globals():
        total_iter = globals()['total_iteraciones']
    
    while iter_count < max_iter:
        # Calcular costes reducidos
        r = calcular_costes_reducidos(cb, B_inv, An, cn)
        
        # Seleccionar variable de entrada
        q_idx = seleccionar_var_entrada(r)
        if q_idx is None:
            #print(f"Solució {'òptima' if fase == 'II' else 'bàsica factible'} trobada, iteració {total_iter}")
            if fase == 'II':
                print(f"Iteració {total_iter} : q = {q if 'q' in locals() else 'N/A'}, B(p) = {indices_basicas[p] if 'p' in locals() else 'N/A'}, theta*= {theta if 'theta' in locals() else 'N/A'}, z = {z:.6f}")
                print(f"Solució òptima trobada, iteració {total_iter}, z = {z:.6f}")   
                # Construir la solución completa
                solucion_completa = np.zeros(n)
                for i, idx in enumerate(indices_basicas):
                    if idx < n:  # Solo considerar variables originales
                        solucion_completa[idx] = Xb[i]
                
                # Calcular costes reducidos finales para todas las variables
                r_final = np.zeros(n)
                for i, idx in enumerate(indices_no_basicas):
                    if idx < n:  # Solo para variables originales
                        r_final[idx] = r[i]
                
                return solucion_completa, z, indices_basicas, Xb, r_final, total_iter
            else:
                print(f"Iteració {total_iter} : q = {q if 'q' in locals() else 'N/A'}, B(p) = {indices_basicas[p] if 'p' in locals() else 'N/A'}, theta*= {theta if 'theta' in locals() else 'N/A'}, z = {z:.6f}")
                return indices_basicas, Xb, z, total_iter
        
        q = indices_no_basicas[q_idx]
        
        # Calcular dirección de descenso
        db, db_min = calcular_DBF_descenso(q, B_inv, A)
        if db is None:  # Problema no acotado
            print(f"Problema {'no acotado' if fase == 'II' else 'no factible'}")
            return None, None, None, None, None, None
        
        # Calcular theta y variable de salida
        theta, p = calcular_theta_y_var_salida(Xb, db_min)
        
        # Mostrar información de la iteración en el formato requerido
        total_iter += 1
        globals()['total_iteraciones'] = total_iter
        print(f"Iteració {total_iter} : q = {q}, B(p) = {indices_basicas[p]}, theta*= {theta:.3f}, z = {z:.6f}")
        
        # Actualizar
        resultado = actualizar(Xb, z, theta, m, db, r[q_idx], q, p, 
                              indices_basicas, indices_no_basicas, An, B, B_inv, A)
        
        if resultado is None:  # Error en la actualización
            return None, None, None, None, None, None
            
        Xb, z, indices_basicas, indices_no_basicas, An, B, B_inv = resultado
        
        # Actualizar cb y cn para la siguiente iteración
        cb = c[indices_basicas]
        cn = c[indices_no_basicas]
        
        iter_count += 1
    
    print(f"Se alcanzó el límite máximo de {max_iter} iteraciones sin convergencia")
    return None, None, None, None, None, None


def faseI(A, b, c, tol=1e-10):
    """
    Fase I del método simplex para encontrar una solución básica factible inicial.
    """
    m, n = A.shape  # Número de restricciones y variables
    globals()['total_iteraciones'] = 0
    
    # Crear matriz A_fase_1 añadiendo variables artificiales
    A_fase_1 = np.hstack((A, np.eye(m)))
    
    # Función objetivo de la fase I: minimizar la suma de variables artificiales
    c_fase_1 = np.concatenate((np.zeros(n), np.ones(m)))
    #print(c_fase_1)
    # Inicializar con variables artificiales como base
    indices_basicas = np.arange(n, n + m)
    indices_no_basicas = np.arange(n)
    
    # Obtener matrices para la primera iteración
    B = A_fase_1[:, indices_basicas]  # B es la matriz identidad en la fase I
    An = A_fase_1[:, indices_no_basicas]
    cb = c_fase_1[indices_basicas]
    cn = c_fase_1[indices_no_basicas]
    #print(cb,cn)
    # Calcular B_inv y Xb inicial
    B_inv = np.eye(m)  # B_inv es identidad en la fase I
    Xb = np.copy(b)  # Como B es identidad, Xb = b en la primera iteración
    
    # Asegurar que b sea no negativo
    if np.any(Xb < 0):
        # Multiplicar filas con b[i] < 0 por -1
        for i in range(m):
            if Xb[i] < 0:
                Xb[i] = -Xb[i]
                A_fase_1[i, :] = -A_fase_1[i, :]
                A[i, :] = -A[i, :]
                b[i] = -b[i]
    
    # Calcular z inicial
    z = cb @ Xb
    
    print("Inici simplex primal amb regla de Bland")
    print("Fase I")
    
    # Ejecutar iteraciones de la Fase I
    indices_basicas_fase_I, Xb_fase_I, z_fase_I, iter_fase_I = iteracion_simplex(
        A_fase_1, B, B_inv, An, c_fase_1, cb, cn, z, Xb, 
        indices_basicas, indices_no_basicas, indices_basicas, m, n + m, 'I'
    )
    #iter_fase_I += 1
    #print(f"Iteració {iter_fase_I} : q = {0}, B(p) = {0}, theta*= {0:.3f}, z = {z_fase_I:.6f}")

    if indices_basicas_fase_I is None:
        print("La Fase I no encontró solución factible")
        return None, None, None, None, None
    
    # Verificar si la solución tiene variables artificiales en la base
    artificiales_en_base = [i for i in indices_basicas_fase_I if i >= n]
    valores_artificiales = [Xb_fase_I[indices_basicas_fase_I.tolist().index(i)] for i in artificiales_en_base]
    
    if any(v > tol for v in valores_artificiales):
        print(" El problema original no tiene solución factible")
        return None, None, None, None, None
    
    print("Fase II")
    
    # Eliminar variables artificiales y preparar para la Fase II
    indices_basicas_fase_II = []
    for i, idx in enumerate(indices_basicas_fase_I):
        if idx < n:  # Variable original
            indices_basicas_fase_II.append(idx)
        else:  # Variable artificial
            # Buscar una variable original para reemplazar la artificial
            for j in range(n):
                if j not in indices_basicas_fase_II and abs(A[i, j]) > tol:
                    indices_basicas_fase_II.append(j)
                    break
    
    # Verificar que tengamos m variables en la base
    if len(indices_basicas_fase_II) < m:
        print(" No se pueden eliminar todas las variables artificiales")
        return None, None, None, None, None
    
    indices_basicas_fase_II = np.array(indices_basicas_fase_II)
    indices_no_basicas_fase_II = np.array([j for j in range(n) if j not in indices_basicas_fase_II])
    
    # Configurar para la Fase II
    B_fase_II = A[:, indices_basicas_fase_II]
    An_fase_II = A[:, indices_no_basicas_fase_II]
    try:
        B_inv_fase_II = inv(B_fase_II)
    except np.linalg.LinAlgError:
        print("Error al calcular la inversa de B para la Fase II")
        return None, None, None
    
    Xb_fase_II = B_inv_fase_II @ b
    cb_fase_II = c[indices_basicas_fase_II]
    cn_fase_II = c[indices_no_basicas_fase_II]
    z_fase_II = cb_fase_II @ Xb_fase_II
    
    # Ejecutar Fase II
    solucion_fase_II, z_fase_II, indices_basicas_final, Xb_final, r_final, iter_total = iteracion_simplex(
        A, B_fase_II, B_inv_fase_II, An_fase_II, c, cb_fase_II, cn_fase_II, z_fase_II, Xb_fase_II,
        indices_basicas_fase_II, indices_no_basicas_fase_II, [], m, n, 'II'
    )
    
    if solucion_fase_II is None:
        print("La Fase II no encontró solución óptima")
        return None, None, None,None,None
    
    print("Fi simplex primal")
    
    return solucion_fase_II, z_fase_II, indices_basicas_final, Xb_final, r_final

"""
c = np.array([-85, -18, -79, -100, -54, -14, -35, -52, -22, -43, -47, -42, -69, -86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

A = np.array([
    [38, 33, 99, 28, 25, 10, 51, 77, 67, 22, 7, 64, 33, 44, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [31, 64, 58, 66, 73, 29, 24, 79, 36, 9, 24, 10, 80, 94, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [90, 13, 77, 46, 59, 52, 5, 76, 31, 12, 74, 43, 43, 75, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    [77, 42, 7, 32, 70, 62, 39, 17, 47, 58, 61, 13, 38, 21, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [31, 3, 7, 47, 91, 29, 52, 68, 70, 28, 48, 8, 18, 28, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [74, 92, 43, 11, 86, 47, 43, 43, 97, 54, 57, 86, 36, 5, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
    [17, 82, 45, 71, 71, 97, 47, 48, 65, 46, 100, 97, 33, 3, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
    [2, 55, 94, 83, 65, 31, 84, 8, 48, 30, 80, 77, 85, 52, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [20, 22, 50, 7, 89, 23, 95, 45, 19, 1, 29, 73, 22, 85, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    [47, 10, 56, 69, 96, 99, 19, 84, 6, 77, 78, 8, 70, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
])

b = np.array([597, 676, 695, 583, 527, 773, 821, 793, 579, 806])


# Datos del problema ajustados para obtener la solución óptima correcta
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

# Coeficientes de la función objetivo (negativo para maximizar)
c = np.array([44, 18, 60, 27, -79, -51, 92, -78, -69, -35, 26, -56, -10, -55, 0, 0, 0, 0, 0, 0])

# Términos independientes de las restricciones
b = np.array([64, 537, 55, 292, 1017, 6, 441, 312, 381, 398])



c = np.array([65, 18, -41, -76, 96, -84, -59, 91, -29, 28, -32, -71, -3, 69, 0, 0, 0, 0, 0, 0])

A = np.array([
  [-2, 5, 67, -50, 38, 19, -55, 7, -63, -1, 62, 8, -27, 97, 0, 0, 0, 0, 0, 0],
  [95, -15, 15, 33, 100, -34, 81, 77, 53, 45, -91, 19, -85, 72, 0, 0, 0, 0, 0, 0],
  [-96, -7, -73, -6, 31, 88, 36, -24, 25, -9, 20, 5, -9, 52, 0, 0, 0, 0, 0, 0],
  [-17, 19, 58, 69, 46, -54, -97, 59, 96, -89, -50, 100, -97, 55, 0, 0, 0, 0, 0, 0],
  [52, 58, 82, 75, 83, 97, 85, 88, 61, 98, 61, 58, 86, 65, 1, 0, 0, 0, 0, 0],
  [52, 74, 43, 88, -61, 72, 50, -3, 39, -88, -15, -64, -77, -36, 0, 1, 0, 0, 0, 0],
  [-97, 49, 2, 77, 41, 68, 11, 66, 66, 47, 2, -16, -76, -31, 0, 0, 1, 0, 0, 0],
  [88, -20, 43, 95, 97, -45, -66, -58, 70, -3, -52, 55, 60, 55, 0, 0, 0, 1, 0, 0],
  [95, -4, -61, -15, -76, 71, -87, 91, 7, 79, -81, -82, 47, 40, 0, 0, 0, 0, 1, 0],
  [63, 50, -12, 96, 93, 10, 96, -45, 58, -94, 89, -74, 53, 36, 0, 0, 0, 0, 0, 1]
])

b = np.array([105, 365, 33, 98, 1050, 75, 210, 320, 25, 420])
"""
# Ejecutar el algoritmo
""" resultado = faseI(A, b, c)
solucion, z_opt, indices_basicas, Xb, r = resultado
if solucion is not None:
    print()
    print("Solució òptima:")
    print(f"vb = {' '.join(map(str, indices_basicas+1))}")
    print(f"xb = {' '.join([f'{val:.2f}' for val in Xb])}")
    print(f"z = {z_opt:.4f}")
    print(f"r = {' '.join([f'{val:.2f}' for val in r if val > 1e-10])}")
    print() """
