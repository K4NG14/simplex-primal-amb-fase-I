import numpy as np
import re
from solver import *
def parse_file(file_path, alumno_num):
    with open(file_path, 'r') as file:
        content = file.read()
    alumno_section = f"::  OPT/GIA/FIB Curso 2022-23 : alumno {alumno_num} ::"
    alumno_index = content.find(alumno_section)
    transition_section = ":::::::::::::::::::::::::::::::::::::::::"
    start_index = content.find(transition_section, alumno_index )
    if start_index == -1:
        return None  
    end_index = content.find(":::::::::::::::::::::::::::::::::::::::::", start_index + len(alumno_section))
    alumno_content = content[start_index + len(transition_section):end_index]

    return alumno_content

def c_unir_cols(text):
    print(text)
    col1_start = text.find('Column')
    col2_start = text.find('Column', col1_start + 1)
    if col1_start != -1:
        col_space = text.find('\n', col1_start + 1)
        col1_end = col2_start if col2_start != -1 else len(text)-1
        col1 = text[col_space:col1_end].strip()
    if col2_start != -1:
        col_space = text.find('\n', col2_start + 1)
        col2_end = len(text)
        col2 = text[col_space:col2_end].strip()
        col1 = np.fromstring(col1, sep=" ")
        col2 = np.fromstring(col2, sep=" ")
        
        text = np.concatenate((col1, col2))
        return text
    text = np.fromstring(text, sep=" ")    
    return text
   
def A_unir_columnas(text):
    if 'Column' in text:
        chunck1_start = text.find('Column')
        chunck2_start = text.find('Column', chunck1_start + 1)
        chunck1_lineas = [line for line in text[:chunck2_start].split("\n")]
        num_cols_1 = chunck1_lineas[2].split(" ")
        num_cols_1 = [col for col in num_cols_1 if col != '']
        chunck1 = np.fromstring(" ".join(chunck1_lineas[2:]), sep=" ").reshape(-1, len(num_cols_1))
        chunck2_lineas = [line for line in text[chunck2_start:].split("\n")]
        num_cols_2 = chunck2_lineas[2].split(" ")
        num_cols_2 = [col for col in num_cols_2 if col != '']
        chunck2 = np.fromstring(" ".join(chunck2_lineas[2:]), sep=" ").reshape(-1, len(num_cols_2))
        text = np.concatenate((chunck1, chunck2), axis=1)
    else:
        lineas = [line for line in text.split("\n") if 'Column' not in line]
        chunck1 = np.fromstring(lineas[1], sep=" ")
        num_cols = len(chunck1)
        text = np.fromstring(" ".join(lineas), sep=" ").reshape(-1, num_cols)     
    return text
    
    

def parse_simplex_problems(input_text):
    
    problem_sections = input_text.split('--------------------------------------------------------------------------------------------')[2:]
    problems = {}
    cnt = 1
    for problem_section in problem_sections:
        A,b,c,z,vb = None, None, None, None, None
        ind = problem_section.find(f"OPT/GIA/FIB")
        if ind != -1:
            continue
        c_start = problem_section.find('c=')
        c_end = problem_section.find('A=', c_start)
        c = problem_section[c_start + 2:c_end].strip()
        c = c_unir_cols(c)
        A_start = c_end
        A_end = problem_section.find('b=', A_start)
        A = problem_section[A_start + 2:A_end].strip()
        A = A_unir_columnas(A)
        b_start = A_end
        z_start = problem_section.find('z*=', b_start)
        if z_start == -1:
            b_end = len(problem_section)-1
        else:
            b_end = z_start
            z_end = problem_section.find('vb*=', z_start)
            z = problem_section[z_start + 3:z_end].strip()
            z = float(z)
            vb_start = z_end
            vb_end = len(problem_section)-1
            vb = problem_section[vb_start + 4 : vb_end].strip()
            vb = np.fromstring(vb, sep=" ")
        b = problem_section[b_start + 2:b_end].strip()
        b = np.fromstring(b, sep=" ")
        problems[cnt] = {'A': A, 'b': b, 'c': c, 'z': z, 'vb': vb}
        cnt += 1
    return problems
    



#::  OPT/GIA/FIB Curso 2022-23 : alumno 10 ::
#::  OPT/GIA/FIB Curso 2022-23 : alumno  9 ::
#alumno_num = input("Introduce el número de alumno: ")
for alumno_num in range(1, 66):
    alumno_num = str(alumno_num)
    if len(alumno_num) == 1:
        alumno_num = f" {alumno_num}"
    file_path = 'datos.txt'
    result = parse_file(file_path, alumno_num)

    if result is None:
        print("Alumno no encontrado")
        exit() 

    parsed_problems = parse_simplex_problems(result)
    for i, problem in parsed_problems.items():
        print(f"Problema {i}:")
        """ print(f"A: {problem['A']}")
        print(f"b: {problem['b']}")
        print(f"c: {problem['c']}")
        print(f"z: {problem['z']}")
        print(f"vb: {problem['vb']}") """
        A = problem['A']
        b = problem['b']
        c = problem['c']
        resultado = faseI(A, b, c)
        solucion, z_opt, indices_basicas, Xb, r = resultado
        if solucion is not None:
            print()
            print("Solució òptima:")
            print(f"vb = {' '.join(map(str, indices_basicas+1))}")
            print(f"xb = {' '.join([f'{val:.2f}' for val in Xb])}")
            print(f"z = {z_opt:.4f}" if problem['z'] is None else f"z = {z_opt:.4f}, z* = {problem['z']}")
            print(f"r = {' '.join([f'{val:.2f}' for val in r if val > 1e-10])}")
            print()
            with open("solucion.txt", "a") as f:
                f.write("\n")
                f.write(f"Conjunt de dades {alumno_num}, problema {i}\n")
                f.write("Solucio optima:\n")
                f.write(f"vb = {' '.join(map(str, indices_basicas+1))}\n")
                f.write(f"xb = {' '.join([f'{val:.2f}' for val in Xb])}\n")
                f.write(f"z = {z_opt:.4f}\n" if problem['z'] is None else f"z = {z_opt:.4f}, z* = {problem['z']}\n")
                f.write(f"r = {' '.join([f'{val:.2f}' for val in r if val > 1e-10])}\n")
                f.write("\n")

                    



            """
            for i in range(1, 66):
                i = str(i)
                if len(i) == 1:
                    i = f" {i}"
                print(f"Alumno {i}:")
                result = parse_file('datos.txt', i)
                if result is None:
                    print("Alumno no encontrado")
                    continue
                parsed_problems = parse_simplex_problems(result)
                for x, problem in parsed_problems.items():
                    print(f"Problema {x}:")
                    print(f"A: {problem['A']}")
                    print(f"b: {problem['b']}")
                    print(f"c: {problem['c']}")
                    print(f"z: {problem['z']}")
                    print(f"vb: {problem['vb']}")
                    print() 
                    pass
            """
    


