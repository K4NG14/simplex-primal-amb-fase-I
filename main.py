def parse_file(file_path, alumno_num):
    with open(file_path, 'r') as file:
        content = file.read()

    # Buscar la sección del alumno específico
    alumno_section = f"::  OPT/GIA/FIB Curso 2022-23 : alumno {alumno_num} ::"
    alumno_index = content.find(alumno_section)
    transition_section = ":::::::::::::::::::::::::::::::::::::::::"
    start_index = content.find(transition_section, alumno_index )
    if start_index == -1:
        return None  # Alumno no encontrado

    # Extraer la sección del alumno
    end_index = content.find(":::::::::::::::::::::::::::::::::::::::::", start_index + len(alumno_section))
    print(start_index, end_index)
    alumno_content = content[start_index + len(transition_section):end_index]

    return alumno_content


# Ejemplo de uso
#::  OPT/GIA/FIB Curso 2022-23 : alumno 10 ::
#::  OPT/GIA/FIB Curso 2022-23 : alumno  9 ::
alumno_num = input("Introduce el número de alumno: ")
print(len(alumno_num))
if len(alumno_num) == 1:
    alumno_num = f" {alumno_num}"
print(alumno_num)
file_path = 'datos.txt'
result = parse_file(file_path, alumno_num)
print(result)
