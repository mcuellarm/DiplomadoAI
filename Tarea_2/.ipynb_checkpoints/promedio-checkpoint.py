num1 = input("Ingrese un valor para la variable 1 y presione Enter: ")
num2 = input("Ingrese un valor para la variable 2 y presione Enter: ")
num3 = input("Ingrese un valor para la variable 3 y presione Enter: ")
num4 = input("Ingrese un valor para la variable 4 y presione Enter: ")

num1, num2, num3, num4 = float(num1), float(num2), float(num3), float(num4)

promedio = round((num1 + num2 + num3 + num4)/4, 3)

print("El promedio de los números ingresados es: ", promedio)