#
# Autor: Juan R. Rico
# Asignatura: Desafíos de programación
#
# Descripción: En aprendizaje automático es común verificar si los resultados de dos experimentos
#   distintos son equivalentes, o por el contrario, uno es mejor que el otro.
#   Para ello podemos usar el test de Wilcoxon (wilcox.test) para averiguarlo. 
# 

#
# Versión online para trabajar con el lengueje estadístico R
# http://www.r-fiddle.org/
#

#
# Ejemplo de un par de vectores 'x' e 'y' con una serie de resultados para verificar.
# Supongamos que se trata de los porcentajes de aciertos de dos clasificadores ante los mismos datos.
#

x <- c(81,82,86,88,90,92,93,91)
y <- c(81,83,85,87,89,91,90,90)

# Queremos comprobar si los resutados de 'x' son mejores que los de 'y' sabiendo que las pruebas
# se comparan por pares (paired=TRUE)

wilcox.test(x,y,paired=TRUE,alternative="greater")

# Como el p-value es 0.03538 < 0.05, rechazamos la hipótesis nula y determinamos que 'x' es mejor que 'y' 
# con una confianza mayor del 95%
