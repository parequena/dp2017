/*
	La distancia de edición o de Levenshtein calcula el número de operaciones
	mı́nimas para transformar una cadena de caracteres en otra. Inicialmente
	tuvo interés en el área de biologı́a para comparar secuencias de ADN y saber
	cuanto se parecı́an. En imagen 2D se pueden representar objetos mediante sus
	contornos codificados con las 8 posibles direcciones en 2D (llamadas códigos
	de Freeman). Podemos ver un ejemplo en la figura 1.

	borrado b(a): del carácter "a" de la cadena s1 .
	inserción i(b): del carácter "b" en la cadena s2 .
	sustitución s(a, b): del carácter "a" de s1 , y de "b" de s2 .
	insetado w(ab, ba): del par "ab" de s1 , y de "ba" de s2 .
*/
// Solución NO optima para el problema P100, aunque cumple el 100% de los casos (lento, sin optimizar).
package AA;

public class P100
{
	private static String data1 = "356";
	private static String data2 = "365";

//	public static void main(String[] args)
//	{
//		String[] data = {data1, data2};
//
//		System.out.println(bestSolution(data));
//	}

    public static String bestSolution(String[] data)
    {
        // System.out.println("data[0]: " + data[0]);
        // System.out.println("data[1]: " + data[1]);

    	return DamLevenDistance(data[0].toCharArray(), data[1].toCharArray());
    }

    private static int min(int a, int b, int c)
    {
    	return Math.min(a, Math.min(b, c));
    }

    private static String DamLevenDistance(char[] str1, char[] str2)
    {
        // Creamos los auxilares para el tamaño.
    	int s1Size = str1.length;
    	int s2Size = str2.length;

    	// Creamos las matrices de tam + 1
        int [][]distance = new int[s1Size + 1][s2Size + 1];
        String [][] almacen = new String[s1Size + 1][s2Size + 1];

        String aux;

        // Rellenamos distance y el almacen, para cadena2 vacía
        for(int i = 0; i <= s1Size; i++)
        {
            distance[i][0] = i;
            // almacen[i][0] = almacen[i - 1][0] + "b" + str1[i];
        }

        // Rellenamos distance y el almacen, para cadena1 vacía
        for(int i = 0; i <= s2Size; i++)
        {
            distance[0][i] = i;
            // almacen[0][i] = almacen[0][i - 1] + "i" + str2[i];
        }

        int coste = 0;
        int a, b, c;

        // Rellenamos las matrices.
        for(int i = 1; i <= s1Size; i++)
        {
            for(int j = 1; j <= s2Size; j++)
            {
            	if(str1[i - 1] == str2[j - 1])
            		coste = 0;
            	else coste = 1;

                a = distance[i - 1][j] + 1; // Borrado
                b = distance[i][j - 1] + 1; // Insertado
                c = distance[i - 1][j - 1] + coste; // Sustitucion
                // aux = almacen[i - 1][j - 1];

                distance[i][j] = min(a,b,c);

                if(i > 1 && j > 1 && (str1[i - 1] == str2[j - 2] && str1[i - 2] == str2[j - 1])) //  && str1[i - 1] != str1[i - 2]) // No estoy seguro de que haga falta la última comprobación.
                 {
                 	//System.out.println("linea");
                    distance[i][j] = Math.min(distance[i][j], distance[i - 2][j - 2] + coste);

                 	//System.out.println(distance[i][j]);
                 }

                if(distance[i][j] == a)
                {
                	// Borrado
                	almacen[i][j] = "b" + str1[i - 1];
                }
                else if(distance[i][j] == b)
                {
                	almacen[i][j] = "i" + str2[j - 1];
                }
				else if(distance[i][j] == c)
                {
                	almacen[i][j] = "s" + str1[i - 1] + str2[j - 1];
                }
                else
                {
                	almacen[i][j] = "w" + str1[i - 2] + str1[i - 1] + str2[j - 2] + str2[j - 1];
                }
            }
        }

        String ret = "";


        int i=s1Size,j=s2Size;
        
        while(i >= 1 && j >= 1)
        {
        	if(almacen[i][j].charAt(0) == 'i')
        	{
        		ret = almacen[i][j] + ret;
        		--j;
        	}
        	else if(almacen[i][j].charAt(0) == 'b')
        	{
        		ret = almacen[i][j] + ret;
        		--i;
        	}
        	else if(almacen[i][j].charAt(0) == 's')
        	{
        		ret = almacen[i][j] + ret;
        		--i;
        		--j;
        	}
        	else
        	{
        		ret = almacen[i][j] + ret;
        		--i;
        		--j;
        		--i;
        		--j;	
        	}
        }

        return ret;
    }
}
