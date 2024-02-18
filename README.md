# Construyendo un árbol de decisión en paralelo con python
 Este es un proyecto donde se paraleliza un arbol de decisión utilizando el algoritmo C4.5

# En que consiste este proyecto?
 Se pretende construir un árbol de decisión aplicando programación paralela para mejorar la eficiencia de ese algoritmo. El algoritmo a utilizar es C4.5, la explicación del algoritmo la pueden encontrar aqui:
    https://anderfernandez.com/blog/programar-arbol-decision-python-desde-0/

Para paralelizar se pretende construir el árbol a lo ancho, encolando cada llamada recursiva que generaría un nuevo nodo. Para ello se utilizará la biblioteca multiprocessing de python que permite generar paralelismo sin los impedimentos del General Interpreter Lock de Python
