from math import cos,sin,sqrt,pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


print("Este programa calcula la trayectoria de una bola de béisbol , para los siguientes casos:")
print("1. Sin efecto Magnus, a nivel del mar")
print("2. Sin efecto Magnus, a 3715 m de altura")
print("3. Sin efecto Magnus, a 12500 m de altura")
print("4. Con efecto Magnus, a nivel del mar")
print("5. Distintos tipos del lanzamiento del pitcher\n")

#Definimos las constantes necesarias

g=9.8                      #Gravedad
x0, y0, z0= 0., 0., 0.     #Punto inicial
t0=0                       #Tiempo inicial
dt=0.01                    #Paso de tiempo
rho_0=1.225                #Densidad del aire al nivel del mar
a=6.5*10**(-3)             #Constante de temperatura
T0=288.15                  #Temperatura al nivel del mar
alpha=2.5                  #Exponente de temperatura
vd, D = 35, 5              #Constantes para el coeficiente de arrastre
HR=168                     #Distancia necesaria para hacer home run
z02=3715                   #Altura a la que se encuentra el campo de beisbol 2
z03=12500                  #Altura a la que se encuentra el campo de beisbol 3
z_0=10000                  #Constante de altura en la atmósfera isotérmica
S0m=4.1*10**(-4)           #Coeficiente de Magnus
w=314.16                   #Velocidad angular de la bola de beisbol (3000RPM)


# %%
# En primer lugar, calcularemos el ángulo theta de lanzamiento para el cual v0 para hacer home run se hace mínimo
# Lo calculamos en el caso ideal y lo comparamos con la solución analítica, para un ángulo de 45º
# Esto nos permite ver si nuestro código es correcto

alcance=0.
v0=0.
theta=45.       #Ángulo de lanzamiento ideal

#Paramos el bucle cuando la pelota haga HR:                       
while alcance<HR:                
    #Variamos v0 para hallar la mínima    
    v0+=1                        
    v0x=v0*(np.cos(np.radians(theta)))      
    v0z=v0*(np.sin(np.radians(theta)))
    #Listas para la trayectoria de la bola de béisbol
    t,x,z,vx,vz=[t0],[x0],[z0],[v0x],[v0z]  

    i=0
    #Paramos cuando la pelota llegue al suelo
    while z[i]>=0:
        i+=1
        t.append(t[i-1]+dt)
        x.append(x[i-1]+vx[i-1]*dt)
        z.append(z[i-1]+vz[i-1]*dt)

        vx.append(vx[i-1])
        vz.append(vz[i-1]-g*dt)
    
    alcance=(x[-1]+(x[-2]-x[-1])/2)
    
v0_num=v0

#Ahora, lo calculamos analíticamente para ver si nuestro programa funciona

v0_an=np.sqrt((168*g)/(np.sin(2*np.radians(theta))))

#Exponemos los resultados
print("En el caso ideal a 45º la velocidad inicial calculada es:")
print(f"{v0} m/s mediante métodos numéricos")
print(f"{v0_an:.{2}f} m/s mediante métodos analíticos\n")

# Si ambos coinciden, nuestro programa funciona correctamente
# A continuación, calculamos theta óptimo en el caso no ideal (v0 sera mayor que en el caso previo)

# listv0=[]                       # Lista para guardar los v0 de cada ángulo
# rangetheta=np.arange(30,41,1)   # Distintos ángulos de lanzamiento
# for theta in rangetheta:
#     v0=v0_num
#     alcance=0.                                                     
#     while alcance<HR:                # Paramos el bucle cuando la pelota haga HR
#         v0+=0.01
#         v0x=v0*(np.cos(np.radians(theta)))
#         v0z=v0*(np.sin(np.radians(theta)))
#         # Listas para la trayectoria de la bola
#         t,x,z,vx,vz,v=[t0],[x0],[z0],[v0x],[v0z],[v0]             
#         i=0
#         while z[i]>=0:
#             i+=1
#             t.append(t[i-1]+dt)
#             x.append(x[i-1]+vx[i-1]*dt)
#             z.append(z[i-1]+vz[i-1]*dt)
        
#         # Nos encontramos en el modelo de atmósfera adiabática
#             B2m=0.0039+0.0058/(1+np.exp((v[i-1]-vd)/D))                                         
#             rho= rho_0*(1-(a*z[i])/T0)**alpha               
#             B2m_=B2m*rho/rho_0                              

#             vx.append(vx[i-1]-B2m_*v[i-1]*vx[i-1]*dt)
#             vz.append(vz[i-1]+(-B2m_*v[i-1]*vz[i-1]-g)*dt)
#             v.append(np.sqrt(vx[i]**2+vz[i]**2))
        
#         alcance=(x[-1]+(x[-2]-x[-1])/2)
#     listv0.append(v0)

# v0min=min(listv0)
# indexmin=listv0.index(v0min)
# theta0=rangetheta[indexmin]

#Comentado para no tener que calcularlo en cada iteración, ya que tarda mucho

theta0=33
v0min=62.05
v0=v0min

print(f"En el caso no ideal, teniendo en cuenta el rozamiento del aire:")
print(f"El ángulo óptimo de lanzamiento en el caso no ideal es {theta0}º, con una v0 necesaria de {v0min:.{2}f} m/s\n")


# %%
#A continuación, vamos a calcular la trayectoria de la bola de béisbol, para distintos ángulos propuestos y la v0 calculada

fig=plt.figure(1)                              #Plot sin efecto Magnus
fig_=plt.figure(2)                             #Plot con efecto Magnus

nm=fig.add_subplot(111,projection='3d')         
ym=fig_.add_subplot(111,projection='3d')

theta=[15,25,theta0,45,55]
alcance = []                              #Listas para guardar los valores de alcance sin Magnus
alcance1, desvio1 = [], []                #Listas para guardar los valores de alcance y desvio en el caso Magnus
alcance2, desvio2 = [], []                #Listas para guardar los valores de alcance y desvio en el caso a 3715 m
alcance3, desvio3 = [], []                #Listas para guardar los valores de alcance y desvio en el caso a 12500 m

for m in range(len(theta)):

    #Calculamos las velocidades en función del ángulo de lanzamiento
    v0x=v0*(np.cos(np.radians(theta[m])))   
    v0y=0.
    v0z=v0*(np.sin(np.radians(theta[m])))

    t,x,y,z,vx,vy,vz,v=[t0],[x0],[y0],[z0],[v0x],[v0y],[v0z],[v0]           #Listas para el caso sin efecto Magnus a la altura del mar
    t1,x1,y1,z1,vx1,vy1,vz1,v1=[t0],[x0],[y0],[z0],[v0x],[v0y],[v0z],[v0]   #Listas para el caso con efecto Magnus a la altura del mar
    t2,x2,y2,z2,vx2,vy2,vz2,v2=[t0],[x0],[y0],[z02],[v0x],[v0y],[v0z],[v0]  #Listas para el caso sin efecto Magnus a 3715 m
    t3,x3,y3,z3,vx3,vy3,vz3,v3=[t0],[x0],[y0],[z03],[v0x],[v0y],[v0z],[v0]  #Listas para el caso sin efecto Magnus a 12500 m

    #Comenzamos con el caso básico: Sin Magnus a nivel del mar
    i=0
    while z[i]>=0:                                  
        i+=1

        #Usando el método de Euler:
        t.append(t[i-1]+dt)
        x.append(x[i-1]+vx[i-1]*dt)
        y.append(y[i-1]+vy[i-1]*dt)
        z.append(z[i-1]+vz[i-1]*dt)

        B2m=0.0039+0.0058/(1+np.exp((v[i-1]-vd)/D))     #Coeficiente de arrastre                                      
        rho= rho_0*(1-(a*z[i])/T0)**alpha               #Densidad del aire a una altura z (modelo de atmósfera estándar)
        B2m_=B2m*rho/rho_0                              #Coeficiente de arrastre corregido

        vx.append(vx[i-1]-B2m_*v[i-1]*vx[i-1]*dt)
        vy.append(vy[i-1]-B2m_*v[i-1]*vy[i-1]*dt)
        vz.append(vz[i-1]+(-B2m_*v[i-1]*vz[i-1]-g)*dt)
        v.append(np.sqrt(vx[i]**2+vy[i]**2+vz[i]**2))

    #Dibujo la trayectoria sin Magnus a la altura del mar
    nm.plot(x, y, z, label=f"y(t) - {theta[m]}°")
    nm.legend(fontsize=14)
    nm.set_xlabel("Eje x (m)",fontsize=16)
    nm.set_ylabel("Eje y (m)",fontsize=16)
    nm.set_zlabel("Eje z (m)",fontsize=16)
    nm.set_title("Trayectoria de la pelota de beisbol sin efecto Magnus al nivel del mar",fontsize=16)

    #Se calcula el alcance
    alcance.append((x[-1]+x[-2])/2)  

    #Segundo caso: sin magnus a 3715 m
    i=0
    while z2[i]>=3715:                                  
        i+=1

        #Usando Euler:
        t2.append(t2[i-1]+dt)
        x2.append(x2[i-1]+vx2[i-1]*dt)
        y2.append(y2[i-1]+vy2[i-1]*dt)
        z2.append(z2[i-1]+vz2[i-1]*dt)

        B2m=0.0039+0.0058/(1+np.exp((v2[i-1]-vd)/D))     #Coeficiente de arrastre                                      
        rho= rho_0*(1-(a*z2[i])/T0)**alpha               #Densidad del aire a una altura z (modelo de atmósfera estándar)
        B2m_=B2m*rho/rho_0                               #Coeficiente de arrastre corregido

        vx2.append(vx2[i-1]-B2m_*v2[i-1]*vx2[i-1]*dt)
        vy2.append(vy2[i-1]-B2m_*v2[i-1]*vy2[i-1]*dt)
        vz2.append(vz2[i-1]+(-B2m_*v2[i-1]*vz2[i-1]-g)*dt)
        v2.append(np.sqrt(vx2[i]**2+vy2[i]**2+vz2[i]**2))

    alcance2.append((x2[-1]+x2[-2])/2)  #Se calcula el alcance

    #Tercer caso: sin magnus a 12500 m
    i=0
    while z3[i]>=12500:                                  
        i+=1
        t3.append(t3[i-1]+dt)
        x3.append(x3[i-1]+vx3[i-1]*dt)
        y3.append(y3[i-1]+vy3[i-1]*dt)
        z3.append(z3[i-1]+vz3[i-1]*dt)

        #Densidad del aire a una altura z (modelo de atmósfera isotérmica)
        B2m=0.0039+0.0058/(1+np.exp((v3[i-1]-vd)/D))                                       
        rho= rho_0*np.exp(-z3[i]/z_0)                      
        B2m_=B2m*rho/rho_0                              

        vx3.append(vx3[i-1]-B2m_*v3[i-1]*vx3[i-1]*dt)
        vy3.append(vy3[i-1]-B2m_*v3[i-1]*vy3[i-1]*dt)
        vz3.append(vz3[i-1]+(-B2m_*v3[i-1]*vz3[i-1]-g)*dt)
        v3.append(np.sqrt(vx3[i]**2+vy3[i]**2+vz3[i]**2))

    alcance3.append((x3[-1]+x3[-2])/2)  #Se calcula el alcance

    #Último caso: Con Magnus a nivel del mar
    i=0
    while z1[i]>=0:
        i+=1

        #Usando el método de Euler:
        t1.append(t1[i-1]+dt)
        x1.append(x1[i-1]+vx1[i-1]*dt)
        y1.append(y1[i-1]+vy1[i-1]*dt)
        z1.append(z1[i-1]+vz1[i-1]*dt)

        #Nos encontramos en el caso de la atmósfera adiabática
        B2m=0.0039+0.0058/(1+np.exp((v1[i-1]-vd)/D))                                          
        rho= rho_0*(1-(a*z1[i])/T0)**alpha               
        B2m_=B2m*rho/rho_0

        vx1.append(vx1[i-1]-B2m_*v1[i-1]*vx1[i-1]*dt)
        vy1.append(vy1[i-1]+(-B2m_*v1[i-1]*vy1[i-1]+S0m*w*vx1[i-1])*dt)
        vz1.append(vz1[i-1]+(-B2m_*v1[i-1]*vz1[i-1]-g)*dt)
        v1.append(np.sqrt(vx[i]**2+vy[i]**2+vz[i]**2))

    #Dibujo la trayectoria con Magnus a nivel del mar
    ym.plot(x1, y1, z1, label=f"y(t) - {theta[m]}°")
    ym.legend(fontsize=14)
    ym.set_xlabel("Eje x (m)",fontsize=16)
    ym.set_ylabel("Eje y (m)",fontsize=16)
    ym.set_zlabel("Eje z (m)",fontsize=16)
    ym.set_title("Trayectoria de la pelota de beisbol con efecto Magnus al nivel del mar",fontsize=16)

    distance = np.sqrt(x1[-1]**2 + y1[-1]**2)        #Último punto de la trayectoria
    distance_1= np.sqrt(x1[-2]**2 + y1[-2]**2)       #Penultimo punto de la trayectoria

    #Se calcula el alcance promediando los dos últimos puntos
    alcance1.append((distance+distance_1)/2)         

#Imprimimos los ángulos con los que se consigue home run
found=False
print("En el caso con rozamiento, sin fuerza de Magnus:")
for m in range(len(theta)):
    if alcance[m]>HR:
        print(f"Con un ángulo de {theta[m]}° se consigue home run")
        found=True
if not found:
    print("No se consigue home run para ningún ángulo")

print("\nEn el caso de rozamiento y fuerza de Magnus en el eje y:")
for m in range(len(theta)):
    if alcance1[m]>HR:
        print(f"Con un ángulo de {theta[m]}° se consigue home run")

print(f"\nPara una velocidad inicial de {v0min:.{2}f} m/s, y un ángulo de {theta0}º:")
print(f"El alcance sin Magnus a nivel del mar es de {alcance[2]} m")
print(f"El alcance con Magnus a nivel del mar es de {alcance1[2]} m")
print(f"El alcance sin Magnus a 3715 m es de {alcance2[2]} m")
print(f"El alcance sin Magnus a 12500 m es de {alcance3[2]} m")

#Finalmente dibujamos el campo de beisbol para hacerlo más visual

field_radius = 168                             # Radio del campo en metros
theta = np.linspace(-pi/4, pi/4, 100)          # Ángulo de apertura de 90º para el campo

# Pasamos polares a cartesianas
field_x = field_radius * np.cos(theta)
field_y = field_radius * np.sin(theta)
field_z = np.zeros_like(field_x)

# Añadir el punto inicial y final para cerrar el campo
field_x = np.concatenate(([0], field_x, [0]))
field_y = np.concatenate(([0], field_y, [0]))
field_z = np.concatenate(([0], field_z, [0]))

#Dibujamos el campo de beisbol
nm.plot(field_x, field_y, field_z, color='green')
ym.plot(field_x, field_y, field_z, color='green')

# Pintar el suelo del campo de verde
nm.plot_trisurf(field_x, field_y, field_z, color='green', alpha=0.5)
ym.plot_trisurf(field_x, field_y, field_z, color='green', alpha=0.5)

# %%
#Podemos ahora ver la trayectoria con la que el pitcher lanza la pelota al lanzador, usando el efecto Magnus

figP=plt.figure(3)
Pplot=figP.add_subplot(111,projection='3d')

#Hay tres casos:
#Overhand curveball, Sidearm curveball, fastball

x0,y0,z0=0,0,1.5  #Posición de lanzamiento de la bola
distancia=20  # Distancia del pitcher al golpeador
theta0M= 0.5  # Ángulo de lanzamiento
v0M= 90  # Velocidad de lanzamiento

#Lanzamiento sin rotación
v0x=v0M*(np.cos(np.radians(theta0M)))   
v0y=0.
v0z=v0M*(np.sin(np.radians(theta0M)))
t,x,y,z,vx,vy,vz,v=[t0],[x0],[y0],[z0],[v0x],[v0y],[v0z],[v0M]
i=0
while x[i]<distancia:                                  
    i+=1

    #Usando el método de Euler:
    t.append(t[i-1]+dt)
    x.append(x[i-1]+vx[i-1]*dt)
    y.append(y[i-1]+vy[i-1]*dt)
    z.append(z[i-1]+vz[i-1]*dt)

    B2m=0.0039+0.0058/(1+np.exp((v[i-1]-vd)/D))     #Coeficiente de arrastre                                      
    rho= rho_0*(1-(a*z[i])/T0)**alpha               #Densidad del aire a una altura z (modelo de atmósfera estándar)
    B2m_=B2m*rho/rho_0                              #Coeficiente de arrastre corregido

    vx.append(vx[i-1]-B2m_*v[i-1]*vx[i-1]*dt)
    vy.append(vy[i-1]-B2m_*v[i-1]*vy[i-1]*dt)
    vz.append(vz[i-1]+(-B2m_*v[i-1]*vz[i-1]-g)*dt)
    v.append(np.sqrt(vx[i]**2+vy[i]**2+vz[i]**2))


Pplot.plot(x,y,z,label=f'Sin rotación')

#Overhand curveball
v0x=v0M*(np.cos(np.radians(theta0M)))   
v0y=0.
v0z=v0M*(np.sin(np.radians(theta0M)))
t,x,y,z,vx,vy,vz,v=[t0],[x0],[y0],[z0],[v0x],[v0y],[v0z],[v0M]
i=0
while x[i]<distancia:                                  
    i+=1

    #Usando el método de Euler:
    t.append(t[i-1]+dt)
    x.append(x[i-1]+vx[i-1]*dt)
    y.append(y[i-1]+vy[i-1]*dt)
    z.append(z[i-1]+vz[i-1]*dt)
    
    #Modelo de atmósfera adiabática
    B2m=0.0039+0.0058/(1+np.exp((v[i-1]-vd)/D))                                       
    rho= rho_0*(1-(a*z[i])/T0)**alpha               
    B2m_=B2m*rho/rho_0                              

    vx.append(vx[i-1]-B2m_*v[i-1]*vx[i-1]*dt)
    vy.append(vy[i-1]-B2m_*v[i-1]*vy[i-1]*dt)
    vz.append(vz[i-1]+(-B2m_*v[i-1]*vz[i-1]-g-S0m*w*vx1[i-1])*dt)
    v.append(np.sqrt(vx[i]**2+vy[i]**2+vz[i]**2))

Pplot.plot(x,y,z,label=f'Overhand curveball')

#Fastball
v0x=v0M*(np.cos(np.radians(theta0M)))   
v0y=0.
v0z=v0M*(np.sin(np.radians(theta0M)))
t,x,y,z,vx,vy,vz,v=[t0],[x0],[y0],[z0],[v0x],[v0y],[v0z],[v0M]
i=0
while x[i]<distancia:                                  
    i+=1

    #Usando el método de Euler:
    t.append(t[i-1]+dt)
    x.append(x[i-1]+vx[i-1]*dt)
    y.append(y[i-1]+vy[i-1]*dt)
    z.append(z[i-1]+vz[i-1]*dt)

    B2m=0.0039+0.0058/(1+np.exp((v[i-1]-vd)/D))     #Coeficiente de arrastre                                      
    rho= rho_0*(1-(a*z[i])/T0)**alpha               #Densidad del aire a una altura z (modelo de atmósfera estándar)
    B2m_=B2m*rho/rho_0                              #Coeficiente de arrastre corregido

    vx.append(vx[i-1]-B2m_*v[i-1]*vx[i-1]*dt)
    vy.append(vy[i-1]-B2m_*v[i-1]*vy[i-1]*dt)
    vz.append(vz[i-1]+(-B2m_*v[i-1]*vz[i-1]-g+S0m*w*vx1[i-1])*dt)
    v.append(np.sqrt(vx[i]**2+vy[i]**2+vz[i]**2))


Pplot.plot(x,y,z,label=f'Fastball')

#Sidearm curveball
v0x=v0M*(np.cos(np.radians(theta0M)))   
v0y=0.
v0z=v0M*(np.sin(np.radians(theta0M)))
t,x,y,z,vx,vy,vz,v=[t0],[x0],[y0],[z0],[v0x],[v0y],[v0z],[v0M]
i=0
while x[i]<distancia:                                  
    i+=1

    #Usando el método de Euler:
    t.append(t[i-1]+dt)
    x.append(x[i-1]+vx[i-1]*dt)
    y.append(y[i-1]+vy[i-1]*dt)
    z.append(z[i-1]+vz[i-1]*dt)

    B2m=0.0039+0.0058/(1+np.exp((v[i-1]-vd)/D))     #Coeficiente de arrastre                                      
    rho= rho_0*(1-(a*z[i])/T0)**alpha               #Densidad del aire a una altura z (modelo de atmósfera estándar)
    B2m_=B2m*rho/rho_0                              #Coeficiente de arrastre corregido

    vx.append(vx[i-1]-B2m_*v[i-1]*vx[i-1]*dt)
    vy.append(vy[i-1]+(-B2m_*v[i-1]*vy[i-1]-S0m*w*vx1[i-1])*dt)
    vz.append(vz[i-1]+(-B2m_*v[i-1]*vz[i-1]-g)*dt)
    v.append(np.sqrt(vx[i]**2+vy[i]**2+vz[i]**2))


Pplot.plot(x,y,z,label=f'Sidearm Curveball')

# Dimensiones aproximadas de la strike zone (en metros)
bajo_strike_zone = 1.0
alto_strike_zone = 1.6  
ancho_strike_zone = 0.5  

xs = [distancia,distancia,distancia,distancia]
ys = [ancho_strike_zone/2, -ancho_strike_zone/2, ancho_strike_zone/2, -ancho_strike_zone/2]
zs = [bajo_strike_zone, bajo_strike_zone, alto_strike_zone, alto_strike_zone]

# Convertir a una malla 2D (necesario para plot_surface)
X = np.array([xs[:2], xs[2:]]) 
Y = np.array([ys[:2], ys[2:]])  
Z = np.array([zs[:2], zs[2:]])  

# Dibuja la superficie de la diana y la pelota
Pplot.plot_surface(X,Y,Z, color='red',edgecolors='black',alpha=0.25)
Pplot.scatter(x0, y0, z0, color='white',edgecolors='black', s=100)

# Títulos y etiquetas
Pplot.set_title("Trayectoria de la pelota con distintos tipos de lanzamiento del pitcher",fontsize=16)
Pplot.set_xlabel("Distancia (m)",fontsize=16)  # Eje X es la distancia al lanzador
Pplot.set_ylabel("Ancho (m)",fontsize=16)      # Eje Y es el ancho de la zona de strike
Pplot.set_zlabel("Altura (m)",fontsize=16)     # Eje Z es la altura de la zona de strike
Pplot.set_xlim([0.,20.])
Pplot.set_ylim([-0.5,0.5])
Pplot.set_zlim([0.5,2.])
Pplot.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Máximo 5 marcas en el eje X

Pplot.legend(fontsize=14)
plt.show()

# %%
