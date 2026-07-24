import numpy as np
import scienceplots 
import matplotlib.pyplot as plt


# rocket variables
dry_mass = 25600   #kg
propellent_mass = 395700
m0 =  421300
A_e = 10.8 #rocket cross sectional area
SL_thrust = 7607000 
V_thrust = 8227000 
SL_Isp = 282 
V_Isp = 311 

# natural variables 
RE = 6371000 # Radius of the earth 
g0 = 9.80665 # gravitational acceleration at sea level
t0 = 288.15  #Temp at sea level  kelvin
L = 0.0065   #lapse rate kelvin per metre
L3 = 0.001 #lapse rate in stratosphere subsection 3
P0 = 101325   #pressure at sea level Pascal
p0 = 1.225 # sea level air density
M_air = 0.0289644  #molar mass of air kg/mol
R = 8.31447  #universal gas constant 
R_specific = R / M_air  #specific gas constant
Cd = 0.3 #drag coefficient 

#state variables
dt = 0.5
t_current = 0
current_height = 0
current_v = 0
current_mass = m0
t_history = [t_current]
height_history = [current_height]
v_history = [current_v]
mass_history = [current_mass]
S = (current_height, current_v, current_mass)



def get_gravity(current_height):
    gravity = g0 * ((RE)**2)/(RE + current_height)**2
    return(gravity)

def get_air_density(current_height):
    if 0 <= current_height < 11000:
        Temp =  t0 - (L * current_height)
        pressure = P0 * (Temp/t0)**((g0 * M_air)/(R * L))
    elif 11000 <= current_height < 20000:
        Temp = 216.65
        P_11km = 22632
        pressure = P_11km * np.e**( - g0 * M_air * (current_height - 11000) / (R * Temp))
    elif 20000 <= current_height < 32000:
        T_20km = 216.65
        P_20km = 5475 
        Temp = T_20km + L3 * (current_height - 20000)
        pressure = P_20km * (T_20km / Temp)**(g0 * M_air / (R *L3))
    else:
        Temp = 228.65
        P_32km = 868 
        pressure = P_32km * np.e**(-(current_height - 32000) / 7000)
    density =  pressure / (R_specific * Temp)
    return(density,pressure)

def get_thrust(height,mass):
    if mass > dry_mass:
        density, pressure = get_air_density(height)
        thrust =  V_thrust - (V_thrust - SL_thrust) * (pressure / P0)
        Isp = V_Isp - (V_Isp - SL_Isp) * (pressure / P0)
        v_e = Isp * g0
        mass_flow_rate = thrust/v_e
    elif mass <= dry_mass:
        thrust = 0
        mass_flow_rate = 0
    return(thrust, mass_flow_rate)


def derivative(S, t):
    height = S[0]
    velocity = S[1]
    mass = S[2]
    density, pressure = get_air_density(height)
    drag = 0.5 * density * (velocity**2) * A_e * Cd
    thrust, mass_flow_rate = get_thrust(height,mass)
    gravity = get_gravity(height) * mass
    net_force = thrust - (gravity + drag)
    acceleration = net_force/mass
    dhdt = S[1]
    dvdt = acceleration
    dmdt = - mass_flow_rate
    return(dhdt, dvdt, dmdt)

while current_mass > dry_mass and t_current < 300:
    k1h, k1v, k1m = derivative(S ,t_current)
    S_temp = (current_height + k1h * dt/2, current_v + k1v * dt/2, current_mass + k1m * dt/2)
    k2h, k2v, k2m = derivative(S_temp, t_current + dt/2)
    S_temp2 = (current_height + k2h * dt/2, current_v + k2v * dt/2, current_mass + k2m * dt/2)
    k3h,k3v,k3m = derivative(S_temp2, t_current+ dt/2)
    S_temp3 = (current_height + k3h * dt, current_v + k3v * dt, current_mass + k3m * dt)
    k4h, k4v, k4m = derivative(S_temp3, t_current + dt)

    h_new = current_height + (dt/6) * (k1h + 2*k2h + 2*k3h + k4h)
    v_new = current_v + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
    m_new = current_mass + (dt/6) * (k1m + 2*k2m + 2*k3m + k4m)

    if m_new < dry_mass:
        m_new = dry_mass

    current_height = h_new
    current_v = v_new
    current_mass = m_new
    S = (current_height, current_v, current_mass)
    t_current = t_current + dt

    t_history.append(t_current)
    height_history.append(current_height)
    v_history.append(current_v)
    mass_history.append(current_mass)

plt.style.use(['science', 'notebook', 'grid'])
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

t_array = np.array(t_history)
height_array = np.array(height_history)
v_array = np.array(v_history)
mass_array = np.array(mass_history)

maxh = np.argmax(height_array)
minm = np.argmin(mass_array)
maxv = np.argmax(v_array)

axs[0, 0].plot(t_array, mass_array)
axs[0, 0].set_title("Mass")
axs[0,0].scatter(t_array[minm], mass_array[minm], color = "red")
axs[0,0].annotate(
    f"Fuel Over: \n ({t_array[minm]:.2f}, {mass_array[minm]:.2f})",
    (t_array[minm], mass_array[minm]),
    )

axs[0, 1].plot(t_array, height_array)
axs[0, 1].set_title("Height")
axs[0,1].scatter(t_array[maxh], height_array[maxh], color = "red")
axs[0,1].annotate(
    f"Max height: \n ({t_array[maxh]:.2f}, {height_array[maxh]:.2f})",
    (t_array[maxh], height_array[maxh]),
    )

axs[1, 0].plot(t_array, v_array)
axs[1, 0].set_title("Velocity")
axs[1,0].scatter(t_array[maxv], v_array[maxv], color = "red")
axs[1,0].annotate(
    f"Max Velocity: \n ({t_array[maxv]:.2f}, {v_array[maxv]:.2f})",
    (t_array[maxv], v_array[maxv]),
    )

axs[1, 1].axis('off')


plt.tight_layout()
plt.show()


   


