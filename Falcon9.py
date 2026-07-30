import numpy as np
import scienceplots 
import matplotlib.pyplot as plt


# rocket variables
dry_mass1 = 25600
dry_mass2 = 4000      
propellent_mass1 = 395700
propellent_mass2 = 92670    
m0 =  517970
A_e = 10.8 #rocket cross sectional area
SL_thrust = 7607000 
V_thrust = 8227000 
SL_Isp = 282 
V_Isp = 311 
P2_Isp = 348
Merlin_thrust = 934000

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
t_vertical = 10   # Time for straight vertical flight
kick_duration = 5
kick_angle = np.deg2rad(5)
height = 0
x = 0
v = 0
current_mass = m0
gamma = np.deg2rad(90)
current_stage = 1



t_history = [t_current]
height_history = [height]
x_history = [x]
v_history = [v]
mass_history = [current_mass]
gamma_history = [gamma]
S = (height, x, v, current_mass, gamma)



def get_gravity(height):
    gravity = g0 * ((RE)**2)/(RE + height)**2
    return(gravity)

def get_air_density(height):
    if 0 <= height < 11000:
        Temp =  t0 - (L * height)
        pressure = P0 * (Temp/t0)**((g0 * M_air)/(R * L))
    elif 11000 <= height < 20000:
        Temp = 216.65
        P_11km = 22632
        pressure = P_11km * np.e**( - g0 * M_air * (height - 11000) / (R * Temp))
    elif 20000 <= height < 32000:
        T_20km = 216.65
        P_20km = 5475 
        Temp = T_20km + L3 * (height - 20000)
        pressure = P_20km * (T_20km / Temp)**(g0 * M_air / (R *L3))
    else:
        Temp = 228.65
        P_32km = 868 
        pressure = P_32km * np.e**(-(height - 32000) / 7000)
    density =  pressure / (R_specific * Temp)
    return(density,pressure)

def get_thrust(height,mass,current_stage):
    if (dry_mass1 + dry_mass2 + propellent_mass2) <= mass <= (dry_mass1 + dry_mass2 + propellent_mass1 + propellent_mass2):
        density, pressure = get_air_density(height)
        thrust =  V_thrust - (V_thrust - SL_thrust) * (pressure / P0)
        Isp = V_Isp - (V_Isp - SL_Isp) * (pressure / P0)
        v_e = Isp * g0
        mass_flow_rate = thrust/v_e
    elif current_stage == 2:
        density, pressure = get_air_density(height)
        thrust =  Merlin_thrust 
        Isp = P2_Isp
        v_e = Isp * g0
        mass_flow_rate = thrust/v_e
    else:
        thrust = 0
        mass_flow_rate = 0
    return(thrust, mass_flow_rate)

def kick_phase(t):   
    if t_vertical <= t < (t_vertical + kick_duration) :
        theta = kick_angle* (t - t_vertical)/(kick_duration)
    else: 
        theta = 0
    return(theta)


def derivative(S, t):
    height = S[0]
    x = S[1]
    v = S[2]
    mass = S[3]
    gamma = S[4]
    density, pressure = get_air_density(height)
    theta = kick_phase(t)
    drag = 0.5 * density * (v**2) * A_e * Cd
    thrust, mass_flow_rate = get_thrust(height,mass, current_stage)
    gravity = get_gravity(height) * mass
    if t < t_vertical:
        commanded_thrust_angle = np.deg2rad(90)
    elif  t_vertical <= t < (t_vertical + kick_duration):
        commanded_thrust_angle = np.deg2rad(90) - theta
    else:
        commanded_thrust_angle = gamma
    mismatch = commanded_thrust_angle - gamma
    F = (thrust * np.cos(mismatch)) - drag - (gravity * np.sin(gamma))
    a = F/mass
    dhdt = v * np.sin(gamma)
    dxdt = v * np.cos(gamma)
    dvdt = a
    dmdt = - mass_flow_rate
    if t < t_vertical:
        dgdt = 0
    elif  t_vertical <= t < (t_vertical + kick_duration):
        dgdt = ((thrust * np.sin(mismatch)) - (gravity * np.cos(gamma)))/(v * mass)
    else:
        dgdt = ((thrust * np.sin(mismatch)) - (gravity * np.cos(gamma)))/(v * mass)
    return(dhdt, dxdt, dvdt, dmdt, dgdt)

seperated = False 
while current_mass > dry_mass2 and t_current < 600:
    k1h, k1x, k1v, k1m, k1g = derivative(S ,t_current)
    S_temp = (height + k1h * dt/2, x + k1x * dt/2, v + k1v * dt/2, current_mass + k1m * dt/2, gamma +k1g * dt/2)
    k2h, k2x, k2v, k2m, k2g = derivative(S_temp, t_current + dt/2)
    S_temp2 = (height + k2h * dt/2, x + k2x * dt/2, v + k2v * dt/2, current_mass + k2m * dt/2, gamma + k2g * dt/2)
    k3h, k3x, k3v, k3m, k3g= derivative(S_temp2, t_current+ dt/2)
    S_temp3 = (height + k3h * dt, x + k3x * dt, v + k3v * dt, current_mass + k3m * dt, gamma + k3g * dt)
    k4h, k4x, k4v, k4m, k4g = derivative(S_temp3, t_current + dt)
    h_new = height + (dt/6) * (k1h + 2*k2h + 2*k3h + k4h)
    x_new = x + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt/6) * (k1v + 2*k2v + 2*k3v+ k4v)
    m_new = current_mass + (dt/6) * (k1m + 2*k2m + 2*k3m + k4m)
    gamma_new = gamma + (dt/6) * (k1g + 2*k2g + 2*k3g + k4g)

    if m_new < dry_mass2:
        m_new = dry_mass2

    height = h_new
    x = x_new
    v = v_new
    current_mass = m_new
    gamma = gamma_new
    S = (height, x, v, current_mass, gamma)
    t_current = t_current + dt

    t_history.append(t_current)
    height_history.append(height)
    x_history.append(x)
    v_history.append(v)
    mass_history.append(current_mass)
    gamma_history.append(gamma)

    if current_stage == 1 and current_mass <= dry_mass1 + dry_mass2 + propellent_mass2:
        current_mass -= dry_mass1
        current_stage = 2
    S = (height, x, v, current_mass, gamma)

plt.style.use(['science', 'notebook', 'grid'])
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

t_array = np.array(t_history)
height_array = np.array(height_history)
x_array = np.array(x_history)
v_array = np.array(v_history)
mass_array = np.array(mass_history)
gamma_array = np.array(gamma_history)

maxh = np.argmax(height_array)
xmax = np.argmax(x_array)
maxv = np.argmax(v_array)
minm = np.argmin(mass_array)
ming = np.argmin(gamma_array)



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

axs[2, 0].plot(t_array, x_array)
axs[2, 0].set_title("Range")
axs[2,0].scatter(t_array[xmax], x_array[xmax], color = "red")
axs[2,0].annotate(
    f"Max Range: \n ({t_array[xmax]:.2f}, {x_array[xmax]:.2f})",
    (t_array[xmax], x_array[xmax]),
    )

axs[2, 1].plot(t_array, gamma_array)
axs[2, 1].set_title("gamma")
axs[2,1].scatter(t_array[ming], gamma_array[ming], color = "red")
axs[2,1].annotate(
    f"Max Range: \n ({t_array[ming]:.2f}, {gamma_array[ming]:.2f})",
    (t_array[ming], gamma_array[ming]),
)


plt.tight_layout()
plt.show()



   


