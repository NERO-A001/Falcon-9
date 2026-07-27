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
t_vertical = 10   # Time for straight vertical flight
t_pitch_end = 45  # Time between this and vertical is the duration of pitch over
theta = 0
theta_target = np.deg2rad(45)
height = 0
x = 0
vy = 0
vx = 0
current_mass = m0
t_history = [t_current]
height_history = [height]
x_history = [x]
vy_history = [vy]
vx_history = [vx]
mass_history = [current_mass]
S = (height, x, vy, vx, current_mass)



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

def get_pitch_angle(t):   # angle is increasing linearly per second during the duration
    if t_vertical <= t < t_pitch_end :
        theta = theta_target * (t - t_vertical)/(t_pitch_end - t_vertical)
    elif t >= t_pitch_end:
        theta = theta_target    
    else:
        theta = 0
    return(theta)


def derivative(S, t):
    height = S[0]
    x = S[1]
    vy = S[2]
    vx = S[3]
    mass = S[4]
    v = np.sqrt((vy**2) + (vx**2))
    density, pressure = get_air_density(height)
    theta = get_pitch_angle(t)
    drag = 0.5 * density * (v**2) * A_e * Cd
    if v == 0:
        Dy = 0
        Dx = 0
    else:
        Dy = -drag * (vy/v)
        Dx = -drag * (vx/v)
    thrust, mass_flow_rate = get_thrust(height,mass)
    Tx = thrust * np.sin(theta) 
    Ty = thrust * np.cos(theta)
    gravity = get_gravity(height) * mass
    Fy = Ty - (gravity + Dy)
    Fx = Tx - Dx
    ay = Fy/mass
    ax = Fx/mass
    dhdt = S[2]
    dxdt = S[3]
    dvydt = ay
    dvxdt = ax
    dmdt = - mass_flow_rate
    return(dhdt, dxdt, dvydt, dvxdt, dmdt)

while current_mass > dry_mass and t_current < 300:
    k1h, k1x, k1vy, k1vx, k1m = derivative(S ,t_current)
    S_temp = (height + k1h * dt/2, x + k1x * dt/2, vy + k1vy * dt/2, vx + k1vx * dt/2, current_mass + k1m * dt/2)
    k2h, k2x, k2vy, k2vx, k2m = derivative(S_temp, t_current + dt/2)
    S_temp2 = (height + k2h * dt/2, x + k2x * dt/2, vy + k2vy * dt/2,vx + k2vx * dt/2, current_mass + k2m * dt/2)
    k3h, k3x, k3vy, k3vx, k3m = derivative(S_temp2, t_current+ dt/2)
    S_temp3 = (height + k3h * dt, x + k3x * dt, vy + k3vy * dt, vx + k3vx * dt, current_mass + k3m * dt)
    k4h, k4x, k4vy, k4vx, k4m = derivative(S_temp3, t_current + dt)

    h_new = height + (dt/6) * (k1h + 2*k2h + 2*k3h + k4h)
    x_new = x + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
    vy_new = vy + (dt/6) * (k1vy + 2*k2vy + 2*k3vy+ k4vy)
    vx_new = vx + (dt/6) * (k1vx + 2*k2vx + 2*k3vx+ k4vx)
    m_new = current_mass + (dt/6) * (k1m + 2*k2m + 2*k3m + k4m)

    if m_new < dry_mass:
        m_new = dry_mass

    height = h_new
    x = x_new
    vy = vy_new
    vx = vx_new
    current_mass = m_new
    S = (height, x, vy, vx, current_mass)
    t_current = t_current + dt

    t_history.append(t_current)
    height_history.append(height)
    x_history.append(x)
    vy_history.append(vy)
    vx_history.append(vx)
    mass_history.append(current_mass)

plt.style.use(['science', 'notebook', 'grid'])
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

t_array = np.array(t_history)
height_array = np.array(height_history)
x_array = np.array(x_history)
vy_array = np.array(vy_history)
vx_array = np.array(vx_history)
mass_array = np.array(mass_history)

maxh = np.argmax(height_array)
minm = np.argmin(mass_array)
maxvy = np.argmax(vy_array)
maxvx = np.argmax(vx_array)
xmax = np.argmax(x_array)

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

axs[1, 0].plot(t_array, vy_array)
axs[1, 0].set_title("Vertical Velocity")
axs[1,0].scatter(t_array[maxvy], vy_array[maxvy], color = "red")
axs[1,0].annotate(
    f"Max Velocity: \n ({t_array[maxvy]:.2f}, {vy_array[maxvy]:.2f})",
    (t_array[maxvy], vy_array[maxvy]),
    )

axs[1, 1].plot(vx_array, vy_array)
axs[1, 1].set_title("Velocity")
axs[1,1].scatter(vx_array[maxvy], vy_array[maxvy], color = "red")
axs[1,1].annotate(
    f"Y max: \n ({vx_array[maxvy]:.2f}, {vy_array[maxvy]:.2f})",
    (vx_array[maxvy], vy_array[maxvy]),
    )
axs[1,1].scatter(vx_array[maxvx], vy_array[maxvx], color = "red")
axs[1,1].annotate(
    f"X max: \n ({vx_array[maxvx]:.2f}, {vy_array[maxvx]:.2f})",
    (vx_array[maxvx], vy_array[maxvx]),
    )

axs[2, 0].plot(t_array, x_array)
axs[2, 0].set_title("Range")
axs[2,0].scatter(t_array[xmax], x_array[xmax], color = "red")
axs[2,0].annotate(
    f"Max Range: \n ({t_array[xmax]:.2f}, {x_array[xmax]:.2f})",
    (t_array[xmax], x_array[xmax]),
    )

axs[2,1].axis('off')


plt.tight_layout()
plt.show()



   


