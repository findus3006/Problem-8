import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def RungeKutta(t,interval,func,initial):
    dt = abs(t[2]-t[1]);
    df = np.zeros([np.size(initial),np.size(t)],'complex');
    df[:,0] = initial;
    for i in range(np.size(t)-1):
        k1 = func(df[:,i],interval)
        k2 = func(df[:,i]+dt/2*k1,interval)
        k3 = func(df[:,i]+dt*k2/2,interval)
        k4 = func(df[:,i]+dt*k3,interval)
        df[:,i+1] = df[:,i] + 1/6*(k1+2*k2+2*k3+k4)*dt
    return df

def FourierDerivative(func,interval,ndif):
    k0 = 2*np.pi/(interval[1]-interval[0])
    L = int(np.size(func)/2)
    m = np.concatenate([np.linspace(0,L-1,L),np.linspace(-L,-1,L)])
    for i in range(ndif):
        func = np.fft.ifft(1j*m*k0*np.fft.fft(func))
        i = i+1
    return func

def Hamiltonian(func,interval):
    x = np.linspace(interval[0],interval[1], N+1)
    x = x[:-1]
    return -1j*0.5*(-FourierDerivative(func,interval,2)+x**2*func)


L = 20; # Length of interval
N = 100; # Points of discretization
tmax = 4*np.pi; # endtime
dt = 1/N; # Time step

x = np.linspace(-L/2,L/2, N+1)
x = x[:-1]

t = np.arange(0,tmax,dt) # time grid
t = t[:-1]

# Initialize
initial = np.exp(-0.25*(x**2))

#df = Hamiltonian(initial,[0,tmax-dt],x)
initialHamiltonian = Hamiltonian(initial,[-L/2,L/2])
data = RungeKutta(t,[-L/2,L/2],Hamiltonian,initialHamiltonian)

x,t = np.meshgrid(x,t)
fig = plt.figure();
ax = fig.add_subplot(1,1,1)
colormesh = ax.pcolormesh(x,t,np.log10(abs(np.transpose(data))**2),shading='auto',cmap='jet')
fig.colorbar(colormesh,ax=ax)
#ax.plot(x,initialHamiltonian)
plt.show()

# fig = plt.figure();
# ax = fig.add_subplot()

# def update(i):
#     ax.clear()
#     ax.plot(x,data[:,i])
#     ax.set_xlim(-L/2, L/2)
#     ax.set_ylim(-1, 1)
#     ax.set_xlabel('x')
#     ax.text(0.35,0.8,f't = {round(t[i],2)}',fontsize='12')
#     ax.grid()

# ani = animation.FuncAnimation(fig, update,frames = np.arange(0, len(t), 10), interval = 33.3)
# plt.show()