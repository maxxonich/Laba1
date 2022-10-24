# %%
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

L = 10
# %%
T = L
# n = 50
# t = np.linspace(0, T, n)
step=0.1
t = np.linspace(0, T, round(T/step))
x0_arr = [[0, 1], [1, 1], [2, 5]]

epsilons = [0.5, 0.1, 0.01, 0.001, 0.0001]


def x1Eps0(t1,X0):
    c1=X0[0]
    c2=X0[1]
    return c1 * np.cos(t1) + c2 * np.sin(t1)

def x2Eps0(t1,X0):
    c1=X0[0]
    c2=X0[1]
    return -c1 * np.sin(t1) + c2 * np.cos(t1)


# %%
def portrait_eps0(X0):
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot(x1Eps0(t,X0), x2Eps0(t,X0))
    plt.title("x1(0)=" + str(X0[0]) + "\nx2(0)=" + str(X0[1]))
    plt.show()


    plt.plot(t, x1Eps0(t,X0))
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("x1(0)=" + str(X0[0]) + "\nx2(0)=" + str(X0[1]))
    plt.show()

for X0 in x0_arr:
    portrait_eps0(X0)

# %%
fig, ax = plt.subplots()

X1_eps_arr_fin = []
X2_eps_arr_fin = []
X1_eps0_arr_fin = []
X2_eps0_arr_fin = []
t_fin=[]

X1_eps_arr_asymt = []
X2_eps_arr_asymt = []
X1_eps0_arr_asymt = []
X2_eps0_arr_asymt = []
t_asymt=[]

#
# def get_numerical_sol_eps0(t1, T1, X0):
#     def f(t_tmp, X):
#         return (X[1], -X[0])
#
#     return solve_ivp(f, (0, T1), X0, t_eval=t1).y


# %%

def plot_phase_portrait(frame, asympt, X0):
    eps = epsilons[frame]
    ax.clear()
    T1 = 0
    if asympt:
        T1 = L / eps
    else:
        T1 = L

    def f(t1, X):
        return (X[1], -4*eps*(np.cos(t1)**2) - X[0])

    # t1 = np.linspace(0, T1, n)
    t1 = np.linspace(0, T1, round(T1 / step))

    x1_eps, x2_eps = solve_ivp(f, (0, T1), X0, t_eval=t1).y
    x1_eps0 = x1Eps0(t1,X0)
    x2_eps0= x2Eps0(t1,X0)

    if asympt:
        X1_eps_arr_asymt.append(x1_eps)
        X2_eps_arr_asymt.append(x2_eps)
        X1_eps0_arr_asymt.append(x1_eps0)
        X2_eps0_arr_asymt.append(x2_eps0)
        if len(t_asymt)!=len(epsilons)+1:
            t_asymt.append(t1)
    else:
        X1_eps_arr_fin.append(x1_eps)
        X2_eps_arr_fin.append(x2_eps)
        X1_eps0_arr_fin.append(x1_eps0)
        X2_eps0_arr_fin.append(x2_eps0)
        global t_fin
        if len(t_fin)==0:
            t_fin=t1

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.plot(x1_eps, x2_eps, label="eps=" + str(eps))
    ax.plot(x1_eps[0], x2_eps[0], '.')
    ax.plot(x1_eps0, x2_eps0, label="eps=0")

    ax.set_title("x1(0)=" + str(X0[0]) + "\nx2(0)=" + str(X0[1]))
    ax.legend()



# %%
for X0 in x0_arr:
    anim = FuncAnimation(fig=fig, frames=len(epsilons), func=plot_phase_portrait, interval=3000, fargs=(False, X0))
    anim.save("finite_interval_plot_X1X2_with_X0_" + str(X0[0]) + str(X0[1]) + ".gif")

    anim = FuncAnimation(fig=fig, frames=len(epsilons), func=plot_phase_portrait, interval=3000, fargs=(True, X0))
    anim.save("asymptotic_interval_plot_X1X2_with_X0_" + str(X0[0]) + str(X0[1]) + ".gif")
#%%
def plot_Xt(frame,asympt,label_x,X0,i1):
    eps = epsilons[frame]
    x=0
    x_eps0=0
    t1=0

    I = i1 * len(epsilons) + frame + 1 + i1
    if asympt:
        t1=t_asymt[I%(len(epsilons)+1)]
        if label_x=="X1":
            x=X1_eps_arr_asymt[I]
            x_eps0=X1_eps0_arr_asymt[I]
        elif label_x=="X2":
            x= X2_eps_arr_asymt[I]
            x_eps0=X2_eps0_arr_asymt[I]
    else:
        t1=t_fin
        if label_x=="X1":
            x=X1_eps_arr_fin[I]
            x_eps0=X1_eps0_arr_fin[I]
        elif label_x=="X2":
            x= X2_eps_arr_fin[I]
            x_eps0=X2_eps0_arr_fin[I]


    ax.clear()
    ax.set_ylabel(label_x)
    ax.set_xlabel("t")
    ax.plot(t1, x, label="eps=" + str(eps))
    ax.plot(t1[0], x[0], '.')
    ax.plot(t1,x_eps0 , label="eps=0")
    ax.set_title("x1(0)=" + str(X0[0]) + "\nx2(0)=" + str(X0[1]))
    ax.legend()

# %%

for i in range(len(x0_arr)):
    anim = FuncAnimation(fig=fig, frames=len(epsilons), func=plot_Xt, interval=3000, fargs=(True,"X1", x0_arr[i],i))
    anim.save("asymptotic_interval_plot_X1t_with_X0_" + str(x0_arr[i][0]) + str(x0_arr[i][1]) + ".gif")

    anim = FuncAnimation(fig=fig, frames=len(epsilons), func=plot_Xt, interval=3000, fargs=(False,"X1", x0_arr[i],i))
    anim.save("finite_interval_plot_X1t_with_X0_" + str(x0_arr[i][0]) + str(x0_arr[i][1]) + ".gif")

    anim = FuncAnimation(fig=fig, frames=len(epsilons), func=plot_Xt, interval=3000, fargs=(True,"X2", x0_arr[i],i))
    anim.save("asymptotic_interval_plot_X2t_with_X0_" + str(x0_arr[i][0]) + str(x0_arr[i][1]) + ".gif")

    anim = FuncAnimation(fig=fig, frames=len(epsilons), func=plot_Xt, interval=3000, fargs=(False,"X2", x0_arr[i],i))
    anim.save("finite_interval_plot_X2t_with_X0_" + str(x0_arr[i][0]) + str(x0_arr[i][1]) + ".gif")
# %%
# X1_eps_arr_fin = X1_eps_arr_fin[1:]
# X2_eps_arr_fin = X2_eps_arr_fin[1:]
# X1_eps0_arr_fin = X1_eps0_arr_fin[1:]
# X2_eps0_arr_fin = X2_eps0_arr_fin[1:]
#
# X1_eps_arr_asymt = X1_eps_arr_asymt[1:]
# X2_eps_arr_asymt = X2_eps_arr_asymt[1:]
# X1_eps0_arr_asymt = X1_eps0_arr_asymt[1:]
# X2_eps0_arr_asymt = X2_eps0_arr_asymt[1:]

for i in range(len(x0_arr)):
    print("------------------------\n")
    print("x1(0)=" + str(x0_arr[i][0]) + "; x2(0)=" + str(x0_arr[i][1]))

    print("Distance on the finite interval")
    for j in range(len(epsilons)):
        # the №0 frame is the same as the №1 in for each X0 in x0_arr, it isn't needed, that's why +1+i
        # It's like jumping over it
        I = i * len(epsilons) + j+1+i
        print("eps=" + str(epsilons[j]) + ": ",
              np.sqrt(
                  ((X1_eps_arr_fin[I] - X1_eps0_arr_fin[I]) *
                   (X1_eps_arr_fin[I] - X1_eps0_arr_fin[I]) +
                   (X2_eps_arr_fin[I] - X2_eps0_arr_fin[I])
                   * (X2_eps_arr_fin[I] - X2_eps0_arr_fin[I])
                   ).max())
              )

    print("\nDistance on the asymptotic interval")
    for j in range(len(epsilons)):
        I = i * len(epsilons) + j+1+i
        print("eps=" + str(epsilons[j]) + ": ",
              np.sqrt(
                  ((X1_eps_arr_asymt[I] - X1_eps0_arr_asymt[I]) *
                   (X1_eps_arr_asymt[I] - X1_eps0_arr_asymt[I]) +
                   (X2_eps_arr_asymt[I] - X2_eps0_arr_asymt[I]) *
                   (X2_eps_arr_asymt[I] - X2_eps0_arr_asymt[I])
                   ).max())
              )