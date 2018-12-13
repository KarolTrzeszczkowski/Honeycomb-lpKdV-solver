from decimal import (
    Decimal,
    getcontext
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from copy import deepcopy

getcontext().prec = 1000
c, c1, c2, D, lamb = Decimal('0'), Decimal('1'), Decimal('1'), Decimal('0'), Decimal('1')
p, q, r = Decimal('1.1'), Decimal('1.2'), Decimal('1.3')

ap = p + lamb
am = p - lamb
bp = q + lamb
bm = q - lamb
gp = r + lamb
gm = r - lamb
LEN =20


def analytic_ground(m, n, k):
    return - p * m - q * n - r * k + D


def kink(m, n, k):
    mp = ap ** m * bp ** n * gp ** k
    mm = am ** m * bm ** n * gm ** k
    return (c1 * mp - c2 * mm) / (c1 * mp + c2 * mm)


def analytic_plus(m, n):
    m1 = Decimal((2 * m + n + 1) // 3)
    n1 = Decimal((2 * n + m - 1) // 3)
    k1 = Decimal((n - m + 1) // 3)
    return analytic_ground(m1, n1, k1) + lamb * kink(m1, n1, k1)


def analytic_minus(m, n):
    m1 = Decimal((2 * m + n - 1) // 3)
    n1 = Decimal((2 * n + m + 1) // 3)
    k1 = Decimal((n - m - 1) // 3)
    return analytic_ground(m1, n1, k1) + lamb * kink(m1, n1, k1)


def analytic_ground_plus(m, n):
    m1 = Decimal((2 * m + n + 1) // 3)
    n1 = Decimal((2 * n + m - 1) // 3)
    k1 = Decimal((n - m + 1) // 3)
    return analytic_ground(m1, n1, k1)


def analytic_ground_minus(m, n):
    m1 = Decimal((2 * m + n - 1) // 3)
    n1 = Decimal((2 * n + m + 1) // 3)
    k1 = Decimal((n - m - 1) // 3)
    return analytic_ground(m1, n1, k1)


def gen_soliton_init(p1, p2, vec):
    result = []
    for i in range(0, LEN // 2):
        result.append(_get_init_point(p2, analytic_plus, vec, i))
        result.append(_get_init_point(p1, analytic_minus, vec, i + 1))
    return result


def gen_ground_init(p1, p2, vec):  # p1 minusowy p2 plusowy
    result = []
    for i in range(0, LEN // 2):
        result.append(_get_init_point(p2, analytic_ground_plus, vec, i))
        result.append(_get_init_point(p1, analytic_ground_minus, vec, i + 1))
    return result

def gen_custom_init(p1, p2, vec, tab):  # p1 minusowy p2 plusowy
    result = []
    for i in range(0, LEN // 2):
        result.append(_get_init_point(p2, analytic_ground_plus, vec, i))
        result.append(_get_init_point(p1, analytic_ground_minus, vec, i + 1))
    for i in range(len(result)):
        try:
            result[i].value+=Decimal(tab[i])
        except IndexError:
            result[i].value+=Decimal(tab[-1])
    return result

def gen_const_init(p1, p2, vec, constant):  # p1 minusowy p2 plusowy
    result = []
    for i in range(0, LEN // 2):
        result.append(_get_init_point(p2, analytic_ground_plus, vec, i, constant))
        result.append(_get_init_point(p1, analytic_ground_minus, vec, i + 1, constant))
    return result


def _get_init_point(point_vec, point_func, direction_vector, modifier, contsant=Decimal(0.)):
    m = point_vec[0] + modifier * direction_vector[0]
    n = point_vec[1] + modifier * direction_vector[1]
    value = point_func(m, n) + contsant
    return Point(value=value, m=m, n=n)


class Point:
    def __init__(self, value, m, n):
        self.cartesian = [m + 0.5 * n, n * (3 ** 0.5) / 2]
        self.rhombic = [m, n]
        self.value = value
        if (2 * m + n) % 3 == 1:
            self.sign = -1
        if (2 * m + n) % 3 == 2:
            self.sign = 1

    def get(self, raw, cutoff):
        if raw:
            if (cutoff > self.value > -cutoff) or cutoff == 0:
                return [self.cartesian[0], self.cartesian[1], self.value]

            elif self.value < -cutoff:
                return [self.cartesian[0], self.cartesian[1], -cutoff]
            else:
                return [self.cartesian[0], self.cartesian[1], cutoff]
        background = 0
        m, n = self.rhombic[0], self.rhombic[1]
        if self.sign == -1:
            background = analytic_ground_minus(m, n)
        if self.sign == 1:
            background = analytic_ground_plus(m, n)
        if (cutoff > self.value - background > -cutoff) or cutoff == 0:
            return [self.cartesian[0], self.cartesian[1], self.value - background]
        elif self.value - background < -cutoff:
            return [self.cartesian[0], self.cartesian[1], -cutoff]
        else:
            return [self.cartesian[0], self.cartesian[1], cutoff]


class Initial:
    def __init__(self, branches):
        g, b, d = branches[0][0], branches[1][0], branches[2][0]
        self.center = self.get_central_point(g, b, d)
        self.v = []
        self.v.append(branches[1][::-1] + [self.center] + branches[0][0:LEN - 1])
        self.v.append(branches[2][::-1] + [self.center] + branches[1][0:LEN - 1])
        self.v.append(branches[0][::-1] + [self.center] + branches[2][0:LEN - 1])
        print("Initial condition generated...")

    @staticmethod
    def get_central_point(g, b, d):
        cm = (g.rhombic[0] + b.rhombic[0] + d.rhombic[0]) / 3
        cn = (g.rhombic[1] + b.rhombic[1] + d.rhombic[1]) / 3
        pq, qr, rp = (p ** 2 - q ** 2), (q ** 2 - r ** 2), (r ** 2 - p ** 2)
        gv, bv, dv = g.value, b.value, d.value
        value = (pq * bv * gv + qr * dv * gv + rp * dv * bv) / (-qr * bv - rp * gv - pq * dv)
        return Point(value, cm, cn)


def det_point_type(a, b, ce):
    cases = {(0, 1): 0, (0, -1): 0, (-1, 0): 1, (1, 0): 1, (1, -1): 2, (-1, 1): 2}
    vec = (2 * b.rhombic[0] - a.rhombic[0] - ce.rhombic[0], 2 * b.rhombic[1] - a.rhombic[1] - ce.rhombic[1])
    return cases[vec], [b.rhombic[0] + vec[0], b.rhombic[1] + vec[1]]


def solve_for_point(a, b, ce):
    central = b.value
    left = a.value - central
    right = ce.value - central
    pq, qr, rp = (p ** 2 - q ** 2), (q ** 2 - r ** 2), (r ** 2 - p ** 2)
    eqs = [
        -(rp * right * left / (pq * left + qr * right)) + central,
        -(qr * right * left / (rp * left + pq * right)) + central,
        -(pq * right * left / (qr * left + rp * right)) + central,
    ]
    ptype, rhombic = det_point_type(a, b, ce)
    value = eqs[ptype]
    return Point(value, rhombic[0], rhombic[1])


def propagate(v, sink):
    newv = []
    r1 = len(v) // 2
    for i in range(0, r1, 2):
        try:
            newv.append(solve_for_point(v[r1 + i], v[r1 + 1 + i], v[r1 + 2 + i]))
            newv.insert(0, solve_for_point(v[r1 - 3 - i], v[r1 - 2 - i], v[r1 - 1 - i]))
            r2 = len(newv) // 2
            newv.append(solve_for_point(newv[r2 - 1 + i], newv[r2 + i], v[r1 + i + 1]))
            newv.insert(0, solve_for_point(v[r1 - 2 - i], newv[r2 - 1 - i], newv[r2 - i]))
        except IndexError:
            if newv:
                sink += newv
                return propagate(newv, sink)
            else:
                print("Region propagated...")
                break

def scatter(pkty, raw=False, cutoff=0):
    ini = Initial(pkty)
    rozwiazanie1 = deepcopy(ini.v[0])
    rozwiazanie2 = deepcopy(ini.v[1])
    rozwiazanie3 = deepcopy(ini.v[2])
    propagate(ini.v[0], rozwiazanie1)
    propagate(ini.v[1], rozwiazanie2)
    propagate(ini.v[2], rozwiazanie3)
    nareszcie = [rozw.get(raw, cutoff) for rozw in rozwiazanie1 + rozwiazanie2 + rozwiazanie3]
    w1 = [rozw.get(raw, cutoff) for rozw in pkty[0]]
    w2 = [rozw.get(raw, cutoff) for rozw in pkty[1]]
    w3 = [rozw.get(raw, cutoff) for rozw in pkty[2]]
    print("Drawing...")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for e in nareszcie:
        if cutoff==0 :
            ax.scatter(float(e[0]), float(e[1]), float(e[2]), zdir='z', c='b')
        else:
            ax.scatter(float(e[0]), float(e[1]), float(e[2]), zdir='z', c=[(0,0,(float(e[2])+cutoff)/(2*cutoff))])#((float(e[2])+3)/5,(float(e[2])+3)/5,(float(e[2])+3)/5)
    for e in w1:
        ax.scatter(float(e[0]), float(e[1]), float(e[2]), zdir='z', c='r')
    for e in w2:
        ax.scatter(float(e[0]), float(e[1]), float(e[2]), zdir='z', c='g')
    for e in w3:
        ax.scatter(float(e[0]), float(e[1]), float(e[2]), zdir='z', c='k')
    #ax.set_zscale('log')
    plt.show()

def heatmap(pkty, raw=False, cutoff=0):
    ini = Initial(pkty)
    rozwiazanie1 = deepcopy(ini.v[0])
    rozwiazanie2 = deepcopy(ini.v[1])
    rozwiazanie3 = deepcopy(ini.v[2])
    propagate(ini.v[0], rozwiazanie1)
    propagate(ini.v[1], rozwiazanie2)
    propagate(ini.v[2], rozwiazanie3)
    nareszcie = [rozw.get(raw, cutoff) for rozw in rozwiazanie1 + rozwiazanie2 + rozwiazanie3]
    x=[float(t[0]) for t in nareszcie]
    y=[float(t[1]) for t in nareszcie]
    z=[float(t[2]) for t in nareszcie]
    zmax=max(z)
    zmin=min(z)
    col=[float(t[2]) for t in nareszcie]
    print("Drawing...")
    plt.scatter(x,y,c=col)
    plt.show()



ic = [2, 0]
i1 = [ic[0], ic[1] + 1]
i2 = [ic[0] - 1, ic[1]]
i3 = [ic[0] + 1, ic[1] - 1]


def hack_draw(pkty, raw=False, cutoff=0,ptype=scatter):
    tmp = gen_soliton_init([2, 0], i3, [1, -2])
    pkty[2][0] = tmp[0]
    ptype(pkty, raw, cutoff)


def hack2_draw(pkty, raw=False, cutoff=0):
    pkty[2][1].value += Decimal('1')
    pkty[2][2].value += Decimal('0')
    heatmap(pkty, raw, cutoff)

#Tutaj wpisujemy kolejne przesunięcia, pierwszy się nie liczy bo później nadpisujemy go solitonowym warunkiem a ostatni jest rozpropagowywany do końca warunku.


punkty = [gen_soliton_init(ic, i1, [1, 1]), gen_soliton_init(ic, i2, [-2, 1]), gen_soliton_init(ic, i3, [1, -2])]


scatter(punkty, raw=False, cutoff=1.4)
#hack_draw(punkty, raw=False, cutoff=20, ptype=heatmap)
