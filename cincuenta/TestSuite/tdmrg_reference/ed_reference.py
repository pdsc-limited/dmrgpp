"""Small ED oracle for prescribed time-dependent star SIAMs.

Sites: impurity 0, baths 1..L; spin-up and spin-down have separate bitmasks.
H = U n0u n0d + sum_i (potential_i+energy_i)n_i + sum_{i>0,s}
(V_i c†_{i,s}c_{0,s}+conj(V_i)c†_{0,s}c_{i,s}). HubbardOneBand
potential[0]=-U/2 gives the shifted interaction, differing from
U(nu-1/2)(nd-1/2) by constant U/4. product_state uses bit/site lists, not
DMRG++ PureState symmetry-index encoding (pilot indices 2=UP, 1=DOWN).
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


def basis(nsites,nup,ndown):
    """Ordered fixed-(Nup,Ndown) basis of (up_mask,down_mask)."""
    def words(n): return [sum(1<<i for i in c) for c in itertools.combinations(range(nsites),n)]
    return [(u,d) for u in words(nup) for d in words(ndown)]


def _apply(mask,site,create):
    occupied=(mask>>site)&1
    if occupied==bool(create): return None
    return mask^(1<<site),(-1 if (mask&((1<<site)-1)).bit_count()%2 else 1)


def sector_hamiltonian(nsites,nup,ndown,U,potential=None,energies=None,couplings=None):
    """Return (basis,H); each coupling V multiplies c†_bath c_imp, plus h.c."""
    b=basis(nsites,nup,ndown); ix={s:i for i,s in enumerate(b)}; L=nsites-1
    potential=np.zeros(nsites) if potential is None else np.asarray(potential)
    energies=np.zeros(nsites) if energies is None else np.asarray(energies)
    couplings=np.zeros(L,complex) if couplings is None else np.asarray(couplings,complex)
    if potential.shape!=(nsites,) or energies.shape!=(nsites,) or couplings.shape!=(L,): raise ValueError('invalid parameter shape')
    H=np.zeros((len(b),len(b)),complex)
    for j,(u,d) in enumerate(b):
        ou=[(u>>i)&1 for i in range(nsites)]; od=[(d>>i)&1 for i in range(nsites)]
        H[j,j]=U*ou[0]*od[0]+sum((potential[i]+energies[i])*(ou[i]+od[i]) for i in range(nsites))
        for s,mask in enumerate((u,d)):
            for k,V in enumerate(couplings,1):
                for src,dst,coef in ((0,k,V),(k,0,V.conjugate())):
                    a=_apply(mask,src,False)
                    if a is None: continue
                    m,x=a; z=_apply(m,dst,True)
                    if z is None: continue
                    m,y=z; state=(m,d) if s==0 else (u,m); H[ix[state],j]+=coef*x*y
    return b,H


def _op(nsites,bfrom,bto,site,spin,create):
    ix={s:i for i,s in enumerate(bto)}; M=np.zeros((len(bto),len(bfrom)),complex)
    for j,(u,d) in enumerate(bfrom):
        a=_apply(u if spin==0 else d,site,create)
        if a:
            m,sgn=a; state=(m,d) if spin==0 else (u,m)
            if state in ix: M[ix[state],j]=sgn
    return M


def product_state(nsites,up_sites=(),down_sites=()):
    """Return (sector, unit vector) for occupied site lists (0 is impurity)."""
    u=sum(1<<i for i in up_sites); d=sum(1<<i for i in down_sites); sec=(u.bit_count(),d.bit_count())
    v=np.zeros(len(basis(nsites,*sec)),complex); v[basis(nsites,*sec).index((u,d))]=1
    return sec,v


def _prop(H,v,dt): return expm_multiply(-1j*dt*H,v) if H.shape[0]>64 else expm(-1j*dt*H)@v


def _hamiltonians(nsites,sector,U,intervals,potential,energies):
    return [sector_hamiltonian(nsites,*sector,U,potential,energies,V)[1] for _,V in intervals]


def _trajectory(v,Hs,intervals):
    out=[np.asarray(v,complex).copy()]
    for (dt,_),H in zip(intervals,Hs): out.append(_prop(H,out[-1],dt))
    return out


def evolve(nsites,U,initial_sector,initial_states,intervals,potential=None,energies=None):
    """Propagate state ensemble over (duration,coupling-vector) intervals; return times, snapshots."""
    intervals=list(intervals); Hs=_hamiltonians(nsites,initial_sector,U,intervals,potential,energies)
    traj=[_trajectory(v,Hs,intervals) for v in initial_states]
    return np.r_[0,np.cumsum([x[0] for x in intervals])],[[x[n] for x in traj] for n in range(len(intervals)+1)]


def observables(nsites,U,sector,states,potential=None,energies=None,couplings=None):
    """Return ensemble site/spin density, impurity docc, and 1-RDM eigenvalues."""
    b = basis(nsites, *sector)
    states = [np.asarray(x, complex) for x in states]
    if not states:
        raise ValueError('at least one ensemble state is required')
    probabilities = sum(abs(x) ** 2 for x in states) / len(states)
    density = np.zeros((nsites, 2))
    for q, (u, d) in enumerate(b):
        for i in range(nsites):
            density[i] += probabilities[q] * np.array([(u >> i) & 1, (d >> i) & 1])
    docc = sum(probabilities[q] * bool(u & 1) * bool(d & 1)
               for q, (u, d) in enumerate(b))
    # <c_i† c_j> = <c_i psi | c_j psi>; annihilation changes sector.
    # Applying separate same-sector c matrices would incorrectly give zero.
    one = np.zeros((2 * nsites, 2 * nsites), complex)
    for spin in (0, 1):
        lower = list(sector)
        lower[spin] -= 1
        if lower[spin] < 0:
            continue
        destination = basis(nsites, *lower)
        operators = [_op(nsites, b, destination, i, spin, False)
                     for i in range(nsites)]
        block = np.zeros((nsites, nsites), complex)
        for state in states:
            removed = np.column_stack([op @ state for op in operators])
            block += removed.conj().T @ removed / len(states)
        sl = slice(spin * nsites, (spin + 1) * nsites)
        one[sl, sl] = block
    return {'density': density, 'double_occupancy': docc,
            'one_body_eigenvalues': np.linalg.eigvalsh(one)}


def green_functions(nsites,U,sector,states,intervals,site=0,spin=0,potential=None,energies=None):
    """Exact causal-grid G</> (lesser i<c†c>, greater -i<cc†>), ensemble and per seed.

    Intervals define grid points; returns ``times``, ensemble ``lesser`` and
    ``greater`` matrices, plus per-input-state matrices in ``per_seed``. Causal
    branches are inserted at every grid point and propagated sector-by-sector.
    """
    intervals=list(intervals); n=len(intervals)+1; Nup,Ndn=sector
    H0=_hamiltonians(nsites,sector,U,intervals,potential,energies)
    sm=(Nup-1,Ndn) if spin==0 else (Nup,Ndn-1); sp=(Nup+1,Ndn) if spin==0 else (Nup,Ndn+1)
    bm=basis(nsites,*sm) if min(sm)>=0 else []; bp=basis(nsites,*sp) if max(sp)<=nsites else []
    b0=basis(nsites,*sector); C=_op(nsites,b0,bm,site,spin,False) if bm else np.zeros((0,len(b0)))
    D=_op(nsites,b0,bp,site,spin,True) if bp else np.zeros((0,len(b0)))
    Cplus=_op(nsites,bp,b0,site,spin,False) if bp else np.zeros((len(b0),0))
    Hm=_hamiltonians(nsites,sm,U,intervals,potential,energies) if bm else []
    Hp=_hamiltonians(nsites,sp,U,intervals,potential,energies) if bp else []
    seed=[]
    for initial in states:
        psi=_trajectory(initial,H0,intervals); less=np.zeros((n,n),complex); greater=np.zeros_like(less)
        # For each insertion index j, evolve c psi[j], c† psi[j] on causal branches.
        for j in range(n):
            minus=C@psi[j]; plus=D@psi[j]
            for k in range(j,n):
                if k>j:
                    dt=intervals[k-1][0]; minus=_prop(Hm[k-1],minus,dt) if bm else minus; plus=_prop(Hp[k-1],plus,dt) if bp else plus
                less[k,j]=1j*np.vdot(minus,C@psi[k])
                greater[k,j]=-1j*np.vdot(psi[k],Cplus@plus) if bp else 0j
        for k in range(n):
            for j in range(k): less[j,k]=-less[k,j].conjugate(); greater[j,k]=-greater[k,j].conjugate()
        seed.append({'lesser':less,'greater':greater})
    avg={key:sum(x[key] for x in seed)/len(seed) for key in ('lesser','greater')}
    return {'times':np.r_[0,np.cumsum([x[0] for x in intervals])],**avg,'per_seed':seed}
