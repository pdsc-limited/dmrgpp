import unittest
import numpy as np
from ed_reference import (basis, sector_hamiltonian, product_state, evolve,
                          observables, green_functions)

class EDReferenceTests(unittest.TestCase):
    def test_atomic_analytic_and_equal_time(self):
        sec,v=product_state(1,[0],[])
        _,h=sector_hamiltonian(1,1,0,4,potential=[-2])
        self.assertAlmostEqual(h[0,0].real,-2)
        _,hd=sector_hamiltonian(1,1,1,4,potential=[-2])
        self.assertAlmostEqual(hd[0,0].real,0)
        g=green_functions(1,4,sec,[v],[(.2,[])],potential=[-2])
        np.testing.assert_allclose(np.diag(g['lesser']),1j)
        np.testing.assert_allclose(np.diag(g['greater']),0)
        np.testing.assert_allclose(np.diag(g['lesser']+g['greater']),1j)
        self.assertEqual(len(g['per_seed']),1)

    def test_complex_hermitian_evolution_observables(self):
        sec,v=product_state(2,[0],[]); V=.3+.4j
        _,h=sector_hamiltonian(2,1,0,1,couplings=[V])
        np.testing.assert_allclose(h,h.conj().T)
        times,ss=evolve(2,1,sec,[v],[(.2,[V]),(.1,[-.2j])])
        self.assertAlmostEqual(np.linalg.norm(ss[-1][0]),1)
        self.assertAlmostEqual(times[-1],.3)
        r=observables(2,1,sec,[ss[-1][0]])
        self.assertAlmostEqual(r['density'].sum(),1)
        self.assertGreaterEqual(r['one_body_eigenvalues'].min(),-1e-12)
        self.assertAlmostEqual(r['one_body_eigenvalues'].sum(),1)
        np.testing.assert_allclose(r['one_body_eigenvalues'],[0,0,0,1],atol=1e-12)

    def test_atomic_paramagnetic_ensemble(self):
        dt = .13
        U = 3.2
        matrices = []
        for up, down in [([0], []), ([], [0])]:
            sector, state = product_state(1, up, down)
            matrices.append(green_functions(1, U, sector, [state], [(dt, [])] * 3,
                                             potential=[-U / 2]))
        lesser = sum(g['lesser'] for g in matrices) / 2
        greater = sum(g['greater'] for g in matrices) / 2
        times = matrices[0]['times']
        difference = times[:, None] - times[None, :]
        np.testing.assert_allclose(lesser, .5j * np.exp(.5j * U * difference), atol=1e-12)
        np.testing.assert_allclose(greater, -.5j * np.exp(-.5j * U * difference), atol=1e-12)
        np.testing.assert_allclose(np.diag(greater - lesser), -1j, atol=1e-12)

    def test_branch_matches_full_fock_nonstationary(self):
        # Independent full-Fock construction with interleaved (site,spin) JW orbitals.
        ns=2; dims=1<<(2*ns)
        def op(q,create):
            M=np.zeros((dims,dims),complex)
            for x in range(dims):
                occ=(x>>q)&1
                if occ==bool(create): continue
                sign=(-1)**((x&((1<<q)-1)).bit_count())
                y=x^(1<<q); M[y,x]=sign
            return M
        cs=[op(2*i+s,False) for i in range(ns) for s in (0,1)]
        cds=[x.conj().T for x in cs]
        U=1.7; potential=np.array([-.6,.1]); eps=np.array([0,.23])
        intervals=[(.17,[.31+.22j]),(.13,[-.18+.37j])]
        Hs=[]
        for dt,V in intervals:
            H=np.zeros((dims,dims),complex)
            for i in range(ns):
                for s in (0,1): H+=(potential[i]+eps[i])*cds[2*i+s]@cs[2*i+s]
            H+=U*cds[0]@cs[0]@cds[1]@cs[1]
            for s in (0,1): H+=V[0]*cds[2+s]@cs[s]+V[0].conjugate()*cds[s]@cs[2+s]
            Hs.append(H)
        # Preserve the original spin-up-only seed, and also exercise U with a
        # doubly occupied impurity for both spin-resolved probe operators.
        cumulative=[np.eye(dims)]
        for (dt,_),H in zip(intervals,Hs):
            cumulative.append(__import__('scipy').linalg.expm(-1j*dt*H)@cumulative[-1])

        cases=[(1,[0],[],0)]
        cases.extend((3,[0],[0],spin) for spin in (0,1))
        for seed_index,up,down,spin in cases:
            x=np.zeros(dims,complex); x[seed_index]=1
            times=len(cumulative); bruteL=np.zeros((times,times),complex); bruteG=np.zeros_like(bruteL)
            c=cs[spin]; cd=cds[spin]
            c_at=[]; cd_at=[]
            for evolution in cumulative:
                c_at.append(evolution.conj().T@c@evolution)
                cd_at.append(evolution.conj().T@cd@evolution)
            for a in range(times):
                for b in range(times):
                    bruteL[a,b]=1j*np.vdot(x,cd_at[b]@c_at[a]@x)
                    bruteG[a,b]=-1j*np.vdot(x,c_at[a]@cd_at[b]@x)
            sec,v=product_state(ns,up,down)
            got=green_functions(ns,U,sec,[v],intervals,spin=spin,
                                potential=potential,energies=eps)
            np.testing.assert_allclose(got['lesser'],bruteL,atol=2e-12)
            np.testing.assert_allclose(got['greater'],bruteG,atol=2e-12)
            if seed_index==1:
                # -i G< and i G> are positive kernels.
                self.assertGreater(np.linalg.eigvalsh(-1j*got['lesser']).min(),-1e-11)
                self.assertGreater(np.linalg.eigvalsh(1j*got['greater']).min(),-1e-11)
                self.assertGreater(np.max(np.abs(got['lesser']-got['lesser'].conj().T)),0)
                self.assertFalse(np.allclose(got['lesser'],got['lesser'][0,0]))

if __name__=='__main__': unittest.main()
