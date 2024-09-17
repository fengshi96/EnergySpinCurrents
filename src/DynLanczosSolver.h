//
// Created by Feng, Shi on 4/2/24.
//

#ifndef LANCZOSCOND_DYNLANCZOSSOLVER_H
#define LANCZOSCOND_DYNLANCZOSSOLVER_H
#include <iomanip>
#include "QuantumBasis.h"
#include "Matrix.h"
#include "Lattice.h"
#include "Hamiltonian.h"
#include <cmath>

typedef Eigen::VectorXf VectorXf;
typedef std::complex<double> dcomplex;    // your typedef

extern "C" void dstev_(char *,int *, double *, double *, double *, int *, double *, int *);
extern "C" void sgeqrf_(int *, int *, double *, int *, double *, double *, int *, int *);
extern "C" void dorgqr_(int *, int *, int *, double *, int *, double *, double *, int *, int *);
extern "C" void zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
                       std::complex<double> *,int *, double *, int *);


class DynLanczosSolver {
public:
    DynLanczosSolver(ConstVariables& variables,
                     QBasis& Basis, Hamiltonian& Hamil):
            variables_(variables), Basis_(Basis), Hamil_(Hamil),
            Nsite_(variables.NumberofSites)
    {
    }

    vector<Eigen::VectorXcd> PsiAll;
    Eigen::VectorXd TriDeval;
    Eigen::VectorXcd Vorig;

    // -------------------------------------------------------
    Eigen::VectorXcd Lanczos_Nirav(string PsiOrAll){
        std::cout << "Begin Lanczos! --- "<<std::endl;
        int iter, Lexit;
        double E0, D0, diff;
        dcomplex zero(0.0,0.0);
        int STARTIT=3; //iteration which diagonz. begins
        int MAXiter=variables_.LanczosSteps;
        int EXITiter=-1;
        double LanczosEps=variables_.LanczosEps;
        double vm, rss;

        const int N=Basis_.basis.size();
        Eigen::VectorXcd V0(N), V1(N), V2(N), e(N), d(N);
        vector<dcomplex> alpha, beta;
        Matrix<dcomplex> TrEVec;
        vector<Eigen::VectorXcd> LancBasis(MAXiter+1);

        V0.setZero(); V1.setZero(); V2.setZero();
        e.setZero(); d.setZero();
        Vorig.resize(N); Vorig.setZero();
        Randomize(Vorig);			//Vorig = Vectpush_backorXf::Random(N);
        Vorig.normalize();			//normalize(Vorig);

//		if(N<512) {
//			return ExactDiag();
//		}

        // ----- Begin Lanczos -------------
        V0 = Vorig;
        //		for (int i=0; i<N; i++) {
        //			cout << V0[i] << endl;
        //		}
        //V0.fill(1.0);
        //V0.normalize();

        beta.push_back(zero);  //beta_0 not defined
        LancBasis[0] = V0;
        V1 = Hamil_.MatrixVectorMult(V0);		// |V1> = H |V0>
        dcomplex tmp = V0.dot(V1);
        alpha.push_back(tmp);

        V1 = V1 - alpha[0]*V0; // (Equation 2.3) Rev. Mod. Phys., 66, 3 (1994) - note beta = 0.0

        beta.push_back(V1.norm());  // beta = <V1 | V1>
        V1.normalize(); //V1 = V1/beta[1];
        LancBasis[1] = V1;

        // done 0th iteration
        Lexit = 0;   //exit flag
        E0 = 1.0;    //previous iteration GS eigenvalue

        iter = 0;
//		// ------ while ----------------
        while(Lexit != 1) {

            iter++;
            V2 = Hamil_.MatrixVectorMult(V1); // V2 = H |V1>
            alpha.push_back(V1.dot(V2)); // <V1 | H | V1>
            V2 = V2 - alpha[iter]*V1 - beta[iter]*V0; // (Equation 2.3) Rev. Mod. Phys. , Vol. 66, No. 3, July 1994

            dcomplex norm = V2.norm();
            beta.push_back(norm);
            V2.normalize(); //V2 = V2/beta[iter+1];
            LancBasis[iter+1] = V2;

            // --- Reorthogonalization of basis states ---- //

            bool ReOrtho=true;
            if(PsiOrAll=="OnlyEgs") ReOrtho=false;
            if(ReOrtho && iter%1==0) {
                //cout << "  Re-orthogonalization " << endl;
                for(int i=0; i<iter+1; i++) {
                    dcomplex rij = LancBasis[i].dot(LancBasis[iter+1]);

                    // Vj = Vj - <Vi|Vj> Vi -- gram-schmid
#pragma omp parallel for
                    for (int h=0;h<N;h++)
                        LancBasis[iter+1](h) -= LancBasis[i](h)*rij;
                }
                LancBasis[iter+1].normalize();
                V2 = LancBasis[iter+1];
            } //--------------------------------------------- //

            V0 = V1;
            V1 = V2;

            //diagonalize tri-di matrix
            D0 = TriDiag('N',alpha,beta,TrEVec, TriDeval); //TrEVec.print();
            diff = abs(E0-D0);

//            if(PsiOrAll=="OnlyEgs" && iter>2) {
//                LancBasis[iter-1].resize(1);
//            }
            process_mem_usage(vm, rss);

            std::cout << iter << " \t " <<setprecision(16)<< D0 <<  " \t "
                      << " Eps: " << diff
                      << " \t    VM (MB): " << int(vm/1024)
                      << "       RSS (MB): " << int(rss/1024)
                      << std::endl;

            if(diff < LanczosEps) {
                Lexit = 1;
                E0 = D0;
                std::cout << std::endl;
                std::cout << "  Output Energy: " <<setprecision(16)<< D0
                          << " After " << iter << " iterations "
                          << " Eps: " << diff
                          << std::endl;
                std::cout << std::endl;
            }
            else {
                E0=D0;
            }


            //			int pmax = (iter>6) ? 6: iter;
            //			for (int i=0; i<pmax; i++) {
            //				cout << TriDeval[i] << " ";
            //			}

            if(iter==MAXiter-1 || Lexit==1) {
                Lexit=1;
                D0 = TriDiag('V',alpha,beta,TrEVec, TriDeval); //TrEVec.print();
            }
        } // ------ while --------
        EXITiter = iter+1;

        std::cout << "  Lowest Eigen Value: " << D0 << endl;
        std::cout << "  With Lanczos EPS:   " << diff << endl;
        std::cout << "  Alpha and beta of Tridiag Matrix : ";
        for (int m=0; m<EXITiter; m++) {
            cout << "(" << real(alpha[m]) << "," << real(beta[m]) << ") ";
        }
        cout << endl << endl << flush;



        int loopval=100000000;
        Eigen::VectorXcd Psi(N);
        if(PsiOrAll=="OnlyEgs") {
            Psi.setZero();
            return Psi;
        } else if(PsiOrAll=="GS") {
            loopval=1;
            cout << "  creating all eigen-states " << endl;
            PsiAll.resize(loopval);
#pragma omp parallel for
            for (int k=0; k<loopval; k++) {
                PsiAll[k].resize(N);
                PsiAll[k].setZero();
                for (int h=0; h<N; h++) {
                    for (int m=0; m<EXITiter; m++) {
                        dcomplex tmp = LancBasis[m](h)*TrEVec(m,k);
                        //dcomplex val(tmp.real(),0.0);
                        PsiAll[k](h) += tmp;
                        //cout << k << "-" << h << "-" << m << "-" << LancBasis[m](h) << "-" << TrEVec(m,k) << endl;
                    }
                }
                process_mem_usage(vm, rss);
                cout << k << "    "
                     << " \t    VM (MB): " << int(vm/1024)
                     << "       RSS (MB): " << int(rss/1024)
                     << endl;
                cout.flush();
            }
            cout << endl;
            LancBasis.clear();
            Eigen::VectorXcd Psi = PsiAll[0];
            return Psi;
        } else if(PsiOrAll=="All") {
            loopval=EXITiter;
            cout << "  creating all eigen-states " << endl;
            PsiAll.resize(loopval);
#pragma omp parallel for
            for (int k=0; k<loopval; k++) {
                PsiAll[k].resize(N);
                PsiAll[k].setZero();
                for (int h=0; h<N; h++) {
                    for (int m=0; m<EXITiter; m++) {
                        dcomplex tmp = LancBasis[m](h)*TrEVec(m,k);
                        //dcomplex val(tmp.real(),0.0);
                        PsiAll[k](h) += tmp;
                        //cout << k << "-" << h << "-" << m << "-" << LancBasis[m](h) << "-" << TrEVec(m,k) << endl;
                    }
                }
                process_mem_usage(vm, rss);
                cout << k << "    "
                     << " \t    VM (MB): " << int(vm/1024)
                     << "       RSS (MB): " << int(rss/1024)
                     << endl;
                cout.flush();
            }
            cout << endl;
            LancBasis.clear();
            Eigen::VectorXcd Psi = PsiAll[0];
            return Psi;
        } else {
            throw runtime_error("DynLanczosSolver: Use PsiOrAll==GS or All");
        }

    }



    //		cout << "  creating eigen-states " << endl;
    //		//loopval=10;
    //		PsiAll.resize(loopval);
    //		#pragma omp parallel for
    //		for (int k=0; k<loopval; k++) {
    //			PsiAll[k].resize(N);
    //			PsiAll[k].setZero();
    //			for (int h=0; h<N; h++) {
    //				for (int m=0; m<EXITiter; m++) {
    //					dcomplex tmp = LancBasis[m](h)*TrEVec(m,k);
    //					//dcomplex val(tmp.real(),0.0);
    //					PsiAll[k](h) += tmp;
    //					//cout << k << "-" << h << "-" << m << "-" << LancBasis[m](h) << "-" << TrEVec(m,k) << endl;
    //				}
    //			}
    //			process_mem_usage(vm, rss);
    //			cout << k << "    "
    //			     << " \t    VM (MB): " << int(vm/1024)
    //			     << "       RSS (MB): " << int(rss/1024)
    //			     << endl;push_back
    //			cout.flush();
    //		}
    //		cout << endl;
    //		LancBasis.clear();


    void fillFromED(Eigen::MatrixXcd Evec, Eigen::VectorXd eval) {
        int N = Evec.rows();
        TriDeval.resize(N);
        PsiAll.resize(N);

        for (int k=0; k<N; k++) {
            TriDeval[k] = eval[k];
            PsiAll[k].resize(N);
            PsiAll[k].setZero();
            for (int h=0; h<N; h++) {
                PsiAll[k](h) = Evec(h,k);
            }
        }

    }


    // ================================================
    // --------- distribute dynamics ------------------
    Eigen::VectorXcd ExactDiag() {

        std::cout << "Begin Exact Diagonalization! --- "<<std::endl;
        int N = Basis_.basis.size();
        Matrix<dcomplex> Ham(N,N);
        vector<double> D(N);
        TriDeval.resize(N);
        PsiAll.resize(N);

        //make the Hamiltonian
        for (int i=0; i<N; i++) {
            dcomplex Hij=Hamil_.HDiag[i];
            Ham(i,i)+=Hij;
        }

        int hilbert_t=Hamil_.HTSxx.size();
        // Kxx Kitaev - Sector -------
        for(int i=0;i<hilbert_t;i++){
            int Hi=Hamil_.Sxx_init[i];
            int Hj=Hamil_.Sxx_final[i];
            assert(Hi<N && Hj<N);
            dcomplex Hij=Hamil_.HTSxx[i];
            Ham(Hi,Hj) += Hij;
        }

        hilbert_t=Hamil_.HTSyy.size();
        // Kxx Kitaev - Sector -------
        for(int i=0;i<hilbert_t;i++){
            int Hi=Hamil_.Syy_init[i];
            int Hj=Hamil_.Syy_final[i];
            assert(Hi<N && Hj<N);
            dcomplex Hij=Hamil_.HTSyy[i];
            Ham(Hi,Hj) += Hij;
        }
        assert(Ham.IsHermitian());
        Matrix<dcomplex> Evec = Ham;
        Diagonalize('V',Evec,D);
        cout << " Ground-state energy:= " << D[0] << endl;
        //for (int i=0; i<D.size(); i++) cout << D[i] << " \t " << endl;

        cout << " creating eigen-states " << endl;
        for (int k=0; k<N; k++) {
            TriDeval[k] = D[k];
            PsiAll[k].resize(N);
            PsiAll[k].setZero();
            for (int h=0; h<N; h++) {
                PsiAll[k](h) += Evec(h,k);
            }
        }

        return PsiAll[0];
    }

    // -------------------------------------------------------
    void Diagonalize(char option, Matrix<dcomplex>& A, vector<double>& eval){
        char jobz=option;
        char uplo='U';
        int n=A.n_row();
        int lda=A.n_col();
        assert(n==lda);

        eval.resize(n);
        vector<dcomplex> work(3);
        vector<double> rwork(3*n);
        int info,lwork= -1;

        // ---- spin up part of the Hamiltonian --------------------
        fill(eval.begin(),eval.end(),0);
        // query:
        zheev_(&jobz,&uplo,&n,&(A(0,0)),&lda,&(eval[0]),&(work[0]),&lwork,&(rwork[0]),&info);
        lwork = int(real(work[0]))+1;
        work.resize(lwork+1);
        // real work:
        zheev_(&jobz,&uplo,&n,&(A(0,0)),&lda,&(eval[0]),&(work[0]),&lwork,&(rwork[0]),&info);
        if (info!=0) {
            std::cerr<<"info="<<info<<"\n";
            perror("diag: zheev: failed with info!=0.\n");
        }
        //for(int i=0;i<n;i++) { cout << eval[i] << " \t ";} cout << endl;
        //sort(eigs_.begin(),eigs_.end()); // sort Eigenvalues and Hamiltonian
    }

    // -------------------------------------------------------
    double TriDiag(char option, vector<dcomplex>& alpha,
                   vector<dcomplex>& beta, Matrix<dcomplex>& Z,
                   Eigen::VectorXd &eigs){

        char jobz=option;
        char uplo='U';
        int n=alpha.size();
        int lda=n;
        dcomplex zero(0.0,0.0);
        Z.resize(n,lda); Z.fill(zero);
        eigs.resize(n);

        for (int i=0;i<n-1;i++){
            Z(i,i) = alpha[i];
            Z(i,i+1) = beta[i+1];
            Z(i+1,i) = conj(beta[i+1]);
        }
        Z(n-1,n-1) = alpha[n-1];
        //Z.print();

        vector<std::complex<double>> work(3);
        vector<double> rwork(3*n);
        int info;
        int lwork= -1;

        // query:
        zheev_(&jobz,&uplo,&n,&(Z(0,0)),&lda,&(eigs[0]),&(work[0]),&lwork,&(rwork[0]),&info);
        if (info!=0) {
            std::cerr<<"info="<<info<<"\n";
            throw runtime_error("diag: zheev_: failed with info!=0.\n");
        }

        const int NB = 256;
        lwork = std::max(1 + static_cast<int>(std::real(work[0])), (NB + 2)*n);
        work.resize(lwork);
        // real work:
        zheev_(&jobz,&uplo,&n,&(Z(0,0)),&lda,&(eigs[0]),&(work[0]),&lwork,&(rwork[0]),&info);
        if (info!=0) {
            std::cerr<<"info="<<info<<"\n";
            throw runtime_error("diag: zheev: failed with info!=0.\n");
        }

        return eigs[0];
    }

    // -------------------------------------------------------
    double TriDiag(char option, vector<double>& alpha,
                   vector<double>& beta, Matrix<double>& Z,
                   Eigen::VectorXd &D){
        char jobz=option;
        int N=alpha.size();
        int LDZ = N*1;
        int Nwork= 2*N - 2;
        int info= -1;
        Z.resize(LDZ,N);
        Eigen::VectorXd E, work;
        work.resize(Nwork);

        D.resize(N); E.resize(N-1);
        D[0] = alpha[0];
        for(int i = 1; i<N; i++) {
            assert(i<alpha.size());
            D(i) = alpha[i];
            E(i-1) = beta[i];
        }

        // ---- Diagonalize the Tri-diag matrix --------------------
        // dstev_(char *,int *, double *, double *, double *, int *, double *, int *);
        dstev_(&jobz,&N,&(D(0)),&(E(0)),&(Z(0,0)),&LDZ,&(work(0)),&info);
        if (info!=0) {
            std::cerr<<"info="<<info<<"\n";
            perror("diag: dstev: failed with info!=0.\n");
        }

        //for (int i=0; i<N; i++) cout << D[i] << " \t " << endl;

        return D(0);
    }

    // -------------------------------------------------------
    inline void PrintVec(const std::vector<double>& V) {
        for (int i=0; i<V.size(); i++) {
            std::cout<<V[i]<<"\t";
        }
        std::cout<<std::endl;
    }

    // -------------------------------------------------------
    void Randomize(Eigen::VectorXd& V) {
        for (int i=0; i<V.size(); i++) {
            V(i)=-1.0 + variables_.grnd()*2.0;
        }
    }

    void Randomize(Eigen::VectorXcd& V) {
        //cout << variables_.grnd() << endl;
        //exit(1);
        for (int i=0; i<V.size(); i++) {
            double re = -1.0 + variables_.grnd()*2.0;
            double im = -1.0 + variables_.grnd()*2.0;
            dcomplex val(re,im);
            V(i)=val;
        }
    }

private:
    ConstVariables& variables_;
    QBasis& Basis_;
    Hamiltonian& Hamil_;
    int Nsite_,Nup_,Ndn_;
};

#endif //LANCZOSCOND_DYNLANCZOSSOLVER_H
