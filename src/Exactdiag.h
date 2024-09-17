//
// Created by Feng, Shi on 4/2/24.
//

#ifndef LANCZOSCOND_EXACTDIAG_H
#define LANCZOSCOND_EXACTDIAG_H
#include "Observables.h"

typedef Observables ObservType;
typedef std::complex<double> dcomplex;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXcd VectorXcd;

extern "C" void zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
                       std::complex<double> *,int *, double *, int *);

// ===== define functions ==========
Eigen::VectorXcd ReturnRow(Matrix<dcomplex>& mat, int row);
Eigen::VectorXcd ReturnCol(Matrix<dcomplex>& mat, int col);
VectorXf ReturnRow(Matrix<double>& mat, int row);
VectorXf ReturnCol(Matrix<double>& mat, int col);

Matrix<dcomplex> SpinSpinCorr(ObservType& Observ_, Eigen::VectorXcd& Psi, int& Nsite_);
Eigen::MatrixXcd ExactDiag(ConstVariables& variables_, QBasis& Basis_, Hamiltonian& Hamil_,
                           ObservType& Observ_, Eigen::VectorXd& D);
//void Diagonalize(char option, Matrix<dcomplex>& A, vector<double>& eval);
void Diagonalize(char option, Eigen::MatrixXcd& A, Eigen::VectorXd& eval);
void FiniteTemperature(ConstVariables& variables_, QBasis& Basis_,
                       Hamiltonian& Hamil_, ObservType& Observ_,
                       Matrix<dcomplex>& Ham,
                       Matrix<dcomplex>& evec,
                       vector<double>& eval);

// ================================================
// --------- distribute dynamics ------------------
Eigen::MatrixXcd ExactDiag(ConstVariables& variables_, QBasis& Basis_,
                           Hamiltonian& Hamil_, ObservType& Observ_,
                           Eigen::VectorXd& D) {

    std::cout << "Begin Exact Diagonalization! --- "<<std::endl;
    int N = Basis_.basis.size();
    Eigen::MatrixXcd Ham(N,N);
    D.resize(N);

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
    //assert(Ham.IsHermitian());
    //Ham.print();

    Eigen::MatrixXcd Evec = Ham;
    Diagonalize('V',Evec,D);
    for(int i=0;i<D.size();i++) cout << D[i] << ", ";


    cout << " Ground-state energy:= " << setprecision(16) << D[0] << endl;
    //for (int i=0; i<D.size(); i++) cout << D[i] << " \t " << endl;
    return Evec;
}


void Diagonalize(char option, Eigen::MatrixXcd& A, Eigen::VectorXd& eval){
    char jobz=option;
    char uplo='U';
    int n=A.rows();
    int lda=A.cols();
    assert(n==lda);

    eval.resize(n);
    eval.fill(0);
    Eigen::VectorXcd work(3);
    Eigen::VectorXd rwork(3*n);
    int info,lwork= -1;

    // ---- spin up part of the Hamiltonian --------------------
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


// ================================================
// --- return ith row of the matrix;

Eigen::VectorXcd ReturnRow(Matrix<dcomplex>& mat, int row) {
    int N = mat.n_col();
    Eigen::VectorXcd out(N); out.setZero();
    for(int i=0; i<N;i++) out(i) = mat(row,i);
    //VectorXcd out1=out;
    return out;
}

VectorXf ReturnRow(Matrix<double>& mat, int row) {
    int N = mat.n_col();
    VectorXf out(N); out.setZero();
    for(int i=0; i<N;i++) out[i] = mat(row,i);
    return out;
}

Eigen::VectorXcd ReturnCol(Matrix<dcomplex>& mat, int col) {
    int N = mat.n_col();
    Eigen::VectorXcd out(N); out.setZero();
    for(int i=0; i<N;i++) out(i) = mat(i,col);
    return out;
}

VectorXf ReturnCol(Matrix<double>& mat, int col) {
    int N = mat.n_col();
    VectorXf out(N); out.setZero();
    for(int i=0; i<N;i++) out[i] = mat(i,col);
    return out;
}

#endif //LANCZOSCOND_EXACTDIAG_H
