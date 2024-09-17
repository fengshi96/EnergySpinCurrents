#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <random>
#include <complex>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
//#include "Vector.h"
#include "Memusage.h"
#include "Matrix.h"
#include "ParametersEngine.h"
#include "Lattice.h"
#include "QuantumBasis.h"
#include "Hamiltonian.h"
#include "Exactdiag.h"
#include "Observables.h"
#include "dynamics.h"
//#include "FiniteTemp.h"

typedef Eigen::VectorXf VectorXf;
typedef ConstVariables ConstVariablestype;
typedef QBasis Basistype;
typedef Hamiltonian HamilType;
typedef DynLanczosSolver DynLancType;
typedef Observables ObservType;

typedef std::complex<double> dcomp;    // your typedef
void PrintGSSz(Eigen::VectorXcd& Psi, Basistype& Basis, int& Nsite_);
void PrintGSConf(Eigen::VectorXcd& Psi, Basistype& Basis, int& Nsite_);

int main(int argc, char *argv[]) {
    std::cout << "Hello, World!" << std::endl;
    if (argc!=2) {
        throw std::invalid_argument("USE:: executable inputfile");
    }
    string inputfile = argv[1];
    cout << argv[1] << endl;
    double pi=acos(-1.0);
    double vm, rss;
    dcomplex zero(0.0,0.0);

    // Read from the inputfile
    ConstVariablestype Parameters(inputfile);
    omp_set_dynamic(0);
    omp_set_num_threads(Parameters.Threads);
    Lattice Lat(Parameters);
    int Nsite_ = Parameters.NumberofSites;

    // Build binary basis
    Basistype Basis(Parameters);
    int nn=Basis.basis.size();
    process_mem_usage(vm, rss);
    std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;

    // Build non-zero Diagonal and Tight-binding part of Hamiltonian
    HamilType Hamiltonian(Parameters,Lat,Basis);
    Hamiltonian.TightB_Ham();
    process_mem_usage(vm, rss);
    std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;

    Hamiltonian.Diagonal();
    process_mem_usage(vm, rss);
    std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;

    ObservType Observ(Parameters, Lat, Basis, Hamiltonian);
    DynLancType Lanczos(Parameters, Basis, Hamiltonian);
    Eigen::VectorXcd Psi;

    if(Parameters.Solver=="ED") {
        Eigen::VectorXd eval;
        Eigen::MatrixXcd Evec = ExactDiag(Parameters, Basis, Hamiltonian, Observ, eval);
        Psi = Evec.col(0);
        Lanczos.fillFromED(Evec,eval);
    } else {
        Psi = Lanczos.Lanczos_Nirav(Parameters.LancType);
        cout << setprecision(6) << fixed;
        cout << " Ritz values: ";
        for(int i=0;i<Lanczos.TriDeval.size();i++) cout << Lanczos.TriDeval[i] << " ";
        cout << endl;
    }

    cout << setprecision(8) << fixed;


    Eigen::VectorXcd Sxi, Syi, Szi, Spi, Smi;
    Observ.measureLocalS(Psi, Sxi, Syi, Szi, Spi, Smi);
    Observ.measureLocalJe(Psi);
    if (Parameters.LocalSkw){
        cout << " ==================================================================== " << endl;
        cout << "                   Spin Dynamical Correlations		                   " << endl;
        cout << " ==================================================================== " << endl;
        string str = "Spin Current";
        StartDynamics(str,Parameters,Lat,Basis,Hamiltonian,Lanczos,Psi);
    }

    if (Parameters.EnergyCurrDynamics){
        cout << " ==================================================================== " << endl;
        cout << "              Energy Current Dynamical Correlations				   " << endl;
        cout << " ==================================================================== " << endl;
        string str1 = "Energy Current";
        StartDynamics(str1,Parameters,Lat,Basis,Hamiltonian,Lanczos,Psi);
    }

//    Observ.measureLocalJs(Psi);
//    Observ.measureLocalJe(Psi);
//
//    // ------------ Spin-Spin correlations ---------------------
//    std::vector<Eigen::MatrixXcd> A(9);
//    A[0] = Observ.TwoPointCorr("Sx","Sx",Psi); cout << A[0] << endl << endl;
//    A[1] = Observ.TwoPointCorr("Sx","Sy",Psi); cout << A[1] << endl << endl;
//    A[2] = Observ.TwoPointCorr("Sx","Sz",Psi); cout << A[2] << endl << endl;
//    A[3] = Observ.TwoPointCorr("Sy","Sx",Psi); cout << A[3] << endl << endl;
//    A[4] = Observ.TwoPointCorr("Sy","Sy",Psi); cout << A[4] << endl << endl;
//    A[5] = Observ.TwoPointCorr("Sy","Sz",Psi); cout << A[5] << endl << endl;
//    A[6] = Observ.TwoPointCorr("Sz","Sx",Psi); cout << A[6] << endl << endl;
//    A[7] = Observ.TwoPointCorr("Sz","Sy",Psi); cout << A[7] << endl << endl;
//    A[8] = Observ.TwoPointCorr("Sz","Sz",Psi); cout << A[8] << endl << endl;
//
//    cout << " Total Spin-Spin Correlations " << endl;
//    cout << A[0] + A[4] + A[8] << endl << endl;
//    cout << " --> Spin total = " << A[0].sum()   + A[4].sum()   + A[8].sum()   << endl;
//    cout << " --> Spin trace = " << A[0].trace() + A[4].trace() + A[8].trace() << endl << endl;

//    // ------------ Spin-Spin correlations ---------------------
//    std::vector<Eigen::MatrixXcd> A(9);
//    A[0] = Observ.TwoPointCorr("Sx","Sx",Psi); cout << A[0] << endl << endl;
//    A[1] = Observ.TwoPointCorr("Sx","Sy",Psi); cout << A[1] << endl << endl;
//    A[2] = Observ.TwoPointCorr("Sx","Sz",Psi); cout << A[2] << endl << endl;
//    A[3] = Observ.TwoPointCorr("Sy","Sx",Psi); cout << A[3] << endl << endl;
//    A[4] = Observ.TwoPointCorr("Sy","Sy",Psi); cout << A[4] << endl << endl;
//    A[5] = Observ.TwoPointCorr("Sy","Sz",Psi); cout << A[5] << endl << endl;
//    A[6] = Observ.TwoPointCorr("Sz","Sx",Psi); cout << A[6] << endl << endl;
//    A[7] = Observ.TwoPointCorr("Sz","Sy",Psi); cout << A[7] << endl << endl;
//    A[8] = Observ.TwoPointCorr("Sz","Sz",Psi); cout << A[8] << endl << endl;
//
//    cout << " Total Spin-Spin Correlations " << endl;
//    cout << A[0] + A[4] + A[8] << endl << endl;
//    cout << " --> Spin total = " << A[0].sum()   + A[4].sum()   + A[8].sum()   << endl;
//    cout << " --> Spin trace = " << A[0].trace() + A[4].trace() + A[8].trace() << endl << endl;
//
//    // ------------ Js-Js spin current correlations ---------------------
//    std::vector<Eigen::MatrixXcd> B(9);
//    B[0] = Observ.TwoPointCorr("Jsx","Jsx",Psi); cout << B[0] << endl << endl;
//    B[1] = Observ.TwoPointCorr("Jsx","Jsy",Psi); cout << B[1] << endl << endl;
//    B[2] = Observ.TwoPointCorr("Jsx","Jsz",Psi); cout << B[2] << endl << endl;
//    B[3] = Observ.TwoPointCorr("Jsy","Jsx",Psi); cout << B[3] << endl << endl;
//    B[4] = Observ.TwoPointCorr("Jsy","Jsy",Psi); cout << B[4] << endl << endl;
//    B[5] = Observ.TwoPointCorr("Jsy","Jsz",Psi); cout << B[5] << endl << endl;
//    B[6] = Observ.TwoPointCorr("Jsz","Jsx",Psi); cout << B[6] << endl << endl;
//    B[7] = Observ.TwoPointCorr("Jsz","Jsy",Psi); cout << B[7] << endl << endl;
//    B[8] = Observ.TwoPointCorr("Jsz","Jsz",Psi); cout << B[8] << endl << endl;
//
//    cout << " Total Js-Js spin current correlations " << endl;
//    cout << B[0] + B[4] + B[8] << endl << endl;
//    cout << " --> JSpin total = " << B[0].sum()   + B[4].sum()   + B[8].sum()   << endl;
//    cout << " --> JSpin trace = " << B[0].trace() + B[4].trace() + B[8].trace() << endl << endl;
//
//    // ------------ Je-Je energy current correlations ---------------------
//    std::vector<Eigen::MatrixXcd> C(9);
//    C[0] = Observ.TwoPointCorr("Jex","Jex",Psi); cout << C[0] << endl << endl;
//    C[1] = Observ.TwoPointCorr("Jex","Jey",Psi); cout << C[1] << endl << endl;
//    C[2] = Observ.TwoPointCorr("Jex","Jez",Psi); cout << C[2] << endl << endl;
//    C[3] = Observ.TwoPointCorr("Jey","Jex",Psi); cout << C[3] << endl << endl;
//    C[4] = Observ.TwoPointCorr("Jey","Jey",Psi); cout << C[4] << endl << endl;
//    C[5] = Observ.TwoPointCorr("Jey","Jez",Psi); cout << C[5] << endl << endl;
//    C[6] = Observ.TwoPointCorr("Jez","Jex",Psi); cout << C[6] << endl << endl;
//    C[7] = Observ.TwoPointCorr("Jez","Jey",Psi); cout << C[7] << endl << endl;
//    C[8] = Observ.TwoPointCorr("Jez","Jez",Psi); cout << C[8] << endl << endl;
//
//    cout << " Total Je-Je energy current Correlations " << endl;
//    cout << C[0] + C[4] + C[8] << endl << endl;
//    cout << " --> Jenergy total = " << C[0].sum()   + C[4].sum()   + C[8].sum()   << endl;
//    cout << " --> Jenergy trace = " << C[0].trace() + C[4].trace() + C[8].trace() << endl << endl;
//
//    if (Parameters.SpinCurrDynamics){
//        cout << " ==================================================================== " << endl;
//        cout << "               Spin Current Dynamical Correlations					   " << endl;
//        cout << " ==================================================================== " << endl;
//        string str1 = "Spin Current";
//        StartDynamics(str1,Parameters,Lat,Basis,Hamiltonian,Lanczos,Psi);
//    }
//
//    if (Parameters.EnergyCurrDynamics){
//        cout << " ==================================================================== " << endl;
//        cout << "              Energy Current Dynamical Correlations				   " << endl;
//        cout << " ==================================================================== " << endl;
//        string str1 = "Energy Current";
//        StartDynamics(str1,Parameters,Lat,Basis,Hamiltonian,Lanczos,Psi);
//    }

    return 0;
}

/*=======================================================================
 * ======================================================================
 * ======================================================================
*/

void PrintGSSz(Eigen::VectorXcd& Psi, Basistype& Basis, int& Nsite_) {
    int k_maxup = Basis.basis.size();
    int n=k_maxup;

    Eigen::VectorXcd V(Nsite_+1);
    for(int i0=0;i0<n;i0++){
        int ket = Basis.basis[i0];
        dcomplex coef1 = Psi[ket];
        int SzT = Basis.NPartInState(ket,Nsite_);
        V[SzT] += coef1; //*conj(coef1);
    }
    cout << V << endl;
}


void PrintGSConf(Eigen::VectorXcd& Psi, Basistype& Basis, int& Nsite_) {
    int k_maxup = Basis.basis.size();
    int n=k_maxup;

    vector<pair<double,int> >V;
    for(int i=0;i<n;i++){
        pair<double,int>P=make_pair((Psi[i]*conj(Psi[i])).real(),i);
        V.push_back(P);
    }
    sort(V.begin(),V.end());

    for(int i0=n-1;i0>=n-10;i0--){
        int i = V[i0].second;
        int basisLabel1 = i; // i1*k_maxdn + i2;
        int ket = Basis.basis[i];
        double coef = V[i0].first; //Psi[basisLabel1];
        dcomplex coef1 = Psi[basisLabel1];

        cout << basisLabel1 << " - ";
        //if(coef*coef>0.02) {
        for(int s=0;s<Nsite_;s++){
            cout << " \\underline{";
            if(Basis.findBit(ket,s) == 1) {
                cout << " \\uparrow ";
            } else {
                cout << " \\downarrow ";
            }
            cout << "} \\ \\ ";
            if(s==Nsite_-1) cout << " \\ \\ ";
        }
        cout << "& \\ \\ ";
        cout << coef << " \\\\ " << coef1 << endl;
    }
    cout << "   " << endl;

}
