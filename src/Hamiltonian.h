//
// Created by Feng, Shi on 4/2/24.
// This header defines the Hamiltonian matrix
//

#ifndef LANCZOSCOND_HAMILTONIAN_H
#define LANCZOSCOND_HAMILTONIAN_H
#include "QuantumBasis.h"
#include "Matrix.h"
#include "Lattice.h"
#include "Memusage.h"
#include <omp.h>

typedef std::complex<double> dcomplex;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXcd VectorXcd;

class Hamiltonian {
public:
    Hamiltonian(ConstVariables& variables, Lattice& Lat, QBasis& Basis):
            variables_(variables), Basis_(Basis), Lat_(Lat),
            Nsite_(variables.NumberofSites)
    {
        SetupConnectors();
    }

    Matrix<int> Kxx_pair, Kyy_pair, Kzz_pair;
    std::vector<double> Kxx_Conn,Kyy_Conn,Kzz_Conn;
    std::vector<int> Sxx_init, Sxx_final;
    std::vector<int> Syy_init, Syy_final;
    std::vector<int> Szz_init, Szz_final;
    std::vector<dcomplex> HTSxx,HTSyy,HDiag;

    // -------------------------------------------------------
    void clear(){
        Kxx_Conn.resize(0); Kyy_Conn.resize(0); Kzz_Conn.resize(0);
        Sxx_init.clear(); Syy_init.clear(); Szz_init.clear();
        Sxx_final.clear(); Syy_final.clear(); Szz_final.clear();
        HTSxx.clear(); HTSyy.clear(); HDiag.clear();
        Kxx_pair.clear(); Kyy_pair.clear(); Kzz_pair.clear();
    }

    // -------------------------------------------------------
    void SetupConnectors(){
        clear();
        if (variables_.Model=="HeisenbergChain") {
            HeisenbergChain_Connectors();
        } else if (variables_.Model=="Kitaev") {
            Connectors(); // Build Connectors in Single-particle states
        } else {
            std::cerr<<"Model="<<variables_.Model<<"\n";
            throw std::string("Unknown Model parameter \n");
        }
    }

    // -------------------------------------------------------
    void RIXS_killcenter(int& center){


        cout << " ==== RIXS - Intermediate Hamiltonian ==== " << endl;
        int Ksize=Kxx_Conn.size();
        for(int ii=0; ii<Ksize;ii++){
            int i = Kxx_pair(ii,0);
            int j = Kxx_pair(ii,1);
            if(i==center || j==center) {
                Kxx_Conn[ii] = 0;
            }
            cout << i << " \t " << j << " \t " << Kxx_Conn[ii] << endl;
        }

        Ksize=Kyy_Conn.size();
        for(int ii=0; ii<Ksize;ii++){
            int i = Kyy_pair(ii,0);
            int j = Kyy_pair(ii,1);
            if(i==center || j==center) {
                Kyy_Conn[ii] = 0;
            }
            cout << i << " \t " << j << " \t " << Kyy_Conn[ii] << endl;
        }

        Ksize=Kzz_Conn.size();
        for(int ii=0; ii<Ksize;ii++){
            int i = Kzz_pair(ii,0);
            int j = Kzz_pair(ii,1);
            if(i==center || j==center) {
                Kzz_Conn[ii] = 0;
            }
            cout << i << " \t " << j << " \t " << Kzz_Conn[ii] << endl;
        }

    }

    // -------------------------------------------------------
    void TightB_Ham(){
        std::cout << "create Tight-Binding/Off-Diag Hamiltonian" << std::endl;
        int Hsize = Basis_.basis.size();
        std::cout << "	hilbert-space size:  "<< Hsize << std::endl;

        double vm, rss;
        process_mem_usage(vm, rss);
        std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;

        // allocate diagonal term!
        dcomplex izero=(0.0,0.0);
        std::fill(HTSxx.begin(),HTSxx.end(),izero);
        std::fill(HTSyy.begin(),HTSyy.end(),izero);

        process_mem_usage(vm, rss);
        std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;

        // Sxi Sx_j |Psi> = distrop down and create up
        // (create down remove up)_i (create up remove down)_j
        for(int bond=0;bond<Kxx_Conn.size();bond++) {
            int si=Kxx_pair(bond,0);
            int sj=Kxx_pair(bond,1);
            if(Kxx_Conn.size()==1) break;
            assert(si!=sj);
            for(int i1=0; i1<Hsize; i1++) {
                int ket = Basis_.basis[i1];

                int keto1,keto2;
                std::complex<double> Sxi,Sxj;
                Basis_.ApplySx(si,ket,keto1,Sxi);
                Basis_.ApplySx(sj,keto1,keto2,Sxj);

                //cout << Sxi << " \t " << Sxj << " \t " << Sxi*Sxj << endl;
                Sxx_init.push_back(ket);
                Sxx_final.push_back(keto2);
                HTSxx.push_back(Sxi*Sxj*Kxx_Conn[bond]);

            }
        }

        process_mem_usage(vm, rss);
        std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;


        // Syi Sy_j |Psi> = distrop down and create up
        // (create down remove up)_i (create up remove down)_j
        for(int bond=0;bond<Kyy_Conn.size();bond++) {
            int si=Kyy_pair(bond,0);
            int sj=Kyy_pair(bond,1);
            if(Kyy_Conn.size()==1) break;
            assert(si!=sj);
            for(int i1=0; i1<Hsize; i1++) {
                int ket = Basis_.basis[i1];

                int keto1,keto2;  // the output kets
                std::complex<double> Syi,Syj;  // coefs
                Basis_.ApplySy(si,ket,keto1,Syi);
                Basis_.ApplySy(sj,keto1,keto2,Syj);

                //cout << Syi << " \t " << Syj << " \t " << Syi*Syj << endl;
                Syy_init.push_back(ket);
                Syy_final.push_back(keto2);
                HTSyy.push_back(Syi*Syj*Kyy_Conn[bond]);
            }
        }

        process_mem_usage(vm, rss);
        std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;



        // --- ADD Magnetic Field in spin-x and spin-y direction
        // x-direction
        for(int i1=0; i1<Hsize; i1++) {
            int ket = Basis_.basis[i1];
            for(int site=0;site<Nsite_;site++) {
                int keto1=-1;
                //int bit = int(findBit(ketin,site));
                keto1 = Basis_.flipBit(ket,site);
                dcomplex coef=std::complex<double>(0.5,0);; //0.5+0i;
                Sxx_init.push_back(keto1);
                Sxx_final.push_back(ket);
                HTSxx.push_back(coef*variables_.Bxx);

            }
        }

        process_mem_usage(vm, rss);
        std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;


        // y-direction
        for(int i1=0; i1<Hsize; i1++) {
            int ket = Basis_.basis[i1];
            for(int site=0;site<Nsite_;site++) {
                int keto1=-1;
                dcomplex coef=std::complex<double>(0,0); //0.+0.i;
                Basis_.ApplySy(site,ket,keto1,coef);
                dcomplex magy = variables_.Byy*coef;
                Syy_init.push_back(keto1);
                Syy_final.push_back(ket);
                HTSyy.push_back(magy);

            }
        }

        process_mem_usage(vm, rss);
        std::cout << "       VM (MB): " << int(vm/1024) << "       RSS (MB): " << int(rss/1024) << std::endl;


        // z-direction ---- remember zz direction is diagonal - not part of off diagonal terms!

    } // end function

    // -------------------------------------------------------
    double findsign(int newstate,int hopp_Fr,int hopp_To){
        int particles=0;

        if(abs(hopp_Fr-hopp_To)==0 || abs(hopp_Fr-hopp_To)==1) {
            return 1.0;
        }

        if(hopp_To<hopp_Fr){
            for(int pos=hopp_To+1;pos<hopp_Fr;pos++){
                if(Basis_.findBit(newstate,pos)==1) particles++;
            }
        }
        else {
            for(int pos=hopp_Fr+1;pos<hopp_To;pos++){
                if(Basis_.findBit(newstate,pos)==1) particles++;
            }
        }
        return pow(-1.0,particles);
    } // end function

    // -------------------------------------------------------
    void Connectors(){
        std::cout << "creating Connectors:" << std::endl;

        Matrix<double> SinglePart_Kxx, SinglePart_Kyy; // local
        Matrix<double> SinglePart_Kzz;
        SinglePart_Kxx.resize(Nsite_,Nsite_);
        SinglePart_Kyy.resize(Nsite_,Nsite_);
        SinglePart_Kzz.resize(Nsite_,Nsite_);

        for(int i=0;i<Nsite_;i++){ 	// ith site

            int j = Lat_.N1neigh_(i,0);
            if(i<j)  {
                int jx = Lat_.indx_(j);
                int jy = Lat_.indy_(j);
                assert(Lat_.Nc_(jx,jy)!=-1);
                SinglePart_Kzz(i,j) = variables_.Kzz;
            }

            j = Lat_.N1neigh_(i,2);
            if(i<j)  {
                int jx = Lat_.indx_(j);
                int jy = Lat_.indy_(j);
                assert(Lat_.Nc_(jx,jy)!=-1);
                SinglePart_Kxx(i,j) = variables_.Kxx;
            }

            j = Lat_.N1neigh_(i,1);
            if(i<j)  {
                int jx = Lat_.indx_(j);
                int jy = Lat_.indy_(j);
                assert(Lat_.Nc_(jx,jy)!=-1);
                SinglePart_Kyy(i,j) = variables_.Kyy;
            }
        }




        int Kxxbonds=SinglePart_Kxx.numNonZeros();
        int Kyybonds=SinglePart_Kyy.numNonZeros();
        int Kzzbonds=SinglePart_Kzz.numNonZeros();

        cout << " Single particle Kitaev - xx " << endl;
        SinglePart_Kxx.print();

        cout << " Single particle Kitaev - yy " << endl;
        SinglePart_Kyy.print();

        cout << " Single particle Kitaev - zz " << endl;
        SinglePart_Kzz.print();


        if(Kxxbonds==0) Kxxbonds=1;
        if(Kyybonds==0) Kyybonds=1;
        if(Kzzbonds==0) Kzzbonds=1;
        Kxx_pair.resize(Kxxbonds,2); Kyy_pair.resize(Kyybonds,2); Kzz_pair.resize(Kzzbonds,2);
        Kxx_Conn.resize(Kxxbonds);   Kyy_Conn.resize(Kyybonds);   Kzz_Conn.resize(Kzzbonds);

        int counter=0;
        for(int i=0;i<Nsite_;i++) {
            for(int j=0;j<Nsite_;j++) {
                if(SinglePart_Kxx(i,j)!=0.0) {
                    //assert(j>i);
                    Kxx_pair(counter,0) = i;
                    Kxx_pair(counter,1) = j;
                    Kxx_Conn[counter] = SinglePart_Kxx(i,j);
                    counter++;
                }
            }
        }

        counter=0;
        for(int i=0;i<Nsite_;i++) {
            for(int j=0;j<Nsite_;j++) {
                if(SinglePart_Kyy(i,j)!=0.0) {
                    //assert(j>i);
                    Kyy_pair(counter,0) = i;
                    Kyy_pair(counter,1) = j;
                    Kyy_Conn[counter] = SinglePart_Kyy(i,j);
                    counter++;
                }
            }
        }

        counter=0;
        for(int i=0;i<Nsite_;i++) {
            for(int j=0;j<Nsite_;j++) {
                if(SinglePart_Kzz(i,j)!=0.0) {
                    //assert(j>i);
                    Kzz_pair(counter,0) = i;
                    Kzz_pair(counter,1) = j;
                    Kzz_Conn[counter] = SinglePart_Kzz(i,j);
                    counter++;
                }
            }
        }

    } // end function

    // -------------------------------------------------------
    void Diagonal(){
        std::cout << "create Diagonal matrix elements" << std::endl;
        int Hsize = Basis_.basis.size();
        std::cout << "	hilbert-space size:  "<< Hsize << std::endl;

        // allocate diagonal term!
        HDiag.resize(Hsize);
        dcomplex izero=(0.0,0.0);
        std::fill(HDiag.begin(),HDiag.end(),izero);

        // Szi Sz_j |Psi> = distrop down and create up
        // (create down remove up)_i (create up remove down)_j

        for(int bond=0;bond<Kzz_Conn.size();bond++) {
            int si=Kzz_pair(bond,0);
            int sj=Kzz_pair(bond,1);
            if(Kzz_Conn.size()==1) break;
            assert(si!=sj);
#pragma omp parallel for
            for(int i1=0; i1<Hsize; i1++) {
                int ket = Basis_.basis[i1];
                int keto1,keto2;
                dcomplex Szi,Szj;

                Basis_.ApplySz(si,ket,keto1,Szi); assert(ket==keto1);
                Basis_.ApplySz(sj,keto1,keto2,Szj); assert(keto1==keto2);

                HDiag[i1] += Kzz_Conn[bond]*Szi*Szj;
            }
        }


        // --- ADD Magnetic Field in spin-z-direction
#pragma omp parallel for
        for(int i1=0; i1<Hsize; i1++) {
            int ket = Basis_.basis[i1];
            dcomplex SzT=0;

            for(int site=0;site<Nsite_;site++) {
                dcomplex Szi; int keto1;
                Basis_.ApplySz(site,ket,keto1,Szi); assert(ket==keto1);
                SzT+=Szi;
            }
            HDiag[i1] += variables_.Bzz*SzT;
        }

    } // end function



    // -------------------------------------------------------
    void HeisenbergChain_Connectors(){
        std::cout << "creating lattice and Tight-Binding Connectors:" << std::endl;

        Matrix<double> SinglePart_Kxx, SinglePart_Kyy; // local
        Matrix<double> SinglePart_Kzz;
        SinglePart_Kxx.resize(Nsite_,Nsite_);
        SinglePart_Kyy.resize(Nsite_,Nsite_);
        SinglePart_Kzz.resize(Nsite_,Nsite_);

        // Neighbors for each site
        for(int i=0;i<Nsite_;i++){ 	// ith site

            int j = Lat_.N1neigh_(i,0);
            if(i<j)  {
                int jx = Lat_.indx_(j);
                int jy = Lat_.indy_(j);
                assert(Lat_.Nc_(jx,jy)!=-1);
                SinglePart_Kxx(i,j) = variables_.Kxx;
                SinglePart_Kyy(i,j) = variables_.Kyy;
                SinglePart_Kzz(i,j) = variables_.Kzz;
            }

            j = Lat_.N1neigh_(i,1);
            if(i<j)  {
                int jx = Lat_.indx_(j);
                int jy = Lat_.indy_(j);
                assert(Lat_.Nc_(jx,jy)!=-1);
                SinglePart_Kxx(i,j) = variables_.Kxx;
                SinglePart_Kyy(i,j) = variables_.Kyy;
                SinglePart_Kzz(i,j) = variables_.Kzz;
            }

        }

        int Kxxbonds=SinglePart_Kxx.numNonZeros();
        int Kyybonds=SinglePart_Kyy.numNonZeros();
        int Kzzbonds=SinglePart_Kzz.numNonZeros();

        cout << " Single particle Heisenberg Chain - xx " << endl;
        SinglePart_Kxx.print();

        cout << " Single particle Heisenberg Chain - yy " << endl;
        SinglePart_Kyy.print();

        cout << " Single particle Heisenberg Chain - zz " << endl;
        SinglePart_Kzz.print();


        if(Kxxbonds==0) Kxxbonds=1;
        if(Kyybonds==0) Kyybonds=1;
        if(Kzzbonds==0) Kzzbonds=1;
        Kxx_pair.resize(Kxxbonds,2); Kyy_pair.resize(Kyybonds,2); Kzz_pair.resize(Kzzbonds,2);
        Kxx_Conn.resize(Kxxbonds);   Kyy_Conn.resize(Kyybonds);   Kzz_Conn.resize(Kzzbonds);

        int counter=0;
        for(int i=0;i<Nsite_;i++) {
            for(int j=0;j<Nsite_;j++) {
                if(SinglePart_Kxx(i,j)!=0.0) {
                    //assert(j>i);
                    Kxx_pair(counter,0) = i;
                    Kxx_pair(counter,1) = j;
                    Kxx_Conn[counter] = SinglePart_Kxx(i,j);
                    counter++;
                }
            }
        }

        counter=0;
        for(int i=0;i<Nsite_;i++) {
            for(int j=0;j<Nsite_;j++) {
                if(SinglePart_Kyy(i,j)!=0.0) {
                    //assert(j>i);
                    Kyy_pair(counter,0) = i;
                    Kyy_pair(counter,1) = j;
                    Kyy_Conn[counter] = SinglePart_Kyy(i,j);
                    counter++;
                }
            }
        }

        counter=0;
        for(int i=0;i<Nsite_;i++) {
            for(int j=0;j<Nsite_;j++) {
                if(SinglePart_Kzz(i,j)!=0.0) {
                    //assert(j>i);
                    Kzz_pair(counter,0) = i;
                    Kzz_pair(counter,1) = j;
                    Kzz_Conn[counter] = SinglePart_Kzz(i,j);
                    counter++;
                }
            }
        }

    } // end function


    // -------------------------------------------------------
    // Matrix Vector Multiplication H|vin> = |vout>
    Eigen::VectorXcd MatrixVectorMult(Eigen::VectorXcd& Vin) {
        if(Vin.size()!=HDiag.size()) {
            string errorout ="LanczosSolver.h::Dot-Product size mismatch!";
            throw std::invalid_argument(errorout);
        }

        int nthreads = variables_.Threads;
        int hilbert=Vin.size();
        vector<Eigen::VectorXcd> threadVout(nthreads);

        Eigen::VectorXcd Vout(hilbert);
        Vout.setZero();
        //Vout.clear();
        //Vout = Eigen::VectorXcd::Zero(hilbert); // "Just incase" - set all elements to zero

        for(int i=0;i<variables_.Threads;i++) {
            threadVout[i].resize(hilbert);
            threadVout[i].setZero();
        }

        // Diagonal Hamiltonian terms!
#pragma omp parallel for num_threads(nthreads)
        for(int i=0;i<hilbert;i++) {
            dcomplex Hij=HDiag[i];
            //assert(Hij.imag()==0);
            int tid = omp_get_thread_num();
            threadVout[tid](i) += Hij*Vin(i);
            //Vout(i) += Hij*Vin(i);
        }

        int hilbert_t=HTSxx.size();
        // Kxx Kitaev - Sector -------
#pragma omp parallel for num_threads(nthreads)
        for(int i=0;i<hilbert_t;i++){
            int Hi=Sxx_init[i];
            int Hj=Sxx_final[i];
            dcomplex Hij=HTSxx[i];
            assert(Hi<hilbert && Hj<hilbert);
            //assert(Hij.imag()==0);
            int tid = omp_get_thread_num();
            threadVout[tid](Hi) += Hij*Vin(Hj);
            //Vout(Hi) += Hij*Vin(Hj);
            //			Vout[Hi] += Hij.real()*Vin[Hj];
        }

        hilbert_t=HTSyy.size();
        // Kyy Kitaev - Sector -------
#pragma omp parallel for num_threads(nthreads)
        for(int i=0;i<hilbert_t;i++){
            int Hi=Syy_init[i];
            int Hj=Syy_final[i];
            dcomplex Hij=HTSyy[i];
            assert(Hi<hilbert && Hj<hilbert);
            //assert(Hij.imag()==0);
            int tid = omp_get_thread_num();
            //int tmp = omp_get_num_threads(); //omp_get_num_procs();
            threadVout[tid](Hi) += Hij*Vin(Hj);
            //Vout(Hi) += Hij*Vin(Hj);
            //cout << Hij << " \t " << Vin(Hj) << " \t " << Hij*Vin(Hj) << endl;
        }
#pragma omp barrier

        Vout.setZero();
#pragma omp parallel for num_threads(nthreads)
        for(int h=0;h<hilbert;h++) {
            for(int i=0;i<nthreads;i++) {
                Vout(h) += threadVout[i](h);
            }
        }

        //		#pragma omp parallel for num_threads(nthreads)
        //		for(int h=0;h<8;h++) {
        //			cout << h << " - " << omp_get_thread_num() << endl;
        //		}

#pragma omp barrier
        return Vout;
    }


private:
    ConstVariables& variables_;
    Lattice& Lat_;
    QBasis& Basis_;
    std::vector<double> Conn, TJH_Conn, Pair_Conn, TJHSz_Conn;
    int Nsite_,Nup_,Ndn_;
};
#endif //LANCZOSCOND_HAMILTONIAN_H
