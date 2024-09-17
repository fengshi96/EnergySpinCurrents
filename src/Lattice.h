//
// Created by Feng, Shi on 4/2/24.
//

#ifndef LANCZOSCOND_LATTICE_H
#define LANCZOSCOND_LATTICE_H
#include <string>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Matrix.h"
#include "ParametersEngine.h"
using namespace std;

typedef ConstVariables ConstVariablestype;

class Lattice {
public:
    Lattice(ConstVariables& variables)
            : variables_(variables)
    {
        Initialize();
    }

    Matrix<int> Nc_,N1neigh_,N2neigh_;
    Eigen::VectorXi indx_,indy_;

    /*
     * ***********
     *  Functions in Class Coordinates ------
     *  ***********
    */
    void Initialize(){
        Nsite_ = variables_.NumberofSites;
        if(variables_.Model=="Kitaev"){
            Number1Neigh_=3;
            Number2Neigh_=3;
            N1neigh_.resize(Nsite_,Number1Neigh_);
            N2neigh_.resize(Nsite_,Number2Neigh_);
            N1neigh_.fill(-1);
            N2neigh_.fill(-2);
            Honeycomb1();

        } else if(variables_.Model=="HeisenbergChain"){
            Number1Neigh_=2;
            Number2Neigh_=2;
            N1neigh_.resize(Nsite_,Number1Neigh_);
            N2neigh_.resize(Nsite_,Number2Neigh_);
            N1neigh_.fill(-1);
            N2neigh_.fill(-2);
            Chain();

        }
    }
    // ---------------- chain ---------------------------
    void Chain() {
        std::cout << "creating Chain lattice " << std::endl;

        int LLX = Nsite_;
        int LLY = 1;
        // Site labeling
        indx_.resize(Nsite_);   indy_.resize(Nsite_);
        Nc_.resize(LLX,1);
        Nc_.fill(-1);


        int	counter = 0;
        for(int ix=0;ix<LLX;ix++){
            for(int iy=0;iy<LLY;iy++){
                Nc_(ix,iy) = counter;
                indx_(counter)=ix;
                indy_(counter)=iy;
                //print ix, iy, counter
                counter+=1;
            }
        }


        for(int i=0;i<Nsite_;i++){ 	// ith site
            int ix = indx_(i);
            int iy = indy_(i);

            // +x - 1neighbor 0
            if(ix+1<LLX)  {
                int jy = iy;
                int jx = ix+1;
                int j = Nc_(jx,jy);
                assert(j!=-1);
                N1neigh_(i,0) = j;
            }

            // -x - 1neighbor 1
            if(ix-1>-1)  {
                int jy = iy;
                int jx = ix-1;
                int j = Nc_(jx,jy);
                assert(j!=-1);
                N1neigh_(i,1) = j;
            }

            // +x - 2neighbor 0
            if(ix+2<LLX)  {
                int jy = iy;
                int jx = ix+2;
                int j = Nc_(jx,jy);
                assert(j!=-1);
                N2neigh_(i,0) = j;
            }

            // -x - 2neighbor 1
            if(ix-2>-1)  {
                int jy = iy;
                int jx = ix-2;
                int j = Nc_(jx,jy);
                assert(j!=-1);
                N2neigh_(i,1) = j;
            }


            // - Lx periodic -
            if(variables_.IsPeriodicX==true) {

                // +x - 1neighbor 0
                if(ix==LLX-1)  {
                    int jy = iy;
                    int jx = 0;
                    int j = Nc_(jx,jy);
                    assert(j!=-1);
                    N1neigh_(i,0) = j;
                }

                // -x - 1neighbor 1
                if(ix==0)  {
                    int jy = iy;
                    int jx = LLX-1;
                    int j = Nc_(jx,jy);
                    assert(j!=-1);
                    N1neigh_(i,1) = j;
                }

                // x+2 - 1neighbor 0
                if(ix==LLX-2)  {
                    int jy = iy;
                    int jx = 0;
                    int j = Nc_(jx,jy);
                    assert(j!=-1);
                    N2neigh_(i,0) = j;
                }

//				// x+2 - 1neighbor 0
//				if(ix==LLX-1)  {
//					int jy = iy;
//					int jx = 1;
//					int j = Nc_(jx,jy);
//					assert(j!=-1);
//					N2neigh_(i,0) = j;
//				}

//				// x-2 - 1neighbor 1
//				if(ix==0)  {
//					int jy = iy;
//					int jx = LLX-2;
//					int j = Nc_(jx,jy);
//					assert(j!=-1);
//					N2neigh_(i,1) = j;
//				}

//				// x-2 - 1neighbor 1
//				if(ix==1)  {
//					int jy = iy;
//					int jx = LLX-1;
//					int j = Nc_(jx,jy);
//					assert(j!=-1);
//					N2neigh_(i,1) = j;
//				}

            }
        }

        cout << " 1st Nearest neighbors " << endl;
        cout << " site x-neighbor y-neighbor z-neighbor \n";
        for(int i=0; i<Nsite_;i++)
            cout << i << " " << N1neigh_(i,2) << " "  << N1neigh_(i,1) << " "  << N1neigh_(i,0) << endl;

        cout << endl;

        cout << " 2nd Nearest neighbors " << endl;
        N2neigh_.print();
        cout << endl;

    } // end function

    // ---------------- Honeycomb ---------------------------
    void Honeycomb() {
        std::cout << "creating Honeycomb lattice " << std::endl;

        int LLX = variables_.Lx;
        int LLY = variables_.Ly;
        int SqLLX = LLX*2;
        int SqLLY = LLY;
        int SqNsite_=SqLLX*SqLLY;

        Matrix<int> SqNc(SqLLX,SqLLY);
        vector<int> Sqindx(SqNsite_),Sqindy(SqNsite_);
        SqNc.fill(-1);

        int counter=0;
        for(int ix=0;ix<SqLLX;ix++){
            for(int iy=0;iy<SqLLY;iy++){
                if(SqNc(ix,iy)==-1){
                    if(ix%2==1 and iy%4==0) {
                        SqNc(ix,iy) = 1;
                        Sqindx[counter] = ix;
                        Sqindy[counter] = iy;
                        counter+=1;
                        //print ix, iy

                        if(iy+3<SqLLY){
                            SqNc(ix,iy+3) = 1;
                            Sqindx[counter] = ix;
                            Sqindy[counter] = iy+3;
                            counter+=1;
                            //print ix, iy+3
                        }
                    }

                    if(ix%2==0 and iy%4==0) {

                        if(iy+1<SqLLY){
                            SqNc(ix,iy+1) = 1;
                            Sqindx[counter] = ix;
                            Sqindy[counter] = iy+1;
                            counter+=1;
                            //print ix, iy+1
                        }

                        if(iy+2<SqLLY){
                            SqNc(ix,iy+2) = 1;
                            Sqindx[counter] = ix;
                            Sqindy[counter] = iy+2;
                            counter+=1;
                            //print ix, iy+2
                        }
                    }

                }
            }
        }


        // Site labeling
        indx_.resize(Nsite_);   indy_.resize(Nsite_);
        Nc_.resize(SqLLX,SqLLY);


        counter = 0;
        for(int ix=0;ix<SqLLX;ix++){
            for(int iy=0;iy<SqLLY;iy++){
                Nc_(ix,iy) = -1;
                if(SqNc(ix,iy)!=-1) {
                    Nc_(ix,iy) = counter;
                    indx_(counter)=ix;
                    indy_(counter)=iy;
                    //print ix, iy, counter
                    counter+=1;
                }
            }
        }


        for(int i=0;i<Nsite_;i++){ 	// ith site
            int ix = indx_(i);
            int iy = indy_(i);

            // +y - z-bond (kitaev) - neighbor 0
            if(iy+1<SqLLY)  {
                int jy = iy+1;
                int jx = ix;
                if(Nc_(jx,jy)!=-1){
                    int j = Nc_(jx,jy);
                    N1neigh_(i,0) = j;
                }
            }

            if(iy-1>-1)  {
                int jy = iy-1;
                int jx = ix;
                if(Nc_(jx,jy)!=-1){
                    int j = Nc_(jx,jy);
                    N1neigh_(i,0) = j;
                }
            }


            // +x-y - x-bond (kitaev) - neighbor 1
            if(ix+1<SqLLX && iy-1>-1) {
                int jy = iy-1;
                int jx = ix+1;
                if(Nc_(jx,jy)!=-1) {
                    int j = Nc_(jx,jy);
                    N1neigh_(i,1) = j;
                }
            }

            if(ix-1>-1 && iy+1<SqLLY) {
                int jy = iy+1;
                int jx = ix-1;
                if(Nc_(jx,jy)!=-1) {
                    int j = Nc_(jx,jy);
                    N1neigh_(i,1) = j;
                }
            }


            // -x-y - y-bond (kitaev) - neighbor 2
            if(ix-1>-1 && iy-1>-1) {
                int jy = iy-1;
                int jx = ix-1;
                if(Nc_(jx,jy)!=-1) {
                    int j = Nc_(jx,jy);
                    N1neigh_(i,2) = j;
                }
            }

            if(ix+1<SqLLX && iy+1<SqLLY) {
                int jy = iy+1;
                int jx = ix+1;
                if(Nc_(jx,jy)!=-1) {
                    int j = Nc_(jx,jy);
                    N1neigh_(i,2) = j;
                }
            }




            // - Ly periodic -
            if(variables_.IsPeriodicY==true) {
                if(iy==LLY-1) {
                    int jy = 0;
                    int jx = ix;
                    if(Nc_(jx,jy)!=-1){
                        int j = Nc_(jx,jy);
                        N1neigh_(i,0) = j;
                    } else if (ix-1>-1) {
                        jx = ix-1;
                        int j = Nc_(jx,jy);
                        N1neigh_(i,0) = j;
                    } else if (ix==0) {
                        jx = SqLLX-1;
                        int j = Nc_(jx,jy);
                        N1neigh_(i,0) = j;
                    }
                }

                if(iy==0) {
                    int jy = LLY-1;
                    int jx = ix;
                    if(Nc_(jx,jy)!=-1){
                        int j = Nc_(jx,jy);
                        N1neigh_(i,0) = j;
                    } else if (ix+1<SqLLX) {
                        jx = ix+1;
                        //jx = (ix+1>SqLLx-1) ? 0:SqLLX-1;
                        int j = Nc_(jx,jy);
                        N1neigh_(i,0) = j;
                    } else if (ix==SqLLX-1) {
                        jx=0;
                        int j = Nc_(jx,jy);
                        N1neigh_(i,0) = j;
                    }
                }
            }


            // - Lx periodic -
            if(variables_.IsPeriodicX==true) {

                // - Sx bond -
                if(iy-1>-1 && ix==SqLLX-1) {
                    int jy = iy-1;
                    int jx = 0;
                    if(Nc_(jx,jy)!=-1) {
                        int j = Nc_(jx,jy);
                        N1neigh_(i,1) = j;
                    }
                }

                if(iy+1<SqLLY && ix==0) {
                    int jy = iy+1;
                    int jx = SqLLX-1;
                    if(Nc_(jx,jy)!=-1) {
                        int j = Nc_(jx,jy);
                        N1neigh_(i,1) = j;
                    }
                }

                // - Sy bond -
                if(ix==0 && iy-1>-1) {
                    int jy = iy-1;
                    int jx = SqLLX-1;
                    if(Nc_(jx,jy)!=-1) {
                        int j = Nc_(jx,jy);
                        N1neigh_(i,2) = j;
                    }
                }

                if(ix==SqLLX-1 && iy+1<SqLLY) {
                    int jy = iy+1;
                    int jx = 0;
                    if(Nc_(jx,jy)!=-1) {
                        int j = Nc_(jx,jy);
                        N1neigh_(i,2) = j;
                    }
                }
            }
        }

        cout << " 1st Nearest neighbors " << endl;
        N1neigh_.print();
        cout << endl;

        cout << " 2nd Nearest neighbors " << endl;
        N2neigh_.print();
        cout << endl;

    } // end function

    // ---------------- Honeycomb ---------------------------
    void Honeycomb1() {


        int LLX = variables_.Lx;
        int LLY = variables_.Ly/2;
        string shift="right"; // 'right'
        std::cout << "creating Honeycomb lattice: " << shift << " shifted " << std::endl;
        double scalex=2, scaley=4.0/sqrt(3.0);
        vector<double> t1(2), t2(2);
        t1[0] = 1.0*scalex;
        t1[1] = 0;

        if(shift=="left") {
            t2[0] = -0.5*scalex;
            t2[1] = sqrt(3.0)/2.0*scaley;
        } else if (shift=="right") {
            t2[0] = 0.5*scalex;
            t2[1] = sqrt(3.0)/2.0*scaley;
        }

        // Site labeling
        indx_.resize(Nsite_);
        indy_.resize(Nsite_);
        Nc_.resize(LLX*2+LLY,LLY*2);
        Nc_.fill(-1);

        int xv=0, counter=0;
        for(int i=0; i<LLX; i++){
            if (i==0) {xv = 0; } else {xv = xv+t1[0];}
            int x0 = xv;
            int y0 = 0;
            int x1, y1;
            if(shift=="right") {
                x1 = xv+1.0;  y1 = 1.0;
            } else if (shift=="left") {
                x1 = xv-1.0;  y1 = 1.0;
            }

            for (int j=0; j<LLY; j++) {
                int cxa = int(x0 + j*t2[0]), cxb = int(x1 + j*t2[0]);
                int cya = int(y0 + j*t2[1]), cyb = int(y1 + j*t2[1]);

                indx_[counter] = cxa;
                indy_[counter] = cya;
                Nc_(cxa,cya) = counter;
                counter++;

                indx_[counter] = cxb;
                indy_[counter] = cyb;
                Nc_(cxb,cyb) = counter;
                counter++;
            }
        }
        Nc_.print();


        int xmax = indx_.maxCoeff();
        int ymax = indy_.maxCoeff();
        int jx, jy;
        for(int i=0;i<Nsite_;i++){ 	// ith site
            int ix = indx_(i);
            int iy = indy_(i);

            // SxSx (kitaev) - neighbor 0
            jx = int(ix + 1);
            jy = int(iy + 1);
            if(jx<xmax+1 && jy<ymax+1 && Nc_(jx,jy)!=-1)  {
                int j = Nc_(jx,jy);
                N1neigh_(i,2) = j;
                N1neigh_(j,2) = i;
            }

            // SySy (kitaev) - neighbor 1
            jx = int(ix + 1);
            jy = int(iy - 1);
            if(jx<xmax+1 && jy<ymax and jy>=0 && Nc_(jx,jy)!=-1)  {
                int j = Nc_(jx,jy);
                N1neigh_(i,1) = j;
                N1neigh_(j,1) = i;
            }

            // SzSz (kitaev) - neighbor 2
            jx = int(ix + 0);
            jy = int(iy + 1);
            if(jx<xmax && jy<ymax+1 && Nc_(jx,jy)!=-1)  {
                int j = Nc_(jx,jy);
                N1neigh_(i,0) = j;
                N1neigh_(j,0) = i;
            }







            if(variables_.IsPeriodicY==true && shift=="right") {
                jx = int(ix - LLY);
                jy = 0;
                if(jx<xmax && iy==ymax && Nc_(jx,jy)!=-1) {
                    int j = Nc_(jx,jy);
                    N1neigh_(i,0) = j;
                    N1neigh_(j,0) = i;
                }
            } else if(variables_.IsPeriodicY==true && shift=="left") {
                jx = int(ix + LLY);
                jy = 0;
                if(jx<xmax+1 && iy==ymax and Nc_(jx,jy)!=-1) {
                    int j = Nc_(jx,jy);
                    N1neigh_(i,0) = j;
                    N1neigh_(j,0) = i;
                }
            }

            if(variables_.IsPeriodicX==true && shift=="right" && iy%2==0 && Nc_(ix,iy)<LLY*2) {
                jx = int(ix + LLX*2 - 1);
                jy = int(iy + 1);
                if(jx<xmax+1 && iy<ymax+1 && Nc_(jx,jy)!=-1) {
                    int j = Nc_(jx,jy);
                    N1neigh_(i,1) = j;
                    N1neigh_(j,1) = i;
                }
            } else if(variables_.IsPeriodicX==true && shift=="left" && iy%2==1 && Nc_(ix,iy)<LLY*2) {
                jx = int(ix + LLX*2 - 1);
                jy = int(iy - 1);
                if(jx<xmax+1 && iy<ymax+1 && Nc_(jx,jy)!=-1) {
                    int j = Nc_(jx,jy);
                    N1neigh_(i,2) = j;
                    N1neigh_(j,2) = i;
                }
            }
        }


        cout << " 1st Nearest neighbors " << endl;
        N1neigh_.print();
        cout << endl;

//		cout << " 2nd Nearest neighbors " << endl;
//		N2neigh_.print();
//		cout << endl;

    } // end function


private:
    ConstVariables& variables_;
    int Nsite_,Number1Neigh_,Number2Neigh_;
};
#endif //LANCZOSCOND_LATTICE_H
