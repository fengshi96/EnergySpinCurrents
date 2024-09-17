//
// Created by Feng, Shi on 4/2/24.
//

#ifndef LANCZOSCOND_QUANTUMBASIS_H
#define LANCZOSCOND_QUANTUMBASIS_H
#include "ParametersEngine.h"
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
//#include "Matrix.h"

using namespace std;
typedef std::size_t SizeType;
typedef std::complex<double> dcomplex;


class QBasis {
public:
    QBasis(ConstVariables& variables):
            variables_(variables), Nsite_(variables.NumberofSites)
    {
        Initialize();
    }

    std::vector<int> basis;
    int nHil_;

    /*  ***************************************
     *  ------------- Functions  --------------
     *  *************************************** */

    // create basis in binary representation
    // 0 = up
    // 1 = down
    void  Initialize(){
        //std::cout << "Building basis for up and down separatly: " << std::endl;
        nHil_ = pow(2,Nsite_);  // Size of Hilbert space= 2^(number of sites)
        basis.resize(nHil_);  // Initialize the set of all kets
        for(int ket=0; ket<pow(2,Nsite_); ket++) {
            //PrintBitwise(ket);
            // Initialize the set of all kets; e.g. basis[2]=2=00000010
            basis[ket] = ket;
        }
        std::cout << "number of ket basis: " << basis.size() << std::endl;
    } // ------------------

    void  ApplySz(const int& site, int& ketin, int& ketout, std::complex<double>& coef){
        // input: site, ketin
        // output: ketout, coef
        coef=std::complex<double>(0.0,0.0);
        ketout=-1;

        // Note: findBit(int i, int position) is designed to extract the state of
        // a specific bit (spin state) from an integer (i) where each bit represents a spin site
        int bit = int(findBit(ketin,site));
        coef = (bit==0) ? std::complex<double>(0.5,0.0) : std::complex<double>(-0.5,0.0);;
        ketout=ketin;
        assert(ketout>-1);
    } // ------------------

    void  ApplySx(const int& site, const int& ketin, int& ketout, std::complex<double>& coef){
        coef=std::complex<double>(0.5,0.0); //0.5+0i;
        ketout=-1;
        ketout = flipBit(ketin,site);
    } // ------------------

    void  ApplySy(const int& site, const int& ketin, int& ketout, std::complex<double>& coef){
        coef=std::complex<double>(0,0); //0.+0.i;
        ketout=-1;
        int bit = int(findBit(ketin,site));
        ketout = flipBit(ketin,site);
        coef = (bit==0) ? std::complex<double>(0.0,0.5): std::complex<double>(0.0,-0.5); //0.+0.5i : 0.-0.5i;
    } // ------------------

    Eigen::MatrixXi ABSplit(Eigen::VectorXi& S1sites, Eigen::VectorXi& S2sites){

        SizeType S1 = pow(2,S1sites.size());
        SizeType S2 = pow(2,S2sites.size());
        assert(S1sites.size() + S2sites.size() == Nsite_);
        Eigen::MatrixXi Labels(S1,S2);

        for(SizeType S1ket=0; S1ket<S1; S1ket++) {
            Eigen::VectorXi binList(Nsite_);

            for(SizeType i=0; i<S1sites.size(); i++) {
                int site = S1sites[i];
                assert(site<Nsite_);
                binList[site] = findBit(S1ket, i);
            } // - close i

            for(SizeType S2ket=0; S2ket<S2; S2ket++) {

                for(SizeType i=0; i<S2sites.size(); i++) {
                    int site = S2sites[i];
                    assert(site<Nsite_);
                    binList[site] = findBit(S2ket, i);
                } // - close i

                int fullHLabel = binVec2int(binList); // - must come after the i loop
                Labels(S1ket,S2ket) = fullHLabel;
            } // - close S2ket
        } // - close S1ket

        return Labels;
    } // ------------------


    /*  ********************************************
        Here we create a mask, apply the mask to n,
        and then right shift the masked value to get just
        the bit we want. We could write it out more fully as:
        *
        int mask =  1 << position;  			create mask
        int masked_pos = i & position;			apply mask to i
        int thebit = masked_pos >> position;	right shift the masked value to get bit at position
    */
    int findBit(const int i, const int position){
        return (i & ( 1 << position )) >> position;
    } // ---------------

    int flipBit(const int ketin, int position){
        return ketin^(1<<position);
    } // ---------------

    int hoppstate(int initial_state,int hopp_from,int hopp_to) {
        assert(findBit(initial_state,hopp_from)==1);
        assert(findBit(initial_state,hopp_to)==0);

        int mask = (1 << hopp_from) + (1 << hopp_to);
        return mask^initial_state;
    }

    void PrintBitwise(int i){
        for(int j=0;j<Nsite_;j++){
            //std::cout << ((i & ( 1 << j )) >> j) << "  ";
            std::cout << findBit(i,j) << "  ";
        }
        std::cout << std::endl;
    } // ---------------

    int binVec2int(Eigen::VectorXi inV) {
        SizeType length = inV.size();
        int out = 0;
        for(int i=0; i<length; i++) {
            int bin = inV[i];
            out += bin*pow(2,i);
        }
        return out;
    } // ---------------

    int NPartInState(int state, int Nsite_){
        int particles=0;
        for(int i=0; i<Nsite_; i++){
            particles += findBit(state,i);
        }
        return particles;
    }

private:
    ConstVariables& variables_;
    int Nsite_;
};
#endif //LANCZOSCOND_QUANTUMBASIS_H
