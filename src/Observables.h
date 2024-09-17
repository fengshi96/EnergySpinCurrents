//
// Created by Feng, Shi on 4/2/24.
//

#ifndef LANCZOSCOND_OBSERVABLES_H
#define LANCZOSCOND_OBSERVABLES_H
#include "QuantumBasis.h"
#include "Lattice.h"
#include "Hamiltonian.h"

typedef std::complex<double> dcomplex;
typedef Eigen::VectorXf VectorXf;

extern "C" void   dstev_(char *,int *, double *, double *, double *, int *, double *, int *);
class Observables {
public:
    Observables(ConstVariables& variables, Lattice& Lat, QBasis& Basis, Hamiltonian& Hamil):
            variables_(variables), Lat_(Lat), Basis_(Basis), Hamil_(Hamil)
    {

    }

    // -----------------------------------
    Eigen::VectorXcd ApplyOp(Eigen::VectorXcd& inV, const string OpA, const int& site) {
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out;

        if (OpA == "Sz") {
            return ApplySz(inV, site);
        } else if (OpA == "Sy") {
            return ApplySy(inV, site);
        } else if (OpA == "Sx") {
            return ApplySx(inV, site);
        } else if (OpA == "Sp") {
            return ApplySp(inV, site);
        } else if (OpA == "Sm") {
            return ApplySm(inV, site);
        } else if (OpA == "Spe3") {
            return ApplySpe3(inV, site);
        } else if (OpA == "Sme3") {
            return ApplySme3(inV, site);
        } else if (OpA == "I") {
            return inV;
        } else if (OpA == "Jsx") {
            return ApplyJsx(inV, site);
        } else if (OpA == "Jsy") {
            return ApplyJsy(inV, site);
        // } else if (OpA == "Jsz") {
        //     return ApplyJsz(inV, site);
        } else if (OpA == "Jex") {
            return ApplyJex(inV, site);
        } else if (OpA == "Jey") {
            return ApplyJey(inV, site);
        // } else if (OpA == "Jez") {
        //     return ApplyJez(inV, site);
        } else if (OpA == "JexJeyJez") {
            return ApplyJexJeyJez(inV, site);
        } else if (OpA == "JsxJsyJsz") {
            return ApplyJsxJsyJsz(inV, site);
        } else if (OpA == "JexJeyJez") {
            return ApplySxSySz(inV, site);
        } else {
            std::cerr<<"Operator="<<OpA<<" does not exist \n";
            throw std::string("Unknown Operator label \n");
        }
    }

    // -----------------------------------
    void measureLocalJs(Eigen::VectorXcd& GS_) {
        int Nsite_ = variables_.NumberofSites;
        cout << "\n-------- measuring: site, <Js_xi>, <Js_yi>, <Js_zi>, <Js_total> --------" << endl;
        for(int si=0; si<Nsite_; si++) {
            Eigen::VectorXcd tmpx = ApplyJsx(GS_, si);
            Eigen::VectorXcd tmpy = ApplyJsy(GS_, si);
            // Eigen::VectorXcd tmpz = ApplyJsz(GS_, si);

            dcomplex ox = GS_.dot(tmpx);
            dcomplex oy = GS_.dot(tmpy);
            // dcomplex oz = GS_.dot(tmpz);

            cout << si << " \t " << ox << " \t " << oy << " \t " << " \t ";
            cout << ox+oy << endl;
        }
    }

    // -----------------------------------
    void measureLocalJe(Eigen::VectorXcd& GS_) {
        int Nsite_ = variables_.NumberofSites;
        cout << "\n-------- measuring: site, <Je_xi>, <Je_yi>, <Je_zi>, <Je_total> --------" << endl;
        for(int si=0; si<Nsite_; si++) {
            Eigen::VectorXcd tmpx = ApplyJex(GS_, si);
            Eigen::VectorXcd tmpy = ApplyJey(GS_, si);
            // Eigen::VectorXcd tmpz = ApplyJez(GS_, si);
            Eigen::VectorXcd tmptotal = ApplyJexJeyJez(GS_, si);

            dcomplex ox = GS_.dot(tmpx);
            dcomplex oy = GS_.dot(tmpy);
            // dcomplex oz = GS_.dot(tmpz);

            cout << si << " \t " << ox << " \t " << oy << " \t " << " \t ";
            cout << ox+oy << endl;
        }
    }

    // -----------------------------------
    void measureLocalS(Eigen::VectorXcd& GS_, Eigen::VectorXcd& Sxi,
                       Eigen::VectorXcd& Syi, Eigen::VectorXcd& Szi,
                       Eigen::VectorXcd& Spi, Eigen::VectorXcd& Smi) {
        int Nsite_ = variables_.NumberofSites;
        Sxi.resize(Nsite_); Syi.resize(Nsite_); Szi.resize(Nsite_);
        Spi.resize(Nsite_); Smi.resize(Nsite_);

        cout << "\n-------- measuring: site, <sx>, <sy>, <sz>, <spin_total>, <sp>, <sm> --------" << endl;
        for(int si=0; si<Nsite_; si++) {
            Eigen::VectorXcd tmpx = ApplySx(GS_,si);
            Eigen::VectorXcd tmpy = ApplySy(GS_,si);
            Eigen::VectorXcd tmpz = ApplySz(GS_,si);
            Sxi[si] = GS_.dot(tmpx);
            Syi[si] = GS_.dot(tmpy);
            Szi[si] = GS_.dot(tmpz);
            Spi[si] = Sxi[si] + dcomplex(0.0,1.0)*Syi[si];
            Smi[si] = Sxi[si] - dcomplex(0.0,1.0)*Syi[si];
            cout << si << " \t " << Sxi[si] << " \t " << Syi[si] << " \t " << Szi[si] << " \t "
                 << Sxi[si] + Syi[si] + Szi[si] << " \t "
                 << Spi[si] << " \t " << Smi[si] << endl;
        }
    }

    // -----------------------------------
    Eigen::MatrixXcd TwoPointCorr(const string OpA, const string OpB, Eigen::VectorXcd& GS_) {
        //cout << " -------- " << endl;
        std::cout << "Calculating " << OpA << "." << OpB << " correlations " << std::endl;
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();
        Eigen::MatrixXcd Corr(Nsite_,Nsite_);
        dcomplex dczero(0.0,0.0);

        //#pragma omp parallel for
        for(int sj=0; sj<Nsite_; sj++) {
            Eigen::VectorXcd out1(Hsize);
            out1.setZero();
            out1 = ApplyOp(GS_, OpB, sj);

#pragma omp parallel for
            for(int si=sj; si<Nsite_; si++) {
                Eigen::VectorXcd out2(Hsize);
                out2.setZero();
                out2 = ApplyOp(out1, OpA, si);
                Corr(si,sj) = GS_.dot(out2);
                Corr(sj,si) = Corr(si,sj);
            }
#pragma omp barrier
        }

        //Corr.print();
        return Corr;
    }

    // // -----------------------------------
    // Eigen::VectorXcd ApplyJsz(Eigen::VectorXcd& GS_, const int& si) {
    //     int Nsite_ = variables_.NumberofSites;
    //     assert(si<Nsite_);
    //     int xneigh = Lat_.N1neigh_(si,2); // - x neigh
    //     int yneigh = Lat_.N1neigh_(si,1); // - y neigh
    //     int zneigh = Lat_.N1neigh_(si,0); // - z neigh

    //     assert(zneigh<Nsite_); // z - neigh
    //     assert(yneigh<Nsite_); // y - neigh
    //     assert(xneigh<Nsite_); // x - neigh

    //     int Hsize = Basis_.basis.size();
    //     Eigen::VectorXcd outL1(Hsize); outL1.setZero();
    //     Eigen::VectorXcd outL2(Hsize); outL2.setZero();
    //     Eigen::VectorXcd outR1(Hsize); outR1.setZero();
    //     Eigen::VectorXcd outR2(Hsize); outR2.setZero();
    //     Eigen::VectorXcd outB1(Hsize); outB1.setZero();
    //     Eigen::VectorXcd outB2(Hsize); outB2.setZero();

    //     // Sx_i . Sy_i+y
    //     outL1 = ApplyOp(GS_, "Sy", yneigh);  // outL1 = Sy |GS>
    //     outL2 = ApplyOp(outL1, "Sx", si);  // outL2 = Sx (Sy |GS>) = Sx |outL1>
    //     outL2 = variables_.Kyy*outL2;

    //     // Sy_i . Sx_i+x
    //     outR1 = ApplyOp(GS_, "Sx", xneigh);
    //     outR2 = ApplyOp(outR1, "Sy", si);
    //     outR2 = variables_.Kxx*outR2;

    //     // Sy_i
    //     outB1 = ApplyOp(GS_, "Sy", si);
    //     outB1 = variables_.Bxx*outB1;

    //     // Sx_i
    //     outB2 = ApplyOp(GS_, "Sx", si);
    //     outB2 = variables_.Byy*outB2;

    //     return outL2 - outR2; // - outB1 + outB2;
    // }

    // ----------Spin Current---------------------
    Eigen::VectorXcd ApplyJsy(Eigen::VectorXcd& GS_, const int& si) {
        // Jsy along a2, or i+y; 
        // si are sites on B sublattice
        int Nsite_ = variables_.NumberofSites;
        assert(si<Nsite_);
        int xneigh = Lat_.N1neigh_(si,2); // - x neigh
        int yneigh = Lat_.N1neigh_(si,1); // - y neigh
        int zneigh = Lat_.N1neigh_(si,0); // - z neigh

        assert(zneigh<Nsite_); // z - neigh
        assert(yneigh<Nsite_); // y - neigh
        assert(xneigh<Nsite_); // x - neigh

        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd outL1(Hsize); outL1.setZero();
        Eigen::VectorXcd outL2(Hsize); outL2.setZero();
        Eigen::VectorXcd outR1(Hsize); outR1.setZero();
        Eigen::VectorXcd outR2(Hsize); outR2.setZero();
        Eigen::VectorXcd outB1(Hsize); outB1.setZero();
        Eigen::VectorXcd outB2(Hsize); outB2.setZero();

        // Sx_i . Sy_i+y
        outL1 = ApplyOp(GS_, "Sy", yneigh);
        outL2 = ApplyOp(outL1, "Sx", si);
        outL2 = variables_.Kyy*outL2;

        // Sy_i . Sx_i+y
        outR1 = ApplyOp(GS_, "Sx", yneigh);
        outR2 = ApplyOp(outR1, "Sy", si);
        outR2 = variables_.Kyy*outR2;

        // // Sz_i
        // outB1 = ApplyOp(GS_, "Sz", si);
        // outB1 = variables_.Bxx*outB1;

        // // Sx_i
        // outB2 = ApplyOp(GS_, "Sx", si);
        // outB2 = variables_.Bzz*outB2;

        return outL2 - outR2; // + outB1 - outB2;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplyJsx(Eigen::VectorXcd& GS_, const int& si) {
        // Jsx along a1, or i+x; 
        // si are sites on A sublattice
        int Nsite_ = variables_.NumberofSites;
        assert(si<Nsite_);
        int xneigh = Lat_.N1neigh_(si,2); // - x neigh
        int yneigh = Lat_.N1neigh_(si,1); // - y neigh
        int zneigh = Lat_.N1neigh_(si,0); // - z neigh

        assert(zneigh<Nsite_); // z - neigh
        assert(yneigh<Nsite_); // y - neigh
        assert(xneigh<Nsite_); // x - neigh

        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd outL1(Hsize); outL1.setZero();
        Eigen::VectorXcd outL2(Hsize); outL2.setZero();
        Eigen::VectorXcd outR1(Hsize); outR1.setZero();
        Eigen::VectorXcd outR2(Hsize); outR2.setZero();
        Eigen::VectorXcd outB1(Hsize); outB1.setZero();
        Eigen::VectorXcd outB2(Hsize); outB2.setZero();

        // Sx_i . Sy_i+x
        outL1 = ApplyOp(GS_, "Sy", xneigh);
        outL2 = ApplyOp(outL1, "Sx", si);
        outL2 = variables_.Kxx*outL2;

        // Sy_i . Sx_i+x
        outR1 = ApplyOp(GS_, "Sx", xneigh);
        outR2 = ApplyOp(outR1, "Sy", si);
        outR2 = variables_.Kxx*outR2;

        // // Sz_i
        // outB1 = ApplyOp(GS_, "Sz", si);
        // outB1 = variables_.Byy*outB1;

        // // Sy_i
        // outB2 = ApplyOp(GS_, "Sy", si);
        // outB2 = variables_.Bzz*outB2;

        return outL2 - outR2; // - outB1 + outB2;
    }

    // ------------------------------------------
    // ------------Begin Energy Current----------
    // ------------------------------------------
    // 0=z 1=y 2=x neighbors
    // Eigen::VectorXcd ApplyJez(Eigen::VectorXcd& GS_, const int& si) {
    //     int Nsite_ = variables_.NumberofSites;
    //     assert(si<Nsite_);
    //     int iz =   Lat_.N1neigh_(si,0);    // - si - z neigh
    //     int izpx = Lat_.N1neigh_(iz,2);  // - si+z - x neigh
    //     int izpy = Lat_.N1neigh_(iz,1);  // - si+z - y neigh
    //     assert(iz<Nsite_);  assert(izpx<Nsite_);  assert(izpy<Nsite_);

    //     int Hsize = Basis_.nHil_;
    //     Eigen::VectorXcd outL1(Hsize); outL1.setZero();
    //     Eigen::VectorXcd outL2(Hsize); outL2.setZero();
    //     Eigen::VectorXcd outL3(Hsize); outL3.setZero();

    //     Eigen::VectorXcd outR1(Hsize); outR1.setZero();
    //     Eigen::VectorXcd outR2(Hsize); outR2.setZero();
    //     Eigen::VectorXcd outR3(Hsize); outR3.setZero();

    //     // Sz_i . Sy_iz . Sx_iz+x
    //     outL1 = ApplyOp(GS_,   "Sx", izpx);
    //     outL2 = ApplyOp(outL1, "Sy", iz);
    //     outL3 = ApplyOp(outL2, "Sz", si);
    //     outL3 = variables_.Kzz*outL3;

    //     // Sz_i . Sx_iz . Sy_iz+y
    //     outR1 = ApplyOp(GS_,   "Sy", izpy);
    //     outR2 = ApplyOp(outR1, "Sx", iz);
    //     outR3 = ApplyOp(outR2, "Sz", si);
    //     outR3 = variables_.Kyy*outR3;

    //     return outL3 - outR3;
    // }

    // -----------------------------------
    // 0=z 1=y 2=x neighbors
    Eigen::VectorXcd ApplyJey(Eigen::VectorXcd& GS_, const int& si) { 
        // si must be on B sublattice
        int Nsite_ = variables_.NumberofSites;
        assert(si<Nsite_);
        int iy =   Lat_.N1neigh_(si,1);    // - si - y neigh (i + a2)_A
        int iypx = Lat_.N1neigh_(iy,2);  // - si+y - x neigh (i + a2 + a1)_B
        int iypz = Lat_.N1neigh_(iy,0);  // - si+y - z neigh (i + a2)_B
        assert(iy<Nsite_);  
        assert(iypx<Nsite_);  
        assert(iypz<Nsite_);

        int Hsize = Basis_.nHil_;
        Eigen::VectorXcd outL1(Hsize); outL1.setZero();
        Eigen::VectorXcd outL2(Hsize); outL2.setZero();
        Eigen::VectorXcd outL3(Hsize); outL3.setZero();

        Eigen::VectorXcd outR1(Hsize); outR1.setZero();
        Eigen::VectorXcd outR2(Hsize); outR2.setZero();
        Eigen::VectorXcd outR3(Hsize); outR3.setZero();

        Eigen::VectorXcd outHR1(Hsize); outHR1.setZero();
        Eigen::VectorXcd outHR2(Hsize); outHR2.setZero();

        Eigen::VectorXcd outHL1(Hsize); outHL1.setZero();
        Eigen::VectorXcd outHL2(Hsize); outHL2.setZero();

        // Sy_i . Sz_iy . Sx_iy+x
        outL1 = ApplyOp(GS_,   "Sx", iypx);
        outL2 = ApplyOp(outL1, "Sz", iy);
        outL3 = ApplyOp(outL2, "Sy", si);
        outL3 = variables_.Kxx*variables_.Kyy*outL3;

        // Sy_i . Sx_iy . Sz_iy+z
        outR1 = ApplyOp(GS_,   "Sz", iypz);
        outR2 = ApplyOp(outR1, "Sx", iy);
        outR3 = ApplyOp(outR2, "Sy", si);
        outR3 = variables_.Kyy*variables_.Kzz*outR3;

        // Sy_i Sz_iy
        outHL1 = ApplyOp(GS_, "Sz", iy);
        outHL2 = ApplyOp(outHL1, "Sy", si);
        outHL2 = variables_.Bxx*variables_.Kyy*outHL2;

        // Sy_i Sx_iy
        outHR1 = ApplyOp(GS_, "Sx", iy);
        outHR2 = ApplyOp(outHR1, "Sy", si);
        outHR2 = variables_.Bzz*variables_.Kyy*outHR2;

        return outL3 - outR3 + outHL2 - outHR2;
    }

    // -----------------------------------
    // 0=z 1=y 2=x neighbors
    Eigen::VectorXcd ApplyJex(Eigen::VectorXcd& GS_, const int& si) {
        // si must be on A sublattice
        int Nsite_ = variables_.NumberofSites;
        assert(si<Nsite_);
        int ix = Lat_.N1neigh_(si,2);    // - si - x neigh (i + a1)_B
        int ixpy = Lat_.N1neigh_(ix,1);  // - si+x - y neigh (i + a1 + a2)_A
        int ixpz = Lat_.N1neigh_(ix,0);  // - si+x - z neigh (i + a1)_A
        assert(ix<Nsite_);  assert(ixpy<Nsite_);  assert(ixpz<Nsite_);

        int Hsize = Basis_.nHil_;
        Eigen::VectorXcd outL1(Hsize); outL1.setZero();
        Eigen::VectorXcd outL2(Hsize); outL2.setZero();
        Eigen::VectorXcd outL3(Hsize); outL3.setZero();

        Eigen::VectorXcd outR1(Hsize); outR1.setZero();
        Eigen::VectorXcd outR2(Hsize); outR2.setZero();
        Eigen::VectorXcd outR3(Hsize); outR3.setZero();

        Eigen::VectorXcd outHR1(Hsize); outHR1.setZero();
        Eigen::VectorXcd outHR2(Hsize); outHR2.setZero();

        Eigen::VectorXcd outHL1(Hsize); outHL1.setZero();
        Eigen::VectorXcd outHL2(Hsize); outHL2.setZero();

        // Sx_i . Sz_ix . Sy_ix+y
        outR1 = ApplyOp(GS_, "Sy", ixpy);
        outR2 = ApplyOp(outR1, "Sz", ix);
        outR3 = ApplyOp(outR2, "Sx", si);
        outR3 = variables_.Kxx*variables_.Kyy*outR3;

        // Sx_i . Sy_ix . Sz_ix+z
        outL1 = ApplyOp(GS_, "Sz", ixpz);
        outL2 = ApplyOp(outL1, "Sy", ix);
        outL3 = ApplyOp(outL2, "Sx", si);
        outL3 = variables_.Kxx*variables_.Kzz*outL3;

        // Sy_i Sz_iy
        outHL1 = ApplyOp(GS_, "Sy", ix);
        outHL2 = ApplyOp(outHL1, "Sx", si);
        outHL2 = variables_.Bxx*variables_.Kyy*outHL2;

        // Sy_i Sx_iy
        outHR1 = ApplyOp(GS_, "Sz", ix);
        outHR2 = ApplyOp(outHR1, "Sx", si);
        outHR2 = variables_.Bzz*variables_.Kyy*outHR2;

        return outL3 - outR3 + outHL2 - outHR2;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplyJexJeyJez(Eigen::VectorXcd& GS_, const int& si) {

        int Hsize = Basis_.nHil_;
        int Nsite_ = variables_.NumberofSites;
        assert(si<Nsite_);

        Eigen::VectorXcd out1(Hsize); out1.setZero();
        Eigen::VectorXcd out2(Hsize); out2.setZero();
        // Eigen::VectorXcd out3(Hsize); out3.setZero();

        out1 = ApplyJex(GS_,si);
        out2 = ApplyJey(GS_,si);
        // out3 = ApplyJez(GS_,si);

        return out1 + out2;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplyJsxJsyJsz(Eigen::VectorXcd& GS_, const int& si) {

        int Hsize = Basis_.nHil_;
        int Nsite_ = variables_.NumberofSites;
        assert(si<Nsite_);

        Eigen::VectorXcd out1(Hsize); out1.setZero();
        Eigen::VectorXcd out2(Hsize); out2.setZero();
        // Eigen::VectorXcd out3(Hsize); out3.setZero();

        out1 = ApplyJsx(GS_,si);
        out2 = ApplyJsy(GS_,si);
        // out3 = ApplyJsz(GS_,si);

        return out1 + out2;
    }


    // -----------------------------------
    Eigen::VectorXcd ApplySz(Eigen::VectorXcd& GS_, const int& i) {
        int Nsite_ = variables_.NumberofSites;
        assert(i<Nsite_);
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize); out.setZero();

        for(int i1=0; i1<Hsize; i1++) {
            int ket = Basis_.basis[i1];
            int keto1;
            dcomplex Szi;

            Basis_.ApplySz(i,ket,keto1,Szi);
            assert(ket==keto1);
            out(keto1) += Szi*GS_(ket);
        }
        return out;
    }

    // -------------------------Helper Functions for Energy Current-----------------------------
    Eigen::VectorXcd ApplyJexTotal(Eigen::VectorXcd& GS_) {
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize);
        out.setZero();

        for (int s=0; s<Nsite_;s=s+2) {
            out += ApplyJex(GS_, s);
        }
        return out;
    }
    // -----------------------------------
    Eigen::VectorXcd ApplyJeyTotal(Eigen::VectorXcd& GS_) {
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize);
        out.setZero();

        for (int s=1; s<Nsite_;s=s+2) {
            out += ApplyJey(GS_, s);
        }
        return out;
    }

    // // -----------------------------------
    // Eigen::VectorXcd ApplyJezTotal(Eigen::VectorXcd& GS_) {
    //     int Nsite_ = variables_.NumberofSites;
    //     int Hsize = Basis_.basis.size();
    //     Eigen::VectorXcd out(Hsize);
    //     out.setZero();

    //     for (int s=0; s<Nsite_;s++) {
    //         // out += ApplyJez(GS_, s);
    //     }
    //     return out;
    // }


    // ------------------------Single Pauli-------------------------------------
    // ----------------------------------------------------------------------
    Eigen::VectorXcd ApplySy(Eigen::VectorXcd& GS_, const int& i) {
        int Nsite_ = variables_.NumberofSites;
        assert(i<Nsite_);
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize); out.setZero();

        for(int i1=0; i1<Hsize; i1++) {
            int ket = Basis_.basis[i1];
            int keto1;
            dcomplex Szi;

            Basis_.ApplySy(i,ket,keto1,Szi);
            out(keto1) += Szi*GS_(ket);
        }
        return out;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplySx(Eigen::VectorXcd& GS_, const int& i) {
        int Nsite_ = variables_.NumberofSites;
        assert(i<Nsite_);
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize); out.setZero();

        for(int i1=0; i1<Hsize; i1++) {
            int ket = Basis_.basis[i1];
            int keto1;
            dcomplex Szi;

            Basis_.ApplySx(i,ket,keto1,Szi);
            out(keto1) += Szi*GS_(ket);
        }
        return out;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplySp(Eigen::VectorXcd& GS_, const int& i) {

        int Hsize = Basis_.nHil_;
        int Nsite_ = variables_.NumberofSites;
        assert(i<Nsite_);

        Eigen::VectorXcd out1(Hsize); out1.setZero();
        Eigen::VectorXcd out2(Hsize); out2.setZero();

        out1 = ApplySx(GS_,i);
        out2 = ApplySy(GS_,i);

        return out1 + dcomplex(0.0,1.0) * out2;
    }

    Eigen::VectorXcd ApplySpe3(Eigen::VectorXcd& GS_, const int& i) {

        int Hsize = Basis_.nHil_;
        int Nsite_ = variables_.NumberofSites;
        assert(i<Nsite_);

        Eigen::VectorXcd outx(Hsize); outx.setZero();
        Eigen::VectorXcd outy(Hsize); outy.setZero();
        Eigen::VectorXcd outz(Hsize); outz.setZero();

        outx = ApplySx(GS_,i);
        outy = ApplySy(GS_,i);
        outz = ApplySz(GS_,i);

        return dcomplex(-0.408248,0.707107)*outx
                + dcomplex(-0.408248,-0.707107) * outy + 0.816497*outz;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplySm(Eigen::VectorXcd& GS_, const int& i) {

        int Hsize = Basis_.nHil_;
        int Nsite_ = variables_.NumberofSites;
        assert(i<Nsite_);

        Eigen::VectorXcd out1(Hsize); out1.setZero();
        Eigen::VectorXcd out2(Hsize); out2.setZero();

        out1 = ApplySx(GS_,i);
        out2 = ApplySy(GS_,i);

        return out1 - dcomplex(0.0,1.0) * out2;
    }

    Eigen::VectorXcd ApplySme3(Eigen::VectorXcd& GS_, const int& i) {

        int Hsize = Basis_.nHil_;
        int Nsite_ = variables_.NumberofSites;
        assert(i<Nsite_);

        Eigen::VectorXcd outx(Hsize); outx.setZero();
        Eigen::VectorXcd outy(Hsize); outy.setZero();
        Eigen::VectorXcd outz(Hsize); outz.setZero();

        outx = ApplySx(GS_,i);
        outy = ApplySy(GS_,i);
        outz = ApplySz(GS_,i);

        return dcomplex(-0.408248,-0.707107)*outx
               + dcomplex(-0.408248,0.707107) * outy + 0.816497*outz;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplySxSySz(Eigen::VectorXcd& GS_, const int& si) {

        int Hsize = Basis_.nHil_;
        int Nsite_ = variables_.NumberofSites;
        assert(si<Nsite_);

        Eigen::VectorXcd out1(Hsize); out1.setZero();
        Eigen::VectorXcd out2(Hsize); out2.setZero();
        Eigen::VectorXcd out3(Hsize); out3.setZero();

        out1 = ApplySx(GS_,si);
        out2 = ApplySy(GS_,si);
        out3 = ApplySz(GS_,si);

        return out1 + out2 + out3;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplySzTotal(Eigen::VectorXcd& GS_, bool square) {
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();

        Eigen::VectorXcd out(Hsize);
        for(int i1=0; i1<Hsize; i1++) {
            int ket = Basis_.basis[i1];
            int keto1;
            dcomplex Szt=0;

            for (int s=0; s<Nsite_;s++) {
                dcomplex Szi;
                Basis_.ApplySz(s,ket,keto1,Szi);
                assert(ket==keto1);
                assert(Szi.imag()==0);
                Szt+=Szi;
            }
            if(square) {
                out(ket) = Szt*Szt*GS_(ket);
            } else {
                out(ket) = Szt*GS_(ket);
            }

        }
        return out;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplySxTotal(Eigen::VectorXcd& GS_, bool square) {
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize);
        out.setZero();

        for (int s=0; s<Nsite_;s++) {
            out += ApplySx(GS_,s);
        }
        return out;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplySyTotal(Eigen::VectorXcd& GS_, bool square) {
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize);
        out.setZero();

        for (int s=0; s<Nsite_;s++) {
            out += ApplySy(GS_,s);
        }
        return out;
    }

    // -----------------------------------
    Eigen::VectorXcd ApplyJsxTotal(Eigen::VectorXcd& GS_) {
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize);
        out.setZero();

        for (int s=0; s<Nsite_;s=s+2) {
            out += ApplyJsx(GS_, s);
        }
        return out;
    }
    // -----------------------------------
    Eigen::VectorXcd ApplyJsyTotal(Eigen::VectorXcd& GS_) {
        int Nsite_ = variables_.NumberofSites;
        int Hsize = Basis_.basis.size();
        Eigen::VectorXcd out(Hsize);
        out.setZero();

        for (int s=1; s<Nsite_;s=s+2) {
            out += ApplyJsy(GS_, s);
        }
        return out;
    }

    // // -----------------------------------
    // Eigen::VectorXcd ApplyJszTotal(Eigen::VectorXcd& GS_) {
    //     int Nsite_ = variables_.NumberofSites;
    //     int Hsize = Basis_.basis.size();
    //     Eigen::VectorXcd out(Hsize);
    //     out.setZero();

    //     for (int s=0; s<Nsite_;s++) {
    //         out += ApplyJsz(GS_, s);
    //     }
    //     return out;
    // }

private:
    ConstVariables& variables_;
    Lattice& Lat_;
    QBasis& Basis_;
    Hamiltonian& Hamil_;
    //VectorXf& GS_;
};
#endif //LANCZOSCOND_OBSERVABLES_H
