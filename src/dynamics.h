//
// Created by Feng, Shi on 4/2/24.
//

#ifndef LANCZOSCOND_DYNAMICS_H
#define LANCZOSCOND_DYNAMICS_H

#include "DynLanczosSolver.h"
#include "Observables.h"
typedef DynLanczosSolver DynLancType;
typedef Observables ObservType;

// ===== define functions ==========
VectorXf ReturnRow(Matrix<double>& mat, int row);
void StartDynamics(string& dynstr,ConstVariables& variables, Lattice& Lat,
                   QBasis& Basis,Hamiltonian& Hamil,
                   DynLancType& DLancGS, Eigen::VectorXcd& Psi);
void DynSzSz(string& dynstr,ConstVariables& variables, Lattice& Lat,
             QBasis& Basis,Hamiltonian& Hamil,
             DynLancType& DLancGS, Eigen::VectorXcd& Psi);
void LocalDynSzSz(string& dynstr,ConstVariables& variables, Lattice& Lat,
                  QBasis& Basis,Hamiltonian& Hamil,
                  DynLancType& DLancGS, Eigen::VectorXcd& Psi);
void DynJsJs(string& dynstr,ConstVariables& Parameters, Lattice& Lat,
             QBasis& Basis, Hamiltonian& Hamiltonian,
             DynLancType& DLancGS, Eigen::VectorXcd& Psi);
void DynJeJe(string& dynstr,ConstVariables& Parameters, Lattice& Lat,
             QBasis& Basis, Hamiltonian& Hamiltonian,
             DynLancType& DLancGS, Eigen::VectorXcd& Psi);
void LocalDynRIXS(string& dynstr,ConstVariables& Parameters, Lattice& Lat,
                  QBasis& Basis, Hamiltonian& Hamiltonian,
                  DynLancType& DLancGS, Eigen::VectorXcd& Psi);

dcomplex invComplex(double x, double y){
    double re = x/(x*x+y*y);
    double im = y/(x*x+y*y);
    dcomplex out(re,im);
    return out;
}

// ================================================
// --------- distribute dynamics ------------------
void StartDynamics(string& dynstr, ConstVariables& variables, Lattice& Lat,
                   QBasis& Basis, Hamiltonian& Hamil,
                   DynLancType& DLancGS, Eigen::VectorXcd& Psi) {
    if(dynstr=="SzSz"){
        DynSzSz(dynstr,variables,Lat,Basis,Hamil,DLancGS,Psi);
    } else if(dynstr=="LocalSzSz"){
        LocalDynSzSz(dynstr,variables,Lat,Basis,Hamil,DLancGS,Psi);
    } else if(dynstr=="Spin Current"){
        DynJsJs(dynstr,variables,Lat,Basis,Hamil,DLancGS,Psi);
    } else if(dynstr=="Energy Current"){
        DynJeJe(dynstr,variables,Lat,Basis,Hamil,DLancGS,Psi);
    } else if(dynstr=="LocalRIXS"){
        LocalDynRIXS(dynstr,variables,Lat,Basis,Hamil,DLancGS,Psi);
    } else {
        std::cerr<<"dynstr="<<dynstr<<"\n";
        throw std::string("Unknown dynstr parameter \n");
    }
}

// ================================================
// --------- LOCAL - SzSz Dynamics ---- <gs|Siz(t) Siz(0)|gs>
void LocalDynSzSz(string& dynstr,ConstVariables& Parameters, Lattice& Lat,
                  QBasis& Basis, Hamiltonian& Hamiltonian,
                  DynLancType& DLancGS, Eigen::VectorXcd& Psi){
    cout << " --> Startin Dynamics: " << dynstr << endl;
    int Nsite_ = Parameters.NumberofSites;
    int centeral_site = Parameters.SpecialSite;
    cout << setprecision(8);

    int omegasteps=1000;
    double domega=0.002;
    double eta = 0.005;
    double Eg = DLancGS.TriDeval(0);

    ObservType Observ(Parameters, Lat, Basis, Hamiltonian);

    int ExtState = DLancGS.PsiAll.size();
    int someSt = int(ExtState*1.0);
    //cout << ExtState << " \t " << someSt << endl;

    std::vector<Eigen::MatrixXcd> Overlap(6);
    Eigen::VectorXcd Sx(Nsite_), Sy(Nsite_), Sz(Nsite_), St(Nsite_), Sp(Nsite_), Sm(Nsite_);
    Overlap[0].resize(Nsite_,ExtState); Overlap[0].setZero();
    Overlap[1].resize(Nsite_,ExtState); Overlap[1].setZero();
    Overlap[2].resize(Nsite_,ExtState); Overlap[2].setZero();
    Overlap[3].resize(Nsite_,ExtState); Overlap[3].setZero();
    Overlap[4].resize(Nsite_,ExtState); Overlap[4].setZero();
    Overlap[5].resize(Nsite_,ExtState); Overlap[5].setZero();

    for (int si=0; si<Nsite_; si++) {
        Eigen::VectorXcd Sxi = Observ.ApplySx(Psi,si);
        Eigen::VectorXcd Syi = Observ.ApplySy(Psi,si);
        Eigen::VectorXcd Szi = Observ.ApplySz(Psi,si);
        Eigen::VectorXcd Sti = Observ.ApplySxSySz(Psi,si);
        Eigen::VectorXcd Spi = Observ.ApplySpe3(Psi,si); // Sxi + dcomplex(0.0,1.0)*Syi;
        Eigen::VectorXcd Smi = Observ.ApplySme3(Psi,si); // Sxi - dcomplex(0.0,1.0)*Syi;

        Sx[si] = Psi.dot(Sxi);
        Sy[si] = Psi.dot(Syi);
        Sz[si] = Psi.dot(Szi);
        St[si] = Psi.dot(Sti);
        Sp[si] = Psi.dot(Spi);
        Sm[si] = Psi.dot(Smi);

#pragma omp parallel for
        for (int n=0; n<someSt; n++) {
            Eigen::VectorXcd nstate = DLancGS.PsiAll[n];
            Overlap[0](si,n) = nstate.dot(Sxi);// - Sx[si];
            Overlap[1](si,n) = nstate.dot(Syi);// - Sy[si];
            Overlap[2](si,n) = nstate.dot(Szi);// - Sz[si];
            Overlap[3](si,n) = nstate.dot(Sti);// - St[si];
            Overlap[4](si,n) = nstate.dot(Spi);// - Sz[si];
            Overlap[5](si,n) = nstate.dot(Smi);// - Sz[si];
        }
#pragma omp barrier
    }

    ofstream myfile;
    myfile.open ("2SSLocal_outSpectrum.real");
    Eigen::VectorXcd data(8); data.setZero();

    for (int om=0; om<=omegasteps;om++) {
        double omega = domega*om;
        Eigen::MatrixXcd Itensity(3,3);
        Eigen::MatrixXcd ItensitySpinTrans(2,1);  // [0] SmSp, [1] SpSm
        Itensity.setZero();
        ItensitySpinTrans.setZero();
        dcomplex sum(0,0);

        dcomplex dnom1(omega,-eta);
        for (int si=0;si<Nsite_;si++) {
            for (int n=0; n<someSt; n++) {

                double En = DLancGS.TriDeval(n);
                dcomplex denominator(omega-(En-Eg),-eta);

                // --- <n|Si|gs> ---
                dcomplex tmpx = Overlap[0](si,n);
                dcomplex tmpy = Overlap[1](si,n);
                dcomplex tmpz = Overlap[2](si,n);
                dcomplex tmpt = Overlap[3](si,n);
                dcomplex tmpp = Overlap[4](si,n);
                dcomplex tmpm = Overlap[5](si,n);

                // --- <gs|S'|n> <n|S|gs> ---
                dcomplex tmpxx = conj(tmpx) * tmpx; // x x
                dcomplex tmpxy = conj(tmpx) * tmpy; // x y
                dcomplex tmpxz = conj(tmpx) * tmpz; // x z

                dcomplex tmpyy = conj(tmpy) * tmpy; // y y
                dcomplex tmpyz = conj(tmpy) * tmpz; // x z
                dcomplex tmpzz = conj(tmpz) * tmpz; // z z

                dcomplex tmptt = conj(tmpt) * tmpt; // z z

                Itensity(0,0) = Itensity(0,0) + tmpxx/denominator;
                Itensity(0,1) = Itensity(0,1) + tmpxy/denominator;
                Itensity(0,2) = Itensity(0,2) + tmpxz/denominator;

                Itensity(1,1) = Itensity(1,1) + tmpyy/denominator;
                Itensity(1,2) = Itensity(1,2) + tmpyz/denominator;
                Itensity(2,2) = Itensity(2,2) + tmpzz/denominator;

                sum = sum + tmptt/denominator;

                // Spin current related quantities
                dcomplex tmppp = conj(tmpp) * tmpp; // + +
                dcomplex tmpmm = conj(tmpm) * tmpm; // - -

                ItensitySpinTrans(0,0) = ItensitySpinTrans(0,0) + tmppp/denominator;
                ItensitySpinTrans(1,0) = ItensitySpinTrans(1,0) + tmpmm/denominator;
            }

            // --- subtracting background elastic contribution ---
            Itensity(0,0) = Itensity(0,0) - conj(Sx[si])*Sx[si]/dnom1;
            Itensity(0,1) = Itensity(0,1) - conj(Sx[si])*Sy[si]/dnom1;
            Itensity(0,2) = Itensity(0,2) - conj(Sx[si])*Sz[si]/dnom1;
            Itensity(1,1) = Itensity(1,1) - conj(Sy[si])*Sy[si]/dnom1;
            Itensity(1,2) = Itensity(1,2) - conj(Sy[si])*Sz[si]/dnom1;
            Itensity(2,2) = Itensity(2,2) - conj(Sz[si])*Sz[si]/dnom1;
            sum = sum - conj(St[si])*St[si]/dnom1;

            ItensitySpinTrans(0,0) -= Sm[si] * Sp[si]/dnom1;
            ItensitySpinTrans(1,0) -= Sp[si] * Sm[si]/dnom1;
        }

        data(0) = Itensity(0,0);
        data(1) = Itensity(1,1);
        data(2) = Itensity(2,2);
        data(3) = Itensity(0,1);
        data(4) = Itensity(0,2);
        data(5) = Itensity(1,2);
        data(6) = ItensitySpinTrans(0,0);
        data(7) = ItensitySpinTrans(1,0);

        myfile << omega << " \t ";
        for(int kk=0;kk<8;kk++) {
            // myfile << data[kk].real() << " \t " << data[kk].imag() << " \t ";
            myfile << data[kk].imag() << " \t ";
        }
//        myfile << sum.real() << " \t " << sum.imag();
        myfile << " \n ";

        if(om%200==0) {
            cout << omega << " \t ";
            for(int kk=0;kk<8;kk++) cout << data[kk].real() << " \t " << data[kk].imag() << " \t ";
            cout << sum.real() << " \t " << sum.imag();
            cout << " \n ";
        }
    }
    cout << " " << endl;

    cout << endl;
    myfile.close();
}


// ================================================
// --------- LOCAL - Js.Js Dynamics ---------
void DynJsJs(string& dynstr,ConstVariables& Parameters, Lattice& Lat,
             QBasis& Basis, Hamiltonian& Hamiltonian,
             DynLancType& DLancGS, Eigen::VectorXcd& Psi){
    cout << " --> Startin Dynamics: " << dynstr << endl;

    int Nsite_ = Parameters.NumberofSites;
    int centeral_site = Parameters.SpecialSite;
    cout << setprecision(8);

    int omegasteps=1000;
    double domega=0.002;
    double eta = 0.005;
    double Eg = DLancGS.TriDeval(0);

    ObservType Observ(Parameters, Lat, Basis, Hamiltonian);

    int ExtState = DLancGS.PsiAll.size();
    int someSt = int(ExtState*1.0);
    //cout << ExtState << " \t " << someSt << endl;



    Eigen::VectorXcd Jtx = Observ.ApplyJsxTotal(Psi);
    Eigen::VectorXcd Jty = Observ.ApplyJsyTotal(Psi);
    // Eigen::VectorXcd Jtz = Observ.ApplyJszTotal(Psi);
    Eigen::VectorXcd Jt = Jtx + Jty;
    Eigen::VectorXcd JSx(1), JSy(1), JSz(1), JSt(1);
    JSx.setZero(); JSy.setZero(); JSz.setZero(); JSt.setZero();
    JSx[0] = Psi.dot(Jtx);
    JSy[0] = Psi.dot(Jty);
    JSt[0] = Psi.dot(Jt);

    Eigen::MatrixXcd Overlap(ExtState,3); Overlap.setZero();
#pragma omp parallel for
    for (int n=0; n<someSt; n++) {
        Eigen::VectorXcd nstate = DLancGS.PsiAll[n];
        Overlap(n,0) = nstate.dot(Jtx);
        Overlap(n,1) = nstate.dot(Jty);
        Overlap(n,2) = nstate.dot(Jt);
    }
#pragma omp barrier

    ofstream myfile;
    myfile.open ("JsJs_outSpectrum.real");

    Eigen::VectorXcd data(4); data.setZero();
    for (int om=0; om<=omegasteps;om++) {
        double omega = domega*om;
        Eigen::MatrixXcd Itensity(2,2);
        Itensity.setZero();
        dcomplex sum(0,0);

        dcomplex dnom1(omega,-eta);
        for (int n=0; n<someSt; n++) {
            double En = DLancGS.TriDeval(n);
            dcomplex denominator(omega-(En-Eg),-eta);

            // --- <n|J|gs> ---
            dcomplex tmpx = Overlap(n,0);
            dcomplex tmpy = Overlap(n,1);
            dcomplex tmpt = Overlap(n,2);


            // --- <gs|J'|n> <n|J|gs> ---
            dcomplex tmpxx = conj(tmpx) * tmpx; // x x
            dcomplex tmpxy = conj(tmpx) * tmpy; // x y
            dcomplex tmpyy = conj(tmpy) * tmpy; // y y
            dcomplex tmpyx = conj(tmpy) * tmpx; // x z

            dcomplex tmptt = conj(tmpt) * tmpt; // t t

            Itensity(0,0) = Itensity(0,0) + tmpxx/denominator;
            Itensity(0,1) = Itensity(0,1) + tmpxy/denominator;
            Itensity(1,1) = Itensity(1,1) + tmpyy/denominator;
            Itensity(1,0) = Itensity(1,0) + tmpyx/denominator;
            sum = sum + tmptt/denominator;
        }

        // --- subtracting background elastic contribution ---
        Itensity(0,0) = Itensity(0,0) - conj(JSx[0])*JSx[0]/dnom1;
        Itensity(0,1) = Itensity(0,1) - conj(JSx[0])*JSy[0]/dnom1;
        Itensity(1,1) = Itensity(1,1) - conj(JSy[0])*JSy[0]/dnom1;
        Itensity(1,0) = Itensity(1,0) - conj(JSy[0])*JSz[0]/dnom1;
        sum = sum - conj(JSt[0])*JSt[0]/dnom1;

        data(0) = Itensity(0,0);
        data(1) = Itensity(1,1);
        data(2) = Itensity(0,1);
        data(3) = Itensity(1,0);

        myfile << omega << " \t ";
        for(int kk=0;kk<4;kk++) myfile << data[kk].real() << " \t " << data[kk].imag() << " \t ";
        myfile << sum.real() << " \t " << sum.imag();
        myfile << " \n ";

        if(om%10==0) {
            cout << omega << " \t ";
            for(int kk=0;kk<4;kk++) cout << data[kk].imag() << " \t ";
            cout << sum.imag();
            cout << " \n ";
        }
    }
    cout << " " << endl;

    cout << endl;
    myfile.close();
}

// ================================================
// --------- LOCAL - Je.Je Dynamics ---------
void DynJeJe(string& dynstr,ConstVariables& Parameters, Lattice& Lat,
             QBasis& Basis, Hamiltonian& Hamiltonian,
             DynLancType& DLancGS, Eigen::VectorXcd& Psi){
    cout << " --> Startin Dynamics: " << dynstr << endl;

    int Nsite_ = Parameters.NumberofSites;
    int centeral_site = Parameters.SpecialSite;
    cout << setprecision(8);

    int omegasteps=1000;
    double domega=0.002;
    double eta = 0.005;
    double Eg = DLancGS.TriDeval(0);

    ObservType Observ(Parameters, Lat, Basis, Hamiltonian);

    int ExtState = DLancGS.PsiAll.size();
    int someSt = int(ExtState*1.0);
    //cout << ExtState << " \t " << someSt << endl;

    Eigen::VectorXcd Jtx = Observ.ApplyJexTotal(Psi);
    Eigen::VectorXcd Jty = Observ.ApplyJeyTotal(Psi);
    Eigen::VectorXcd Jt = Jtx + Jty;
    Eigen::VectorXcd Jex(1), Jey(1), Jez(1), Jet(1);
    Jex.setZero(); Jey.setZero(); Jet.setZero();
    Jex[0] = Psi.dot(Jtx);
    Jey[0] = Psi.dot(Jty);
    Jet[0] = Psi.dot(Jt);


    Eigen::MatrixXcd Overlap(ExtState,3); Overlap.setZero();
#pragma omp parallel for
    for (int n=0; n<someSt; n++) {
        Eigen::VectorXcd nstate = DLancGS.PsiAll[n];
        Overlap(n,0) = nstate.dot(Jtx);
        Overlap(n,1) = nstate.dot(Jty);
        Overlap(n,2) = nstate.dot(Jt);
    }
#pragma omp barrier

    ofstream myfile;
    myfile.open ("JeJe_outSpectrum.real");

    Eigen::VectorXcd data(4); data.setZero();
    for (int om=0; om<=omegasteps;om++) {
        double omega = domega*om;
        Eigen::MatrixXcd Itensity(2,2);
        Itensity.setZero();
        dcomplex sum(0,0);

        dcomplex dnom1(omega,-eta);
        for (int n=0; n<someSt; n++) {
            double En = DLancGS.TriDeval(n);
            dcomplex denominator(omega-(En-Eg),-eta);

            // --- <n|Je|gs> ---
            dcomplex tmpx = Overlap(n,0);
            dcomplex tmpy = Overlap(n,1);
            dcomplex tmpt = Overlap(n,2);


            // --- <gs|Je'|n> <n|Je|gs> ---
            dcomplex tmpxx = conj(tmpx) * tmpx; // x x
            dcomplex tmpxy = conj(tmpx) * tmpy; // x y
            dcomplex tmpyy = conj(tmpy) * tmpy; // y y
            dcomplex tmpyx = conj(tmpy) * tmpx; // y x
            dcomplex tmptt = conj(tmpt) * tmpt; // t t

            Itensity(0,0) = Itensity(0,0) + tmpxx/denominator;
            Itensity(0,1) = Itensity(0,1) + tmpxy/denominator;
            Itensity(1,1) = Itensity(1,1) + tmpyy/denominator;
            Itensity(1,0) = Itensity(1,0) + tmpyx/denominator;
            sum = sum + tmptt/denominator;
        }

        // --- subtracting background elastic contribution ---
        Itensity(0,0) = Itensity(0,0) - conj(Jex[0])*Jex[0]/dnom1;
        Itensity(0,1) = Itensity(0,1) - conj(Jex[0])*Jey[0]/dnom1;
        Itensity(1,1) = Itensity(1,1) - conj(Jey[0])*Jey[0]/dnom1;
        Itensity(1,0) = Itensity(1,0) - conj(Jey[0])*Jez[0]/dnom1;
        sum = sum - conj(Jet[0])*Jet[0]/dnom1;

        data(0) = Itensity(0,0);
        data(1) = Itensity(1,1);
        data(2) = Itensity(0,1);
        data(3) = Itensity(1,0);

        myfile << omega << " \t ";
        for(int kk=0;kk<4;kk++) myfile << data[kk].imag() << " \t ";
        myfile << sum.imag();
        myfile << " \n ";

        if(om%10==0) {
            cout << omega << " \t ";
            for(int kk=0;kk<4;kk++) cout << data[kk].imag() << " \t ";
            cout << sum.imag();
            cout << " \n ";
        }

        // myfile << omega << " \t ";
        // for(int kk=0;kk<4;kk++) myfile << data[kk].real() << " \t " << data[kk].imag() << " \t ";
        // myfile << sum.real() << " \t " << sum.imag();
        // myfile << " \n ";

        // if(om%50==0) {
        //     cout << omega << " \t ";
        //     for(int kk=0;kk<4;kk++) cout << data[kk].real() << " \t " << data[kk].imag() << " \t ";
        //     cout << sum.real() << " \t " << sum.imag();
        //     cout << " \n ";
        // }
    }
    cout << " " << endl;

    cout << endl;
    myfile.close();
}


// ================================================
// --------- LOCAL - SzSz Dynamics ---- <gs|Siz(t) Siz(0)|gs>
void LocalDynRIXS(string& dynstr,ConstVariables& Parameters, Lattice& Lat,
                  QBasis& Basis, Hamiltonian& Hamiltonian,
                  DynLancType& DLancGS, Eigen::VectorXcd& Psi){
    cout << " ================================================ " << endl;
    cout << " ------- Startin RIXS Local Dynamics " << dynstr << " ------ " << endl;
    cout << " ================================================ " << endl;

    int Nsite_ = Parameters.NumberofSites;
    int centeral_site = Parameters.SpecialSite;
    double Eg = DLancGS.TriDeval(0);
    dcomplex zero(0.0,0.0);

    Hamiltonian.SetupConnectors();
    Hamiltonian.RIXS_killcenter(centeral_site);
    // ================================
    Hamiltonian.TightB_Ham();
    Hamiltonian.Diagonal();
    // ================================

    ObservType Observ(Parameters, Lat, Basis, Hamiltonian);
    //ObservType Observ(Parameters, Basis, Hamiltonian);
    DynLancType DLancInter(Parameters, Basis, Hamiltonian);
    DLancInter.Lanczos_Nirav("All");

    int nExtState = DLancInter.PsiAll.size();
    int mExtState = DLancGS.PsiAll.size();
    cout << nExtState << " \t " << mExtState << endl;
    double eta = 0.1;
    double Gamma = 100;

    Matrix<dcomplex> AI(mExtState,nExtState),ASx(mExtState,nExtState);
    Matrix<dcomplex> ASy(mExtState,nExtState),ASz(mExtState,nExtState);
    AI.fill(zero); ASx.fill(zero); ASy.fill(zero); ASz.fill(zero);
#pragma omp parallel for
    for (int m=0; m<mExtState; m++) {
        cout << m << " ";
        Eigen::VectorXcd mstate = DLancGS.PsiAll[m];
        int cs = centeral_site;
        dcomplex Idsum(zero), Sxsum(zero), Sysum(zero), Szsum(zero);
        for (int n=0; n<nExtState; n++) {
            double En = DLancInter.TriDeval(n);
            Eigen::VectorXcd nstate = DLancInter.PsiAll[n];
            Eigen::VectorXcd Sxi_ns = Observ.ApplySx(nstate,cs);
            Eigen::VectorXcd Syi_ns = Observ.ApplySy(nstate,cs);
            Eigen::VectorXcd Szi_ns = Observ.ApplySz(nstate,cs);
            dcomplex n0 = nstate.dot(Psi);
            dcomplex mIdn = mstate.dot(nstate);
            dcomplex mSxn = mstate.dot(Sxi_ns);
            dcomplex mSyn = mstate.dot(Syi_ns);
            dcomplex mSzn = mstate.dot(Szi_ns);
            dcomplex inv = invComplex(0-(En),Gamma);
            AI(m,n) = n0*mIdn*inv;
            ASx(m,n) = n0*mSxn*inv;
            ASy(m,n) = n0*mSyn*inv;
            ASz(m,n) = n0*mSzn*inv;
        }
    }
    cout << endl;



    ofstream myfile;
    myfile.open ("RIXS_outSpectrum.real");
    for (int om=0; om<=120;om++) {
        double omega = 0.1*om;

        dcomplex Io(zero), Ix(zero), Iy(zero), Iz(zero);
        for (int m=1; m<mExtState; m++) {
            double Em = DLancGS.TriDeval(m);
            double xm = omega-(Em-Eg);
            double mlorentzian = (eta)/(xm*xm + eta*eta);
            int cs = centeral_site;

            dcomplex Idsum(zero), Sxsum(zero), Sysum(zero), Szsum(zero);
            for (int n=0; n<nExtState; n++) {
                Idsum += AI(m,n);
                Sxsum += ASx(m,n);
                Sysum += ASy(m,n);
                Szsum += ASz(m,n);
            }

            Io += Idsum*conj(Idsum)*mlorentzian;
            Ix += Sxsum*conj(Sxsum)*mlorentzian;
            Iy += Sysum*conj(Sysum)*mlorentzian;
            Iz += Szsum*conj(Szsum)*mlorentzian;
        }
        cout << omega << " \t "
             << Io.real() << " \t " << Ix.real() << " \t " << Iy.real() << " \t " << Iz.real() << " \t "
             << Io.imag() << " \t " << Ix.imag() << " \t " << Iy.imag() << " \t " << Iz.imag() << endl;

        myfile << omega << " \t "
               << Io.real() << " \t " << Ix.real() << " \t " << Iy.real() << " \t " << Iz.real() << " \t "
               << Io.imag() << " \t " << Ix.imag() << " \t " << Iy.imag() << " \t " << Iz.imag() << endl;
    }
    myfile.close();

}


// ================================================
// --------- SzSz Dynamics ---- <gs|Sz(t) Sz(0)|gs>
void DynSzSz(string& dynstr,ConstVariables& Parameters, Lattice& Lat,
             QBasis& Basis, Hamiltonian& Hamiltonian,
             DynLancType& DLancGS, Eigen::VectorXcd& Psi){
    cout << " ================================================ " << endl;
    cout << " ------- Startin Dynamics " << dynstr << " ------ " << endl;
    cout << " ================================================ " << endl;

    int Nsite_ = Parameters.NumberofSites;
    int corb = 0;
    ObservType Observ(Parameters, Lat, Basis, Hamiltonian);
    DynLancType DLanc(Parameters, Basis, Hamiltonian);
    DLanc.Lanczos_Nirav("All");

    int ExtState = DLanc.PsiAll.size();
    int someSt = int(ExtState*1.0);
    cout << ExtState << " \t " << someSt << endl;

    ofstream myfile;
    myfile.open ("SzSz_outSpectrum.real");
    double eta = 0.1;
    double Eg = DLanc.TriDeval(0);
    for (int om=0; om<=120;om++) {
        double omega = 0.1*om;
        for (int si=0;si<Nsite_;si++) {
            Eigen::VectorXcd Szi = Observ.ApplySz(Psi,si);
            //Szi.normalize();
            for (int sj=0; sj<Nsite_; sj++) {
                double realI=0.0;
                double imagI=0.0;
                Eigen::VectorXcd Szj = Observ.ApplySz(Psi,sj);
                //Szj.normalize();
                for (int n=0; n<someSt; n++) {
                    Eigen::VectorXcd nstate = DLanc.PsiAll[n]; // ReturnRow(DLanc.PsiAll, n);
                    dcomplex tmp1 = Szj.dot(nstate) * nstate.dot(Szi);
                    double tmp = tmp1.real();
                    double En = DLanc.TriDeval(n);
                    double x = omega-(En-Eg);

                    realI += tmp*x/(x*x + eta*eta);
                    imagI += tmp*eta/(x*x + eta*eta);
                }
                cout << si << " \t " << sj << " \t " << omega << " \t " << realI << " \t " << imagI << endl;
                myfile << si << " \t " << sj << " \t " << omega << " \t " << realI << " \t " << imagI << endl;
            }
            cout << " " << endl;
        }
    }
    myfile.close();
}

#endif //LANCZOSCOND_DYNAMICS_H
