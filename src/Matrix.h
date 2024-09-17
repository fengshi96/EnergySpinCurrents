//
// Created by Feng, Shi on 4/2/24.
//

#ifndef LANCZOSCOND_MATRIX_H
#define LANCZOSCOND_MATRIX_H
#include <string>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
using namespace std;

template<typename T>
class  Matrix  {
public:
    typedef T value_type;

    //set all elements to zero
    Matrix()
            : nrow_(0), ncol_(0)
    {}

    //allocate number of row col and elements
    Matrix(int nrow,int ncol)
            : nrow_(nrow),ncol_(ncol),data_(nrow*ncol)
    {}

    // copy constructor
    Matrix(const Matrix<T>& m) {
        nrow_=m.nrow_;
        ncol_=m.ncol_;
        data_=m.data_;
    }

    const T& operator()(int i,int j) const;
    T& operator()(int i, int j);
    int numNonZeros();
    void print();
    void clear();
    void resize(int newrow, int newcol);
    int n_row();
    int n_col();
    void fill(T val);
    void del();


    bool IsHermitian() {
        bool out=true;
        for(int i=0; i<nrow_; i++) {
            for(int j=0; j<ncol_; j++) {
                T Hij = data_[i+j*nrow_];
                T Hji = data_[j+i*nrow_];
                if(Hij!=conj(Hji)) {
                    std::string tmp = "Hij != Hji "+ std::to_string(i)+"-"+std::to_string(j)+" \n";
                    std::cout << i << " \t " << j << " \t " << Hij << " \t " << Hji << std::endl;
                    out=false;
                    return out;
                }
            }
        }
        return out;
    }

private:
    int nrow_,ncol_;
    std::vector<T> data_;
};



/*
 * ***********
 *  Functions in Class Matrix ------
 *  ***********
*/

template<class T>
int Matrix<T>::n_row() {
    return nrow_;
} // ----------

template<class T>
int Matrix<T>::n_col() {
    return ncol_;
} // ----------

template<class T>
void Matrix<T>::fill(T val) {
    std::fill(data_.begin(),data_.end(),val);
} // ----------

template<class T>
void Matrix<T>::resize(int newrow, int newcol) {
    assert(newrow>0 && newcol>0);
    nrow_=newrow;
    ncol_=newcol;
    data_.clear();
    data_.resize(newrow*newcol);
} // ----------


template<class T>
int Matrix<T>::numNonZeros(){
    int counter=0;
    for(int i=0; i<data_.size(); i++)
        if(data_[i]!=0) counter++;
    return counter;
} // ----------


template<class T>
T& Matrix<T>::operator()(int i,int j){
    assert(i<nrow_ && j<ncol_);
    assert(i+j*nrow_<data_.size());
    return data_[i+j*nrow_];
} // ----------


template<class T>
void Matrix<T>::print(){
    std::cout.precision(8);
    std::cout<<"shape:= ("<<nrow_<<","<<ncol_<<")"<<std::endl;
    for(int i=0; i<nrow_; i++) {
        for(int j=0; j<ncol_; j++) {
            std::cout << data_[i+j*nrow_] << "\t";
        }
        std::cout << std::endl;
    }
    return;
} // ----------

template<class T>
void Matrix<T>::clear(){
    for(int i=0; i<nrow_; i++) {
        for(int j=0; j<ncol_; j++) {
            data_[i+j*nrow_] = 0.0;
        }
    }
    return;
} // ----------

template<class T>
void Matrix<T>::del(){
    nrow_=0; ncol_=0;
    data_.resize(0);
    return;
} // ----------

#endif //LANCZOSCOND_MATRIX_H
