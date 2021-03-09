#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "NeuralNetwork.hpp"
using namespace std;

#ifndef GENETICARRAY_H
#define GENETICARRAY_H

class GeneticArray{
    public:
        //constructor initializes binary array randomly, calculates double array
        GeneticArray();
        
        //constructor initializes binary array to initBinaryArray
        GeneticArray(int initBinaryArray[]);
        
        //copy constructor
        GeneticArray(const GeneticArray& other);
        
        //destructor
        //~GeneticArray();
        
        //randomizes array(same thing as constructor)
        void randomize();
        
        //assignment operator
        GeneticArray operator=(const GeneticArray& other);
        
        //returns genetic array with mixed binary array elements from other
        GeneticArray operator+(const GeneticArray& other);
        
        //returns bit difference between two genetic arrays
        int operator-(GeneticArray& other);
        
        //calculates double array
        void calculateDouble();
        
        //return PSI
        double getPSI(double x);
        
        //return element of double array at idx
        double getDouble(int idx) const;
        
        //return element of binary array at idx
        int getBinary(int idx) const;
        
        //evaluates and returns fitness, R using M random evaluations given by rp
        double getR(double rp[], int M);
        
        //optimizes neural network using back propagation;
        void optimize(double rp[], int M, double alpha);
        
        /*
        //print binary array
        void printBinary();
        
        //print double array
        void printDouble();
        */
        
    private:
        //number of bits in binary array
        const int nBits = 832;
        
        //number of elements in double array
        const int nDouble = 26;
        
        //832 bits for 26 numbers with float precision
        //from left to right, the array stores w_j, b_j, w_jk, and b_k
        int binaryArray[832];
        
        //stores corresponding double values of binary array
        double doubleArray[26];
        
        //stores fitness value of genetic array
        double R;
        
        //Neural Network used to evaluate PSI and fitness function
        NeuralNetwork NN;
};

#endif
