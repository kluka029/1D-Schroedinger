#include <cmath>
#include <iostream>
#include <stdlib.h>
using namespace std;

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
class NeuralNetwork{
    public:
        //default constructor sets PSI, HPSI, and arrays to zero
        NeuralNetwork();
        
        //sets weights and biases using double array from GeneticArray class
        void setParameters(double parameters[]);
        
        //return the fitness function of the neural network. M is the number of random point evaluations
        //within the symmetric bounds -xBound < x < xBound.
        double evaluateR(double rp[], int M);
        
        //evaluates the NN at x and redefines PSI and HPSI
        void evaluate(double x);
        
        //get PSI at x
        double getPSI(double x);
        
        //get HSPI(energy) at x
        double getHPSI(double x);
        
        //update parameters using back-propagation at M random points in rp
        void backpropagate(double rp[], int M, double alpha);
        
        //evaluate derivative of R wrt b_j(where j goes from 0 to 5)
        double evaluate_dRdb_j(double rp[], int M, int j);
        
        //evaluate derivative of R wrt w_j
        double evaluate_dRdw_j(double rp[], int M, int j);
        
        //evaluate derivative of R wrt w_jk
        double evaluate_dRdw_jk(double rp[], int M, int j);
        
        //evaluate derivative of R wrt b_k
        double evaluate_dRdb_k(double rp[], int M, int j);
        
    private:
        //energy is fixed and is known eigen state of hamiltonian
        const double E = 0.5;
        
        //input to hidden layer weights and biases
        //w_j = [w_0, w_1, w_2,...] are weights from input to hidden layer
        //b_j = [b_0, b_1, b_2,...] are biases from input to hidden layer
        double b_j[6];
        double w_j[6];
        
        //hidden to output layer weights and biases
        //w_jk = [w_00, w_01, w_10, w_11,...] where w_01 denotes weight from 1st hidden neuron to second output neuron
        //b_k = [b_1, b_2] are biases from hidden to output layer
        double w_jk[12];
        double b_k[2];
        
        //store PSI and the hamiltonian of PSI
        double PSI;
        double HPSI;
        
        //store o_k and its derivatives wrt x
        double S, A, dS, dA, d2A, d2S;
    
};

#endif
