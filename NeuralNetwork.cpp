#include "NeuralNetwork.hpp"
#include <math.h>

//default constructor sets PSI, HPSI, and arrays to zero
//*GOOD*
NeuralNetwork::NeuralNetwork(){
    for(int i = 0; i < 6; i++){
        w_j[i] = 0;
        b_j[i] = 0;
    }
    for(int i = 0; i < 12; i++){
        w_jk[i] = 0;
    }
    
    for(int i = 0; i < 2; i++){
        b_k[i] = 0;
    }
    
    PSI = 0;
    HPSI = 0;
}

//sets weights and biases using double array from GeneticArray class
//*GOOD*
void NeuralNetwork::setParameters(double parameters[]){
    for(int i = 0; i < 26; i++){
        if(i < 6){
            w_j[i] = parameters[i];
        }
        else if(i >= 6 && i < 12){
            b_j[i - 6] = parameters[i];
        }
        else if(i >= 12 && i < 24){
            w_jk[i - 12] = parameters[i];
        }
        else if(i >= 24){
            b_k[i - 24] = parameters[i];
        }
    }
}

//return the fitness function of the neural network. M is the number of random point evaluations
//within the symmetric bounds -xBound < x < xBound.
//*GOOD*
double NeuralNetwork::evaluateR(double rp[], int M){
    double topSum = 0, bottomSum = 0, x;
    for(int i = 0; i < M; i++){
        x = rp[i];
        evaluate(x);
        topSum += pow(HPSI - E*PSI, 2);
        bottomSum += pow(PSI, 2);
    }
    
    return topSum / bottomSum;
}

//these functions exist to take the limit when the sigmoid function(and derivatives) seems to approach infinity
//*GOOD*
double sigma(double n_j){
    double val;
    if((1 - exp(-n_j)) == -INFINITY && (1 + exp(-n_j)) == INFINITY){
        val = -1;
    }else{
        val = (1 - exp(-n_j)) / (1 + exp(-n_j));
    }
    
    return val;
}

double dsigma(double n_j){
    double val;
    if(2*exp(n_j) == INFINITY && pow(1 + exp(n_j),2) == INFINITY){
        val = 0;
    }else{
        val = (2*exp(n_j)) / pow(1 + exp(n_j),2);
    }
    
    return val;
}

double d2sigma(double n_j){
    double val;
    if(-2*exp(n_j) * (exp(n_j) - 1) == -INFINITY && pow(1 + exp(n_j),3) == INFINITY){
        val = -2;
    }else{
        val = (-2*exp(n_j) * (exp(n_j) - 1) / pow(1 + exp(n_j),3));
    }
    
    return val;
}

double d3sigma(double n_j){
    double val;
    if(isnan(2*exp(n_j)*(1 - 4*exp(n_j) + exp(2*n_j))) && pow(1+exp(n_j),4) == INFINITY){
        val = 0;
    }else{
        val = 2*exp(n_j)*(1 - 4*exp(n_j) + exp(2*n_j)) / pow(1+exp(n_j),4);
    }
    
    return val;
}
//************************************************************************************

//evaluates the NN at x
//*GOOD*
void NeuralNetwork::evaluate(double x){
    double n_j;
    S = 0;
    A = 0;
    dS = 0;
    dA = 0;
    d2S = 0;
    d2A = 0;
    //calculate A, S and their derivatives wrt x to find PSI and HSPI
    for(int i = 0; i < 6; i++){
        n_j = w_j[i] * x + b_j[i];
        A += w_jk[2*i] * sigma(n_j) + b_k[0];
        S += w_jk[2*i + 1] * sigma(n_j) + b_k[1];
        dA += w_jk[2*i] * dsigma(n_j) * w_j[i];
        dS += w_jk[2*i + 1] * dsigma(n_j) * w_j[i];
        d2A += w_jk[2*i] * d2sigma(n_j) * pow(w_j[i],2);
        d2S += w_jk[2*i + 1] * d2sigma(n_j) * pow(w_j[i],2);
    }
    PSI = A * sin(S);
    HPSI = -0.5*(sin(S)*(-A*pow(dS,2) + d2A) + cos(S)*(A*d2S + 2*dA*dS)) + 0.5*pow(x,2)*A*sin(S);
    
}

//evaluate and return PSI
//*GOOD*
double NeuralNetwork::getPSI(double x){
    evaluate(x);
    return PSI;
}

//get HSPI(energy) at x
double NeuralNetwork::getHPSI(double x){
    evaluate(x);
    return HPSI;
}

//update parameters using back-propagation
void NeuralNetwork::backpropagate(double rp[], int M, double alpha){
    //you need to have separate loops for dR and when you redefine the parameters since changing the parameters in the
    //same loop would mean that PSI is evaluated using the ith weights and biases at the NEXT time step.
    double dRdb_j[6], dRdw_j[6], dRdw_jk[12], dRdb_k[2];
    
    //calculate dR
    for(int i = 0; i < 6; i++){
        dRdb_j[i] = evaluate_dRdb_j(rp, M, i);
        dRdw_j[i] = evaluate_dRdw_j(rp, M, i);
    }
    
    int j[] = {0,0,1,1,2,2,3,3,4,4,5,5};
    for(int i = 0; i < 12; i++){
        dRdw_jk[i] = evaluate_dRdw_jk(rp, M, j[i]);
    }
    
    for(int i = 0; i < 2; i++){
        dRdb_k[i] = evaluate_dRdb_k(rp, M, i);
    }
    
    //calculate new parameters
    for(int i = 0; i < 6; i++){
        b_j[i] = b_j[i] - alpha*dRdb_j[i];
        w_j[i] = w_j[i] - alpha*dRdw_j[i];
    }
    
    for(int i = 0; i < 12; i++){
        w_jk[i] = w_jk[i] - alpha*dRdw_jk[i];
    }
    
    for(int i = 0; i < 2; i++){
        b_k[i] = b_k[i] - alpha*dRdb_k[i];
    }
    
}

//evaluate derivative of R wrt b_j(where j goes from 0 to 5)
//*OKAY*
double NeuralNetwork::evaluate_dRdb_j(double rp[], int M, int j){
    //for each x, calculate and add to top, bottom, dtop, and dbottom
    //return dR = (bottom*dtop - top*dbottom) / bottom^2
    double x;
    double topSum = 0, bottomSum = 0, dtopSum = 0, dbottomSum = 0;
    double S_theta, A_theta, dA_theta, dS_theta, d2A_theta, d2S_theta;
    double w_j1 = w_jk[2*j + 1], w_j0 = w_jk[2*j];
    double n_j;
    
    for(int i = 0; i < M; i++){
        x = rp[i];
        n_j = w_j[j] * x + b_j[j];
        evaluate(x); //gets PSI, HPSI, and o_k and its derivatives
        //easily get top and bottom sums
        topSum += pow(HPSI - E*PSI,2);
        bottomSum += pow(PSI,2);
        
        //compute dtop and dbottom sums
        S_theta = w_j1 * dsigma(n_j);
        A_theta = w_j0 * dsigma(n_j);
        dS_theta = w_j1 * d2sigma(n_j) * w_j[j];
        dA_theta = w_j0 * d2sigma(n_j) * w_j[j];
        d2S_theta = w_j1 * d3sigma(n_j) * pow(w_j[j],2);
        d2A_theta = w_j0 * d3sigma(n_j) * pow(w_j[j],2);
        
        /*
        dtopSum += cos(S)*(S_theta*(A*pow(dS,2) - d2A) - A_theta*d2S - A*d2S_theta - 2*(dA_theta*dS + dA*dS_theta) - 0.5*pow(x,2)*dS_theta) + sin(S)*(A_theta*pow(dS,2) + 2*A*dS*dS_theta - d2A_theta + S_theta*(A*d2S + 2*dA*dS) - 0.5*pow(x,2)*A_theta);
        */
        
        dtopSum += 2*(HPSI - E*PSI)*( -sin(S)*(-2*A*dS*dS_theta - A_theta*pow(dS,2) + d2A_theta) 
        - cos(S)*S_theta*(-A*pow(dS,2) + d2A) - cos(S)*(A*d2S_theta + A_theta*d2S + 2*dA_theta*dS + 2*dA*dS_theta)
        + sin(S)*S_theta*(A*d2S + 2*dA*dS) + (0.5*pow(x,2) - E)*(A*cos(S)*S_theta + A_theta*sin(S))
        );
        
        dbottomSum += (2*A*A_theta*pow(sin(S),2) + 2*pow(A,2)*S_theta*sin(S)*cos(S));
    }
        
    return (bottomSum*dtopSum - topSum*dbottomSum) / pow(bottomSum,2);
}

//evaluate derivative of R wrt w_j
//*OKAY*
double NeuralNetwork::evaluate_dRdw_j(double rp[], int M, int j){
    double x;
    double topSum = 0, bottomSum = 0, dtopSum = 0, dbottomSum = 0;
    double S_theta, A_theta, dA_theta, dS_theta, d2A_theta, d2S_theta;
    double w_j1 = w_jk[2*j + 1], w_j0 = w_jk[2*j];
    double n_j;
    for(int i = 0; i < M; i++){
        x = rp[i];
        n_j = w_j[j] * x + b_j[j];
        evaluate(x); //gets PSI, HPSI, and o_k and its derivatives
        //easily get top and bottom sums
        topSum += pow(HPSI - E*PSI,2);
        bottomSum += pow(PSI,2);
        
        //compute dtop and dbottom sums
        S_theta = w_j1 * dsigma(n_j) * x;
        A_theta = w_j0 * dsigma(n_j) * x;
        dS_theta = w_j1 * (d2sigma(n_j)*x*w_j[j] + dsigma(n_j));
        dA_theta = w_j0 * (d2sigma(n_j)*x*w_j[j] + dsigma(n_j));
        d2S_theta = w_j1 * (d3sigma(n_j)*x*pow(w_j[j],2) + d2sigma(n_j)*2*w_j[j]);
        d2A_theta = w_j0 * (d3sigma(n_j)*x*pow(w_j[j],2) + d2sigma(n_j)*2*w_j[j]);
        
        /*
        dtopSum += cos(S)*(S_theta*(A*pow(dS,2) - d2A) - A_theta*d2S - A*d2S_theta - 2*(dA_theta*dS + dA*dS_theta) - 0.5*pow(x,2)*dS_theta) + sin(S)*(A_theta*pow(dS,2) + 2*A*dS*dS_theta - d2A_theta + S_theta*(A*d2S + 2*dA*dS) - 0.5*pow(x,2)*A_theta);
        */
        
        dtopSum += 2*(HPSI - E*PSI)*( -sin(S)*(-2*A*dS*dS_theta - A_theta*pow(dS,2) + d2A_theta) 
        - cos(S)*S_theta*(-A*pow(dS,2) + d2A) - cos(S)*(A*d2S_theta + A_theta*d2S + 2*dA_theta*dS + 2*dA*dS_theta)
        + sin(S)*S_theta*(A*d2S + 2*dA*dS) + (0.5*pow(x,2) - E)*(A*cos(S)*S_theta + A_theta*sin(S))
        );
        
        dbottomSum += (2*A*A_theta*pow(sin(S),2) + 2*pow(A,2)*S_theta*sin(S)*cos(S));
    }
    
    
    return (bottomSum*dtopSum - topSum*dbottomSum) / pow(bottomSum,2);
}

//evaluate derivative of R wrt w_jk
//*OKAY*
double NeuralNetwork::evaluate_dRdw_jk(double rp[], int M, int j){
    double x;
    double topSum = 0, bottomSum = 0, dtopSum = 0, dbottomSum = 0;
    double S_theta, A_theta, dA_theta, dS_theta, d2A_theta, d2S_theta;
    double w_j1 = w_jk[2*j+1], w_j0 = w_jk[2*j];
    double n_j;
    for(int i = 0; i < M; i++){
        x = rp[i];
        n_j = w_j[j] * x + b_j[j];
        evaluate(x); //gets PSI, HPSI, and o_k and its derivatives
        //easily get top and bottom sums
        topSum += pow(HPSI - E*PSI,2);
        bottomSum += pow(PSI,2);
        
        //compute dtop and dbottom sums
        S_theta = sigma(n_j);
        A_theta = dsigma(n_j);
        dS_theta = dsigma(n_j)*w_j[j];
        dA_theta = dsigma(n_j)*w_j[j];
        d2S_theta = d2sigma(n_j)*pow(w_j[j],2);
        d2A_theta = d2sigma(n_j)*pow(w_j[j],2);
        
        /*
        dtopSum += cos(S)*(S_theta*(A*pow(dS,2) - d2A) - A_theta*d2S - A*d2S_theta - 2*(dA_theta*dS + dA*dS_theta) - 0.5*pow(x,2)*dS_theta) + sin(S)*(A_theta*pow(dS,2) + 2*A*dS*dS_theta - d2A_theta + S_theta*(A*d2S + 2*dA*dS) - 0.5*pow(x,2)*A_theta);
        */
        
        dtopSum += 2*(HPSI - E*PSI)*( -sin(S)*(-2*A*dS*dS_theta - A_theta*pow(dS,2) + d2A_theta) 
        - cos(S)*S_theta*(-A*pow(dS,2) + d2A) - cos(S)*(A*d2S_theta + A_theta*d2S + 2*dA_theta*dS + 2*dA*dS_theta)
        + sin(S)*S_theta*(A*d2S + 2*dA*dS) + (0.5*pow(x,2) - E)*(A*cos(S)*S_theta + A_theta*sin(S))
        );
        
        dbottomSum += (2*A*A_theta*pow(sin(S),2) + 2*pow(A,2)*S_theta*sin(S)*cos(S));
    }
    
    return (bottomSum*dtopSum - topSum*dbottomSum) / pow(bottomSum,2);
}

//evaluate derivative of R wrt b_k
//*OKAY*
double NeuralNetwork::evaluate_dRdb_k(double rp[], int M, int j){
        double x;
    double topSum = 0, bottomSum = 0, dtopSum = 0, dbottomSum = 0;
    double S_theta, A_theta, dA_theta, dS_theta, d2A_theta, d2S_theta;
    double w_j1 = w_jk[2*j+1], w_j0 = w_jk[2*j];
    double n_j;
    for(int i = 0; i < M; i++){
        x = rp[i];
        n_j = w_j[j] * x + b_j[j];
        evaluate(x); //gets PSI, HPSI, and o_k and its derivatives
        //easily get top and bottom sums
        topSum += pow(HPSI - E*PSI,2);
        bottomSum += pow(PSI,2);
        
        //compute dtop and dbottom sums
        S_theta = 1;
        A_theta = 1;
        dS_theta = 0;
        dA_theta = 0;
        d2S_theta = 0;
        d2A_theta = 0;
        
        /*
        dtopSum += cos(S)*(S_theta*(A*pow(dS,2) - d2A) - A_theta*d2S - A*d2S_theta - 2*(dA_theta*dS + dA*dS_theta) - 0.5*pow(x,2)*dS_theta) + sin(S)*(A_theta*pow(dS,2) + 2*A*dS*dS_theta - d2A_theta + S_theta*(A*d2S + 2*dA*dS) - 0.5*pow(x,2)*A_theta);
        */
        
        dtopSum += 2*(HPSI - E*PSI)*( -sin(S)*(-2*A*dS*dS_theta - A_theta*pow(dS,2) + d2A_theta) 
        - cos(S)*S_theta*(-A*pow(dS,2) + d2A) - cos(S)*(A*d2S_theta + A_theta*d2S + 2*dA_theta*dS + 2*dA*dS_theta)
        + sin(S)*S_theta*(A*d2S + 2*dA*dS) + (0.5*pow(x,2) - E)*(A*cos(S)*S_theta + A_theta*sin(S))
        );
        
        dbottomSum += (2*A*A_theta*pow(sin(S),2) + 2*pow(A,2)*S_theta*sin(S)*cos(S));
    }
    
    
    return (bottomSum*dtopSum - topSum*dbottomSum) / pow(bottomSum,2);
}

