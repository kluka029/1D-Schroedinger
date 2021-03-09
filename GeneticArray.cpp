#include "GeneticArray.hpp"

//constructor initializes binary array randomly, calculates double array and passes double array data to NN
//*GOOD*
GeneticArray::GeneticArray(){
    //randomly initialize binary array
    for(int i = 0; i < nBits; i++){
        binaryArray[i] = rand() % 2;
    }
    R = -1;
    calculateDouble();
    NN.setParameters(doubleArray);
}

//constructor initializes binary array to initBinaryArray, calculates double array
//*GOOD*
GeneticArray::GeneticArray(int initBinaryArray[]){
    for(int i = 0; i < nBits; i++){
        binaryArray[i] = initBinaryArray[i];
    }
    R = -1;
    calculateDouble();
    NN.setParameters(doubleArray);
}

//copy constructor
GeneticArray::GeneticArray(const GeneticArray& other){
    for(int i = 0; i < nBits; i++){
        binaryArray[i] = other.binaryArray[i];
    }
    for(int i = 0; i < nDouble; i++){
        doubleArray[i] = other.doubleArray[i];
    }
    NN.setParameters(doubleArray);
}

//randomizes array(same thing as constructor)
void GeneticArray::randomize(){
    //randomly initialize binary array
    for(int i = 0; i < nBits; i++){
        binaryArray[i] = rand() % 2;
    }
    R = -1;
    calculateDouble();
    NN.setParameters(doubleArray);
}

//assignment operator
//*OKAY*
GeneticArray GeneticArray::operator=(const GeneticArray& other){
    for(int i = 0; i < nBits; i++){
        binaryArray[i] = other.binaryArray[i];
    }
    
    for(int i = 0; i < nDouble; i++){
        doubleArray[i] = other.doubleArray[i];
    }
    NN.setParameters(doubleArray);
    
    return *this;
}

//returns offspring of two parents
//*OKAY*
GeneticArray GeneticArray::operator+(const GeneticArray& other){
    //choose random number of crossing points
    int N = rand() % nBits;
    int idx, prevIndices[N], initBinaryArray[nBits];
    double crossProb = 0.1; //probability of crossing
    
    //copy values of GeneticArray
    for(int i = 0; i < nBits; i++){
        if((double)rand()/RAND_MAX < crossProb){
            initBinaryArray[i] = other.binaryArray[i];
        }else{
            initBinaryArray[i] = binaryArray[i];
        }
    }

    return GeneticArray(initBinaryArray);
}

//returns bit difference between two genetic arrays
//*GOOD*
int GeneticArray::operator-(GeneticArray& other){
    int sum = 0;
    for(int i = 0; i < nBits; i++){
        if(binaryArray[i] != other.getBinary(i)){
            sum++;
        }
    }
    
    return sum;
}

//calculate double array from binary array.
//*GOOD*
void GeneticArray::calculateDouble(){
    //calculate corresponding decimal values
    for(int i = 0; i < nDouble; i++){
        double mantissa = 0, exponent = 0, sign;
        sign = pow(-1.0, binaryArray[i*32]);
        
        for(int j = 1; j <= 8; j++){
            exponent += pow(2.0, 8 - j) * binaryArray[i*32 + j];  
        }
        
        for(int j = 9; j <= 31; j++){
            mantissa += pow(2.0, -(j - 8)) * binaryArray[i*32 + j];
        }
        
        doubleArray[i] = sign * (1 + mantissa) * pow(2.0, exponent - 127);
    }
}

//return PSI
//*GOOD*
double GeneticArray::getPSI(double x){
    return NN.getPSI(x);
}

//return element of double array at idx
//*GOOD*
double GeneticArray::getDouble(int idx) const{
    return doubleArray[idx];
}

//return element of binary array at idx
//*GOOD*
int GeneticArray::getBinary(int idx) const{
    return binaryArray[idx];
}

//evaluates and returns fitness, R
double GeneticArray::getR(double rp[], int M){
    R = NN.evaluateR(rp, M);
    return R;
}

//optimizes neural network using back propagation;
void GeneticArray::optimize(double rp[], int M, double alpha){
    NN.backpropagate(rp, M, alpha);
}
