#include "GeneticArray.hpp"
#include "NeuralNetwork.hpp"
#include <typeinfo>
#include <fstream>

//population stage: keeps array at fittestIdx and randomly generates the rest of the population
//*GOOD*
void generatePopulation(GeneticArray population[], GeneticArray parents[], int fittestIdx, int N){
    population[N/2] = population[fittestIdx];
    for(int i = 0; i < N; i++){
        if(i != N/2){
            population[i].randomize();
        }
    }
}

//competition stage: pick 25 pairs of arrays 
//*GOOD*
void compete(GeneticArray population[], GeneticArray parents[], bool isParent[], bool isAlive[], int N, int M, double rp[]){
    int Idx1, Idx2, n = 0;
    for(int i = 0; i < N/2; i++){
        Idx1 = i;
        Idx2 = i + N/2;
        if(population[Idx1].getR(rp, M) <= population[Idx2].getR(rp, M)){
            isAlive[Idx2] = false; //loser dies
            parents[n] = population[Idx1]; //winner becomes parent
            isParent[Idx1] = true; //denote winner as parent
            n++;
        }else{
            isAlive[Idx1] = false;
            parents[n] = population[Idx2];
            isParent[Idx2] = true;
            n++;
        }
    }
}

//mating stage: pick N/2 random pairs of parent arrays(arrays that survived competition stage) and repopulate with offspring
//*GOOD*
void mate(GeneticArray population[], GeneticArray parents[], bool isParent[], int N, int M, double rp[]){
    //fill population with offspring
    int n = 0, randIdx1, randIdx2;
    for(int i = 1; i < N/2; i++){
        randIdx1 = rand() % (N/2);
        randIdx2 = rand() % (N/2);
        while(randIdx2 == randIdx1){
            randIdx2 = rand() % (N/2);
        }
        population[i] = parents[randIdx1] + parents[randIdx2];
    }
    
    for(int i = 0; i < N/2; i++){
        population[i+N/2] = parents[i];
    }

}

//return index of fittest array(one with lowest R value)
//*GOOD*
int findFittestIdx(GeneticArray population[], int N, int M, double rp[]){
    int minIdx = 0;
    double minVal = population[0].getR(rp, M);
    for(int i = 0; i < N; i++){
        if(population[i].getR(rp, M) < minVal){
            minVal = population[i].getR(rp, M);
            minIdx = i;
        }
    }
    
    return minIdx;
}

//calculate the convergence of the generation, D, by calculating the average bitwise difference between fittest array
//and rest of population = (total number of bit differences) / (total number of bits)
//*GOOD*
double calculateD(GeneticArray population[], int fittestIdx, int N){
    double sum = 0.0;
    for(int i = 0; i < N; i++){
        sum += population[i] - population[fittestIdx];
    }
    
    return sum / (832 * (N-1));
}

//generate M random points between -xBound and xBound
//*GOOD*
void generateRP(double rp[], double xBound, int M){
    for(int i = 0; i < M; i++){
        rp[i] = 2 * xBound * (double)rand() / RAND_MAX - xBound;
    }
}

//*****change the private variable, E, in NeuralNetwork.hpp to change energy state of interest******
int main(){
    //pretty good seed for zeroth energy state: 1608089581
    srand(1608089581);
    int N = 50, M = 100, fittestIdx = -1;//N is population size, M is # of random point evaluations
    double D = 1, bestR = 100, xBound = 5, rp[M];//D is convergence of population, rp[] is array of points of evaluation
    bool isParent[N], isAlive[N];//keeps track of indices of parents and living arrays during competition stage
    GeneticArray population[N];
    GeneticArray parents[N/2];
    
//GENETIC ALGORITHM----------------------------------------------------------------------------------------------------------
    int generations = 0;
    ofstream myFS;
    myFS.open("generation_plot.txt");
    //it's best to just loop for some large number, since the GA needs time to sort out badly shaped functions
    for(int i = 0; i < 1000; i++){
        //these arrays keep track of loser and parent arrays during competition stage
        for(int i = 0; i < N; i++){
            isParent[i] = false;
            isAlive[i] = true;
        }
        
        //generate M random values between -xBound and xBound
        generateRP(rp, xBound, M);
        
        //generate population, keep array at fittestIdx
        generatePopulation(population, parents, fittestIdx, N);
        
        //randomly pit N/2 pairs against each other, take survivors to be parents
        compete(population, parents, isParent, isAlive, N, M, rp);
        
        //produce N/2 offspring from parents to repopulate population
        mate(population, parents, isParent, N, M, rp);
        
        //calculate convergence as average bit diff between fittest array and rest of population
        fittestIdx = findFittestIdx(population, N, M, rp);
        D = calculateD(population, fittestIdx, N);
        //go through competition and mating stage until population converges
        while(D > 0.05){
            generateRP(rp, xBound, M);
            for(int i = 0; i < N; i++){
                isParent[i] = false;
                isAlive[i] = true;
            }
            compete(population, parents, isParent, isAlive, N, M, rp);
            mate(population, parents, isParent, N, M, rp);
            fittestIdx = findFittestIdx(population, N, M, rp);
            D = calculateD(population, fittestIdx, N);
            generations++;
        }
        bestR = population[fittestIdx].getR(rp, M);
        myFS << generations << " " << bestR << endl;
        cout << "i = " << i << endl;
        cout << "R = " << bestR << endl;
    }
    myFS.close();
//---------------------------------------------------------------------------------------------------------------------------
    
    //once loop finishes, take best array and use it to plot PSI
    ofstream myFS2;
    myFS2.open("preoptimized_PSI.txt");
    double dx = (2 * xBound) / 100, x;
    for(int i = 0; i < 100; i++){
        x = i * dx - xBound;
        myFS2 << x << " " << population[fittestIdx].getPSI(x) << endl;
    }
    myFS2.close();
    
    //save binary array of best array to file
    ofstream myFS3;
    myFS3.open("best_array.txt");
    for(int i = 0; i < 832; i++){
        myFS3 << population[fittestIdx].getBinary(i) << endl;
    }
    myFS3.close();
    
//BACK-PROPAGATION-----------------------------------------------------------------------------------------------------------
    //implement back propagation for neural network of best array
    cout << "optimization stage: " << endl;
    double alpha = 10e-30; //learning rate, alpha, tends to be very small
    for(int i = 0; i < 100; i++){
        population[fittestIdx].optimize(rp, M, alpha);
        cout << "optimized R = " << population[fittestIdx].getR(rp, M) << endl;
    }
//---------------------------------------------------------------------------------------------------------------------------
    
    //save optimized wave function
    ofstream myFS4;
    myFS4.open("optimized_PSI.txt");
    for(int i = 0; i < 100; i++){
        x = i * dx - xBound;
        myFS4 << x << " " << population[fittestIdx].getPSI(x) << endl;
    }
    myFS4.close();
    
    return 0;
}
