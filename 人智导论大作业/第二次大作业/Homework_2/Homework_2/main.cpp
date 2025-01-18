//
//  main.cpp
//  Homework_2
//
//  Created by apple on 2023/11/24.
//

// Assignment2.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <ctime>
#include "Queen.hpp"
#include "Csp.hpp"
#include "search.hpp"

Csp initCsp(int size)
{
    std::vector<Queen*> variables;
    for (int i = 0; i < size; i++)
    {
        Queen* queen = new Queen(i + 1, size);
        variables.push_back(queen);
    }

    return Csp(variables);
}

// backtrackExperiment和minConflictExperiment只是样例实验，同学可以根据需求自己实现实验函数。
void backtrackExperiment(int maxPuzzleSize)
{
    std::cout << "Backtrack Search Experiment Begin." << std::endl;
    for (int size = 1; size <= maxPuzzleSize; size++)
    {
        std::cout << "Current Size: " << size << std::endl;
        Csp csp = initCsp(size);
        std::vector<Queen*> solution = search::backtrackingSearch(csp);
        search::printSolution(solution);
    }
}
// 由于该算法存在随机性，偶尔存在无法在maxStep范围内找到结果的情况属于正常现象
void minConflictExperiment(int puzzleSize, int iteration, int maxStep)
{
    std::cout << "MinConflict Search Experiment Begin." << std::endl;
    for (int it = 0; it < iteration; it++)
    {
        std::cout << "Iteration: " << it << " begin" << std::endl;
        Csp csp = initCsp(puzzleSize);
        csp.randomAssign();

        std::vector<Queen*> solution = search::minConflict(csp, maxStep);
        search::printSolution(solution);
    }
}
// 搜索函数结果最后通过该函数测试
void searchTest(search::searchFunc searchFunc)
{
    std::cout << "Search Test Begin." << std::endl;
    for (int i = 0; i < 100; i++)
    {
        Csp csp = initCsp(8);
        std::vector<Queen*> solution = searchFunc(csp);
        if (search::failed(solution))
        {
            std::cout << "Iteration " << i << ": failed!" << std::endl;
        }
        else
        {
            std::cout << "Iteration " << i << ": " << search::isSolution(csp, solution) << std::endl;
        }
    }
}

int main(int argc, const char* argv[])
{
    std::srand((unsigned int)std::time(NULL)); // Generate a random seed

    backtrackExperiment(16);
    //minConflictExperiment(8, 100, 200);
    //searchTest(search::backtrackingSearch);
    //searchTest(search::minConflictWrapper);

    system("pause");
    return 0;
}
