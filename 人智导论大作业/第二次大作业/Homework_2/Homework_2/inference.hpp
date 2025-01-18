//
//  inference.hpp
//  
//
//  Created by apple on 2023/11/24.
//
#pragma once

#include "Csp.hpp"
#include "Queen.hpp"
// ac-3算法，用于推理，缩小variables的domian。
namespace inference
{
    bool canSatisfy(Csp& csp, Position& p1, Queen& q2);
    bool revise(Csp& csp, Queen& q1, Queen& q2);
    bool ac3(Csp& csp);
}
