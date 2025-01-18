//
//  relation.hpp
//  Homework_2
//
//  Created by apple on 2023/11/24.
//
#pragma once

#include "Queen.hpp"
// 对应CSP中的constraint
namespace relation
{
    typedef bool (*relationFunc)(Position p1, Position p2);
    bool conflict(Position p1, Position p2);
}
