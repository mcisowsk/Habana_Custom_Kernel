/**********************************************************************
Copyright (c) 2021 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef ADD_F32_TEST_HPP
#define ADD_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "add_f32.hpp"

class AddF32Test : public TestBase
{
public:
    AddF32Test() {}
    ~AddF32Test() {}
    int runTest(); // runs whole test end-to-end
    std::vector<float_5DTensor> prepareTestInputs(); // TODO sending this over to all processes would require sending data, then shape, then recreating the tensors.......
    std::vector<float> Compute(/*std::vector<std::vector<float>> inputs,*/ int seed, Gaudi_Kernel_Name_e NameofKernel = GAUDI_KERNEL_MAX_EXAMPLE_KERNEL);

    inline static void addf32_reference_implementation(
            const float_5DTensor& input0,
            const float_5DTensor& input1,
            float_5DTensor& output);
private:
    AddF32Test(const AddF32Test& other) = delete;
    AddF32Test& operator=(const AddF32Test& other) = delete;

};


#endif /* ADD_F32_TEST_HPP */

